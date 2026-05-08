#!/usr/bin/env python3
"""Test sync and RTC inference paths with dummy observations on CPU.

Usage:
    python test_deploy_paths.py --policy-path /path/to/checkpoint --device cpu
    python test_deploy_paths.py --policy-path /path/to/checkpoint --device cuda

Requires: lerobot with smolvla extras installed.
No robot, cameras, or UDP needed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def resolve_policy_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").exists():
        return path
    checkpoint_pretrained = path / "pretrained_model"
    if (checkpoint_pretrained / "config.json").exists():
        return checkpoint_pretrained
    last_pretrained = path / "checkpoints" / "last" / "pretrained_model"
    if (last_pretrained / "config.json").exists():
        return last_pretrained
    candidates = sorted((path / "checkpoints").glob("*/pretrained_model")) if (path / "checkpoints").exists() else []
    candidates = [c for c in candidates if (c / "config.json").exists()]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Could not find a loadable policy under {path}")


def make_dummy_observation(policy, device: torch.device) -> dict[str, torch.Tensor]:
    """Create a dummy observation dict matching the policy's input features."""
    from lerobot.utils.constants import OBS_STATE

    input_features = policy.config.input_features
    obs = {}

    for key, feature in input_features.items():
        shape = tuple(feature.shape)
        feature_type = getattr(feature.type, "value", str(feature.type))

        if feature_type == "VISUAL":
            # Policy features store images as (C, H, W), but the inference helper
            # expects raw camera images as (H, W, C) before it adds the batch dim.
            if len(shape) != 3:
                raise ValueError(f"Expected visual feature {key!r} to have shape (C, H, W), got {shape}")
            c, h, w = shape
            obs[key] = np.random.randint(0, 256, size=(h, w, c), dtype=np.uint8)
        elif key == OBS_STATE:
            # State vector
            obs[key] = np.random.rand(*shape).astype(np.float32)
        else:
            obs[key] = np.random.rand(*shape).astype(np.float32)

    return obs


def action_sequence_from_chunk(action_chunk: torch.Tensor, name: str) -> torch.Tensor:
    """Return a single action sequence shaped (time_steps, action_dim)."""
    if action_chunk.ndim == 3:
        if int(action_chunk.shape[0]) != 1:
            raise ValueError(f"{name} has batch size {action_chunk.shape[0]}; only batch size 1 is supported")
        return action_chunk.squeeze(0)
    if action_chunk.ndim == 2:
        return action_chunk
    raise ValueError(f"{name} must have shape (1, T, A) or (T, A), got {tuple(action_chunk.shape)}")


def test_sync_path(policy, preprocess, postprocess, device: torch.device, num_steps: int = 5):
    """Test the sync select_action path."""
    print("\n=== Testing SYNC path (select_action) ===")
    policy.reset()

    for step in range(num_steps):
        obs = make_dummy_observation(policy, device)

        from lerobot.policies.utils import prepare_observation_for_inference
        frame = prepare_observation_for_inference(
            obs, device, task="test task", robot_type="franka",
        )

        with torch.inference_mode():
            action_tensor = policy.select_action(preprocess(frame))
            action_tensor = postprocess(action_tensor)

        raw_action = action_tensor.squeeze(0).detach().cpu().numpy()
        print(
            f"  step {step}: action shape={raw_action.shape} "
            f"dtype={raw_action.dtype} "
            f"range=[{raw_action.min():.4f}, {raw_action.max():.4f}]"
        )

    print("  SYNC path: OK")
    return True


def test_rtc_path(
    policy,
    preprocess,
    postprocess,
    device: torch.device,
    num_steps: int = 10,
    rate_hz: float = 30.0,
    rtc_inference_delay: int = 0,
    require_nonempty_leftover: bool = False,
):
    """Test the RTC predict_action_chunk + ActionQueue path."""
    print("\n=== Testing RTC path (predict_action_chunk + ActionQueue) ===")

    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.utils import prepare_observation_for_inference

    # Set up RTC config on the policy
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,  # Will be updated after "measuring" delay
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()

    print(f"  RTC processor initialized: {policy.rtc_processor is not None}")
    print(f"  Model RTC processor: {policy.model.rtc_processor is not None}")
    print(f"  _rtc_enabled(): {policy._rtc_enabled()}")

    if rtc_inference_delay > 0:
        inference_delay = rtc_inference_delay
        print(f"  Using forced inference_delay={inference_delay} steps")
    else:
        # Measure fake inference delay
        print("  Measuring inference delay...")
        obs = make_dummy_observation(policy, device)
        frame = prepare_observation_for_inference(
            obs, device, task="test task", robot_type="franka",
        )

        times = []
        for i in range(3):
            start = time.monotonic()
            with torch.inference_mode():
                _ = policy.predict_action_chunk(preprocess(frame))
            elapsed = time.monotonic() - start
            times.append(elapsed)
            print(f"    warmup {i}: {elapsed*1000:.1f}ms")

        median_s = float(np.median(times))
        step_period_s = 1.0 / rate_hz
        inference_delay = max(1, int(np.ceil(median_s / step_period_s)))
        print(f"  Measured: median={median_s*1000:.1f}ms inference_delay={inference_delay} steps")

    # Set execution_horizon = inference_delay (auto behavior)
    final_execution_horizon = inference_delay
    policy.config.rtc_config.execution_horizon = final_execution_horizon
    action_queue = ActionQueue(policy.config.rtc_config)
    print(f"  ActionQueue created: execution_horizon={final_execution_horizon}")

    # Test: verify ActionQueue API
    print("  Testing ActionQueue API...")
    assert action_queue.empty(), "New queue should be empty"
    assert action_queue.qsize() == 0, "New queue should have qsize 0"
    assert action_queue.get() is None, "get() on empty queue should return None"
    assert action_queue.get_left_over() is None, "get_left_over() on empty queue should return None"
    print("    ActionQueue empty-state API: OK")

    # Run the RTC loop
    rtc_needs_inference = True
    inference_count = 0
    nonempty_leftover_inferences = 0
    empty_leftover_inferences = 0
    for step in range(num_steps):
        obs = make_dummy_observation(policy, device)
        queue_empty = action_queue.empty()

        if rtc_needs_inference or queue_empty:
            frame = prepare_observation_for_inference(
                obs, device, task="test task", robot_type="franka",
            )
            prev_actions = action_queue.get_left_over()

            print(
                f"  step {step}: INFERENCE "
                f"prev_actions={'None' if prev_actions is None else f'shape={tuple(prev_actions.shape)} device={prev_actions.device}'}"
            )
            inference_count += 1
            if prev_actions is not None and prev_actions.numel() > 0:
                nonempty_leftover_inferences += 1
            elif prev_actions is not None:
                empty_leftover_inferences += 1

            action_chunk_raw_batched = policy.predict_action_chunk(
                preprocess(frame),
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )
            with torch.no_grad():
                action_chunk_processed_batched = postprocess(action_chunk_raw_batched.clone())

            action_chunk_raw = action_sequence_from_chunk(action_chunk_raw_batched, "raw RTC action chunk")
            action_chunk_processed = action_sequence_from_chunk(
                action_chunk_processed_batched,
                "processed RTC action chunk",
            )

            print(
                f"           raw: shape={tuple(action_chunk_raw.shape)} device={action_chunk_raw.device} "
                f"dtype={action_chunk_raw.dtype}"
            )
            print(
                f"           processed: shape={tuple(action_chunk_processed.shape)} "
                f"device={action_chunk_processed.device} dtype={action_chunk_processed.dtype}"
            )

            action_queue.merge(action_chunk_raw, action_chunk_processed, inference_delay)
            rtc_needs_inference = False
            print(f"           after merge: qsize={action_queue.qsize()} empty={action_queue.empty()}")

        # Pop one action
        raw_action = action_queue.get()
        if raw_action is not None:
            raw_action_np = raw_action.squeeze(0).detach().cpu().numpy()
            remaining = action_queue.qsize()
            if remaining <= inference_delay:
                rtc_needs_inference = True
            print(
                f"  step {step}: POP action shape={raw_action_np.shape} "
                f"range=[{raw_action_np.min():.4f}, {raw_action_np.max():.4f}] "
                f"remaining={remaining} needs_inference={rtc_needs_inference}"
            )
        else:
            print(f"  step {step}: POP returned None, triggering inference")
            rtc_needs_inference = True

    print(
        "    RTC inference summary: "
        f"inferences={inference_count} "
        f"nonempty_leftover_inferences={nonempty_leftover_inferences} "
        f"empty_leftover_inferences={empty_leftover_inferences}"
    )
    if require_nonempty_leftover and nonempty_leftover_inferences == 0:
        raise AssertionError(
            "Expected at least one RTC inference with non-empty prev_chunk_left_over. "
            "Try --rtc-inference-delay 1 with --rtc-steps 50."
        )

    # Test clear
    action_queue.clear()
    assert action_queue.empty(), "Queue should be empty after clear"
    assert action_queue.qsize() == 0, "Queue qsize should be 0 after clear"
    print("    ActionQueue clear: OK")

    # Disable RTC on policy to leave it clean for other tests
    policy.config.rtc_config = None
    policy.init_rtc_processor()

    print("  RTC path: OK")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sync-steps", type=int, default=5)
    parser.add_argument("--rtc-steps", type=int, default=10)
    parser.add_argument("--rtc-rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--rtc-inference-delay",
        type=int,
        default=0,
        help=(
            "Override measured RTC inference delay in action steps. "
            "Useful on CPU to force non-empty leftovers, e.g. --rtc-inference-delay 1 --rtc-steps 50."
        ),
    )
    parser.add_argument(
        "--require-nonempty-rtc-leftover",
        action="store_true",
        help="Fail unless at least one RTC inference receives non-empty prev_chunk_left_over.",
    )
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-rtc", action="store_true")
    args = parser.parse_args()

    # Set up lerobot imports
    if args.lerobot_root is not None:
        lerobot_src = args.lerobot_root.expanduser() / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
    else:
        # Try to find lerobot
        script_dir = Path(__file__).resolve().parent
        for parent in script_dir.parents:
            candidate = parent / "lerobot" / "src"
            if candidate.exists():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                break

    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors

    device = torch.device(args.device)
    policy_path = str(resolve_policy_path(args.policy_path))

    print(f"Loading SmolVLA from {policy_path} on {device}...")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    print(f"  chunk_size={policy.config.chunk_size}")
    print(f"  n_action_steps={policy.config.n_action_steps}")
    print(f"  max_action_dim={policy.config.max_action_dim}")
    print(f"  max_state_dim={policy.config.max_state_dim}")
    print(f"  num_steps (flow matching)={policy.config.num_steps}")
    print(f"  input_features: {list(policy.config.input_features.keys())}")
    print(f"  output_features: {list(policy.config.output_features.keys())}")

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    results = {}

    if not args.skip_sync:
        try:
            results["sync"] = test_sync_path(policy, preprocess, postprocess, device, args.sync_steps)
        except Exception:
            print(f"\n  SYNC path FAILED:")
            traceback.print_exc()
            results["sync"] = False

    if not args.skip_rtc:
        try:
            results["rtc"] = test_rtc_path(
                policy,
                preprocess,
                postprocess,
                device,
                num_steps=args.rtc_steps,
                rate_hz=args.rtc_rate_hz,
                rtc_inference_delay=args.rtc_inference_delay,
                require_nonempty_leftover=args.require_nonempty_rtc_leftover,
            )
        except Exception:
            print(f"\n  RTC path FAILED:")
            traceback.print_exc()
            results["rtc"] = False

    print("\n=== RESULTS ===")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if not results:
        print("  No tests run.")
        return 2

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
