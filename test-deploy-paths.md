  - --device cpu: works now, but measured RTC delay is unrealistic because CPU inference is very slow. Use forced delay for the best shape/
    overlap smoke test:

    python franka_xr_teleop/tools/test_deploy_paths.py \
      --policy-path pretrained_model \
      --lerobot-root ../lerobot \
      --device cpu \
      --sync-steps 1 \
      --rtc-steps 50 \
      --rtc-inference-delay 1 \
      --require-nonempty-rtc-leftover

  - --device cuda: should work if CUDA is available in the environment. This is more representative for real deployment timing:

    python franka_xr_teleop/tools/test_deploy_paths.py \
      --policy-path pretrained_model \
      --lerobot-root ../lerobot \
      --device cuda \
      --sync-steps 1 \
      --rtc-steps 50 \
      --require-nonempty-rtc-leftover
