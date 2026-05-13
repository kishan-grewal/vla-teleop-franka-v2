SVLA_1: datasets 200-204
SVLA_2: datasets 200-209
SVLA_3: datasets 205-209
SVLA_4: datasets 200-209, penultimate checkpoint
SVLA_5: 200-209, reduced labels
SVLA_6: 205-209, reduced labels
act_1: datasets 200-209


SVLA_3, rtc on, ema_alpha 0.6:

"insert the orange MSD connector inside the orange socket"
- Decent, generally grasped the back half of the plug

"push the plug fully into the port"
- Consistently much too low grab, was grasping the body of the plug

"push the orange plug fully into the blue block port"
- same as above

"push the plug into the port"
- couple of very good grasps, less good at final positioning?
- then started going low + forward

"Drive the orange connector inside the slot"
- low grasps, fairly centred
"
push the orange MSD connector into the port"
- rear grasps, plug was generally forward of the socket though it did track backward, without full insertion

"push the orange MSD connector inside the slot"
- hesitant on grasp, high grasps

"push the orange MSD connector securely into the socket"
- SUCCESS
- Generally good grasp positions, not always commital
- When rtc disabled, excellent grasps but not high enough to clear the socket edge

SVLA_2, rtc on, ema_alpha 0.6:

"push the orange MSD connector securely into the socket"
- inconsistent grasp behaviour
- grasps are either forward or central
- when it gets it right, its able to get close to an insertion, but doesnt time releases correctly
  - it adjusts back and forth seemingly semi-randomly, does hit good release positions, but fails to

ACT_2

ema 0.6
- good, initial grasp was excellent, got over the socket but couldnt position in fully
- other attempts grasped too high + travelled too low to make it over the socket wall

ema 0.8
- no noticeable smoothness change
- same too high+too low combination of grasp/travel

ema 1.0
- slight noticeable change in the judders - present in all variants but least damped here
- one successful insertion - got over in the same was as the 0.6 case, judder pushed it in
  - didnt look like comprehensive control, more like luck - cycled the judder several times
    at regular intervals, then happened to slot in
- further testing with more positional range showed it can grasp well in a small region
  - outside that, it either completely misses or grasps the edge of the handle
  - this then causes it to completely miss the target, 
    - misjudging it (grasped from the front) and bending the handle
    - freezing on approach (grasped from the back)

Async chunk tests 1
- ACT
  - Increased jerk severity
- SmolVLA
  - Introduced significant oscillation/jerky behaviour which destabilised the pathing
(rolled back, present on origin/feat/async-chunking)