from __future__ import annotations

import os

from math import cos, pi

import torch
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz

from .base import BaseTask
from .task_utils import heading_yaw_from_quat, to_heading_frame


@configclass
class CombinedTaskCfg:
    mode_probabilities = (0.25, 0.25, 0.25, 0.25)
    landing_to_takeoff_fraction = 0.20
    # Fraction of ALL randomized resets reserved for a persistent curved DRIVE
    # episode.  This is sampled explicitly below rather than conditionally on
    # first drawing DRIVE, so 0.25 really means 25% of the training batch.
    # Cut from 0.25 to hit a ~7:1 drive:airborne reward ratio.
    #
    # Measured 2026-08-16 at epoch 130: drive terms paid 590 per episode against
    # 23 for airborne, a 26:1 ratio, and the policy rationally stopped leaving
    # the ground -- takeoff success sat at a 0.06% rate while landing collapsed
    # to 8% from the 08-14 run's 82. That earlier run, which did discover both,
    # ran at roughly 4:1.
    #
    # The lever is the number of TRUE PURE DRIVE episodes, not the per-step
    # scales: pure drive episodes are persistent, so they accrue drive reward
    # over the whole episode and dominate the mean. Cutting the fraction to 0.08
    # removes about two thirds of them, which both lowers the drive numerator
    # and raises the airborne denominator, since those resets become flight and
    # takeoff episodes instead.
    pure_drive_fraction = 0.08
    pure_drive_stationary_fraction = 0.30
    pure_drive_straight_fraction = 0.30
    # Reverted to (0.5, 1.5) from a brief (0.0, 1.5) trial. Sampling the
    # zero-error corner is still the right idea -- "already on a stationary
    # reference, do nothing" is a state the policy only extrapolates into -- but
    # it also spawns drive episodes already at the top of the position score,
    # which inflates drive reward against airborne reward, and that balance is
    # the thing currently keeping the policy on the ground. Reinstate it only
    # once the drive:airborne ratio is healthy.
    stationary_drive_error_range_m = (0.5, 1.5)
    # Delayed 150 -> 300. This is the aggressive-randomization switch: it doubles
    # the disturbance scale, forces every non-pure flight reset onto the LANDING
    # route, clamps flight_duration to 4 s and moves the DRIVE share to
    # training_overlay_drive_fraction. All of that at once.
    #
    # The 2026-08-16 run measured the cost: takeoff was not discovered until
    # ~epoch 100 and the behavior did not stabilise until ~600, so the overlay
    # landed at 150 while the policy was still finding the basic rules and it
    # then spent ~1800 epochs grinding out marginal gains. Discovery first,
    # hardening second -- 300 roughly doubles the clean-learning window without
    # deferring robustness to the far end of the run.
    training_overlay_start_epoch = 300.0
    # Drive share once the overlay is active. It used to be a bare 0.5 in
    # reset_initial_state, which meant that at epoch 150 the DRIVE share doubled
    # from 0.25 to 0.50 -- exactly the wrong direction for a policy that is
    # already over-rewarded for staying on the ground, and it would have undone
    # the pure_drive_fraction cut above about ten epochs after it took effect.
    training_overlay_drive_fraction = 0.25
    training_overlay_disturbance_multiplier = 2.0
    # Fraction of ALL randomized resets reserved for persistent FLIGHT.
    pure_flight_fraction = 0.30
    # Recovery envelope based on deployment-scale initial disagreement.  These
    # are reset errors only; the reference path remains continuous.
    pure_drive_position_error_max = 2.0
    pure_drive_yaw_error_max = 45.0 * pi / 180.0
    pure_drive_velocity_error_max = 0.75
    pure_drive_speed_range = (0.45, 1.5)
    drive_reference_height_randomization_m = 0.05
    drive_dynamic_friction_range = (0.08, 0.25)
    drive_static_friction_range = (0.10, 0.35)
    drive_side_friction_spread = 0.15
    pure_drive_lateral_amplitude_range = (3.0, 5.0)
    # At A=5 m, wavelength=28 m and speed=1.5 m/s the crest asks for about
    # 0.058 g, keeping every sampled curve feasible at dynamic friction 0.08.
    pure_drive_lateral_wavelength_range = (28.0, 40.0)
    pure_flight_speed_range = (0.75, 2.25)
    pure_flight_lateral_amplitude_range = (4.0, 6.0)
    # Curvature at a crest is A*k^2, and in the air the lateral accel must come
    # from bank: g*tan(theta). The old band demanded 48 deg of sustained bank at
    # top speed, well outside the 20 deg reset envelope.
    pure_flight_lateral_wavelength_range = (15.6, 31.2)
    pure_flight_vertical_amplitude = 1.0
    pure_flight_vertical_wavelength_range = (24.0, 48.0)
    vertical_trajectory_fraction = 0.50
    drive_duration_range = (1.0, 4.0)
    drive_speed_range = (0.0, 1.0)
    takeoff_duration_range = (2.5, 4.0)
    # Announced takeoff: the mode one-hot flips to TAKEOFF while the reference
    # keeps rolling along the ground for this long before the climb begins. This
    # is the window in which untucking is legitimate rather than a sacrifice of
    # drive tracking. It must cover the slew from full tuck to the liftoff tilt
    # ((pi/2 - takeoff_liftoff_tilt_rad) / morph max velocity), which is what
    # keeps the timing baseline at zero for every starting tilt. Reduced from
    # 3.6 s (which covered the worst-case morph ceiling draw of 0.30 rad/s,
    # 3.49 s of slew) to shrink the window in which the reference rolls along
    # the ground against a deadband-free deviation error. The timing baseline
    # absorbs the difference for nominal-rate draws via the prep subtraction in
    # _takeoff_timing_multiplier; slow ceiling draws now pay some timing bonus
    # they physically cannot avoid, a known trade accepted for the shorter
    # exposure to the deviation wall.
    takeoff_prep_duration_s = 2.0
    # Clip on the phase-event countdown observation. Bounds a long overrun so it
    # cannot dominate the input normaliser's running statistics.
    phase_event_time_clip_s = 5.0
    flight_duration_range = (3.0, 5.0)
    flight_height_range = (1.0, 2.0)
    flight_xy_distance_range = (2.0, 4.0)
    takeoff_end_speed_range = (0.0, 1.0)
    landing_end_speed_range = (0.5, 1.5)
    takeoff_end_heading_offset = 15.0 * pi / 180.0
    hover_attitude_range = 20.0 * pi / 180.0
    morph_balance_recovery_fraction = 0.20
    morph_balance_recovery_max_angle = 5.0 * pi / 180.0
    thrust_center_recovery_fraction = 0.20
    thrust_center_recovery_max_angle = 5.0 * pi / 180.0
    # Airborne with the WRONG posture (Phase 5). Flight resets otherwise start
    # within 5 deg of full flight tilt and landing within 30 deg, so the policy
    # has never been airborne mid-morph -- the state a stalled or lagging morph
    # actually produces. This is a SYMMETRIC tilt, corrected through tilt_mean,
    # which action_authority never gates, so it is safe from stage 1 and is
    # deliberately not scaled by any authority.
    airborne_posture_recovery_fraction = 0.15
    airborne_posture_recovery_range = (40.0 * pi / 180.0, pi / 2)
    trajectory_pos_rew_scale = [1.2, 1.2, 1.2]
    # Doubled from 0.8: the deployed policy behaves like a weak pure-P
    # controller (lateral overshoot on takeoff). Position paid 1.2/s doubled in
    # takeoff while velocity paid 0.8/s through a soft shaping, so derivative
    # action was never worth its noise cost. Pay for arriving damped.
    trajectory_vel_rew_scale = [1.6, 1.6, 1.6]
    drive_position_rew_scale = [2.4, 2.4, 2.4]
    drive_velocity_rew_scale = [1.6, 1.6, 1.6]
    drive_yaw_rew_scale = [1.2, 1.2, 1.2]
    drive_yaw_rate_rew_scale = [1.6, 1.6, 1.6]
    # Dominant DRIVE-only state cost.  Uses the worst controlled joint so one
    # hip untucking or one leg squatting cannot hide inside an average/config
    # score.  At a 22.5 deg error the stage-1 cost is 125 reward/s; at 90 deg
    # it is 500 reward/s, deliberately larger than any tracking benefit.
    # -500 -> -100. At -500 this was the largest term in the whole schema:
    # 4x the takeoff bonus and 2x a perfect landing. It fires the instant the
    # mode flips to DRIVE and demands the full 90 deg tuck, so at the 55 deg
    # landing minimum it charged |55-90|/90 * 500 = 194 reward/s from the
    # moment of touchdown.
    #
    # That is what was driving the early landing. Tucking 0->90 deg takes 4.0 s
    # at the pi/8 slew rate against a 4-5 s landing reference, so the only way
    # to be at drive config when the clock flips is to tuck through the whole
    # descent -- and vertical thrust scales with cos(tilt), so a vehicle tucked
    # past ~60 deg physically cannot hold altitude (hover collective 0.478
    # needs 0.956 at 60 deg, >1.0 at 65). The policy was not choosing to touch
    # down at 50% of the reference; it was tucking to dodge this penalty and
    # falling out of the sky as a consequence.
    drive_config_deviation_pen_scale = [-50.0, -50.0, -50.0]
    # Grace window after a landing hands over to DRIVE, in seconds. The drive
    # config penalty ramps in linearly across it instead of stepping to full
    # strength at the transition, so the tuck can be FINISHED ON THE GROUND
    # where lift no longer matters. This is the other half of the fix above:
    # cutting the scale lowers the pressure, this removes the reason to tuck
    # early at all.
    drive_config_settle_s = 4.0
    # Reference-tracking authority during TAKEOFF and LANDING, as a fraction of
    # the FLIGHT/DRIVE value. Applied to trajectory_pos_rew, trajectory_vel_rew
    # AND the costs the progress terms are differenced from, so the whole
    # tracking channel scales together.
    #
    # REVERTED TO 1.0 (inert). It was 0.25 for 2026-08-16_20-45-26, on the
    # reasoning that takeoff and landing are about safety and objective
    # completion rather than flying a pretty line. That run collapsed: 166
    # epochs, zero landings, zero takeoffs, the policy specialised into a
    # ground vehicle.
    #
    # The mechanism it is suspected of: cutting the dense guidance on TAKEOFF,
    # already the hardest phase to discover, while DRIVE kept full authority.
    # Attitude failures went 33% -> 61% and successful_takeoff_rew PEAKED at
    # epoch 40 then decayed to zero -- the policy tried, was punished, and
    # unlearned. Three changes moved at once in that run so this is not proven,
    # which is exactly why it goes back to 1.0 rather than to a compromise: the
    # next run tests the drive-config and early-touchdown changes against a
    # tracking channel that is known to work.
    #
    # The machinery stays so this is one value to change if it is worth
    # retrying, ideally on its own and on LANDING only.
    trajectory_transition_authority = 1.0
    # Touching down more than this far ahead of the landing reference is a
    # FAILURE, not a quality shortfall -- it terminates and is charged as an
    # invalid contact. Matches landing_early_tolerance_s, which is the same
    # boundary the mode gate already uses, so the reward, the transition and
    # the termination all agree on what "early" means.
    #
    # This is only survivable BECAUSE of the two changes above. Punishing early
    # touchdown while the drive config penalty still forced an early tuck would
    # have terminated nearly every episode on a manoeuvre the vehicle had no
    # way to avoid.
    # Terminal early touchdown is OFF. It was on for 2026-08-16_20-45-26 and
    # that run collapsed to a ground vehicle in 133 epochs -- making the end of
    # the landing route lethal removed the reason to fly at all.
    landing_early_termination = False
    # One-shot penalty for touching down more than landing_early_tolerance_s
    # ahead of the reference, charged once at the first early contact.
    #
    # Sized to make an early landing NET NEGATIVE rather than merely discounted,
    # which is what the three previous attempts failed to do. A poor landing
    # pays 25 + 225*q; at the q~0.2 the policy historically settled on that is
    # ~70, so -150 turns it into -80. A good, on-time landing is untouched at
    # up to 250. That is the difference from a terminal gate: bad timing costs
    # more than it earns, without removing the payoff for flying.
    # ZERO. The penalty shape is retired; an early touchdown is handled by
    # withholding only the unconditional landing baseline.
    #
    # -150 killed landings in 21-17-14: they ran at 5.01 through epoch 30, then
    # decayed to zero by epoch 110 and never came back. The fee exceeded what a
    # reachable landing paid (25 + 225*q at the q~0.3-0.4 that run reached is
    # 92-115), so declining to land dominated. Kept at 0 rather than deleted so
    # the term and its chart survive and it can be re-armed with one number.
    early_touchdown_pen = [0.0, 0.0, 0.0]
    flight_config_rew_scale = [0.5, 0.5, 1.0]
    flight_config_deviation_pen_scale = [-125.0, -125.0, -125.0]
    flight_position_deviation_pen_scale = [-2.5, -2.5, -2.5]
    flight_airborne_rew_scale = [20.0, 20.0, 20.0]
    flight_accel_alignment_rew_scale = [25.0, 25.0, 25.0]
    # Paired with the linear imbalance term: at 60/40 this costs ~60 per episode
    # of airborne time, against ~2.4 under the old squared form at -10.
    # Reduced 4x from [-50, -50, -100]: the imbalance is measured on delivered
    # thrust (kT includes the drawn losses), and holding attitude through
    # disturbances or asymmetric losses can legitimately require a sustained
    # nonzero imbalance -- at the old scale the penalty outbid the correction.
    airborne_diagonal_imbalance_pen_scale = [-12.5, -12.5, -25.0]
    # Stage 3 raised from 125: the tuck-frontier shaping terms zero out at
    # stage 3 (see tuck_absolute_linear_rew_scale / the takeoff guide scales),
    # so the outcome event carries the incentive the shaping used to split.
    # Stages 1-2 keep the original 125 alongside the live shaping.
    # Stage 2 raised 50% from 125: the tuck-frontier shaping is zeroed from
    # stage 2 on, so the outcome event carries more of the incentive.
    successful_takeoff_rew = [125.0, 187.5, 250.0]
    # Unconditional credit for the takeoff event itself, paid on top of the
    # timing-weighted scale above. Mirrors contact_in_acceptance_baseline_rew on
    # the landing side, which the takeoff event had no equivalent of: the bonus
    # was multiplied straight by takeoff_timing_multiplier, which reaches zero
    # about 2.2 s after the climb clock starts, so a first slow clumsy liftoff --
    # exactly what discovery produces -- paid nothing at all. The floor makes the
    # event worth finding; the scale above still pays for finding it quickly.
    successful_takeoff_baseline_rew = [100.0, 100.0, 100.0]
    # Liftoff timing is scored against the fastest liftoff the morph actuator
    # can physically deliver from the tilt held when the takeoff phase begins.
    # Untucking during the drive lowers the measured time and the baseline by
    # the same amount, so sandbagging the drive earns nothing here.
    # Calibration constants. Vertical thrust scales with cos(tilt), so hover is
    # marginal near 60 deg (throttle = hover / cos(tilt) saturates there); real
    # liftoff needs margin above that plus rotor spin-up, hence 30 deg. With the
    # pi/8 rad/s morph slew this puts the baseline at (90 - 30) / 22.5 = 2.67 s
    # from a fully tucked flip, and zero credit at 4.17 s. Retune both from a
    # run: set the baseline to the fastest liftoff actually observed, and the
    # slack so typical episodes land mid-range rather than pinned at 0 or 1.
    takeoff_liftoff_tilt_rad = 30.0 * pi / 180.0
    takeoff_timing_slack_s = 1.5
    takeoff_lift_rew_scale = [20.0, 20.0, 20.0]
    # Dense takeoff/landing discovery shaping is removed after the first 150
    # training epochs; the outcome rewards then carry the objective.
    discovery_reward_end_epoch = 150.0
    # Discovery shaping, stage 1 only: the takeoff guide. Two measured terms
    # replace the old untuck-action pay ("untuck early"): untuck progress (how
    # far the worst hip has come from tucked toward flight posture) and
    # vertical thrust (the world-up component of delivered thrust as a
    # fraction of nominal max, paid until liftoff). Thrust points along the
    # tilted rotor axes, so the vertical component is ~zero while tucked and
    # only becomes earnable as the untuck proceeds -- the pair teaches the
    # sequence untuck-then-push without prescribing the morph action itself.
    # Zero from stage 2 on, where the takeoff outcome carries the incentive
    # and the policy gets freedom in HOW it morphs.
    untuck_progress_rew_scale = [100.0, 0.0, 0.0]
    vertical_thrust_rew_scale = [2.5, 0.0, 0.0]
    invalid_contact_pen = [-0.4, -0.4, -0.4]
    invalid_contact_rate_pen = [-100.0, -100.0, -100.0]
    post_landing_invalid_contact_pen = [-100.0, -100.0, -100.0]
    # Stage 1 ramps this landing-posture requirement from contacts-only to the
    # full threshold. Later stages keep the full threshold.
    # Vertical thrust scales with cos(tilt), so holding altitude needs
    # throttle = hover / cos(tilt). At the measured thrust-to-weight of ~2.3
    # that saturates near 64 deg, and at the nominal 2.0 near 60 deg: 65 deg
    # was never a posture the vehicle could reach and hold, so the gate could
    # not be satisfied and every landing scored the reward floor. 55 deg is
    # sustainable at ~76% throttle, leaves ~24% for attitude, and is the
    # minimum the landing gear can absorb.
    landing_min_tilt_rad = 55.0 * pi / 180.0
    # Per-phase thrust authority, deliberately ASYMMETRIC.
    #
    # TAKEOFF gets no thrust until the arms have untucked to 50 deg (40 deg from
    # flight config). Vertical thrust is cos(tilt) * 4 * kT, so at 60 deg hover
    # already needs 0.956 of full collective and above ~61 deg it is impossible
    # at any throttle. Below that line the rotors cannot lift but their axes
    # point far enough sideways to yaw and shove -- which is how the 2026-08-16
    # policy learned to steer on the ground. Takeoff must commit to the flight
    # configuration before it earns thrust.
    takeoff_thrust_full_tilt_rad = 45.0 * pi / 180.0
    takeoff_thrust_zero_tilt_rad = 65.0 * pi / 180.0
    # A landing may not complete before its reference arrives. Without this the
    # only conditions were contact and posture, so a policy could touch down
    # arbitrarily early, flip to DRIVE, and leave the reference behind -- which
    # then read as the reference "teleporting" to the drive segment.
    landing_early_tolerance_s = 0.5
    # 300 -> 100 start, so the full 55 deg threshold arrives at epoch 300
    # instead of 500.
    #
    # This is a STAGE-1-ONLY ramp (_landing_min_tilt_threshold returns the full
    # target outright at stages 2 and 3), and ATMO trains at stage 1, so it is
    # the schedule that actually governs this run. At 300/200 the landing
    # posture requirement was ZERO for the first 300 epochs and only ~14% of
    # target by epoch 350, where behavior flattened in 2026-08-16_13-43-29.
    # In other words the requirement was never enforced at all in practice, and
    # every landing in the useful part of the run completed on contacts alone.
    # That is the wrong footing for judging the landing-earliness reward, which
    # assumes a landing means an arrival in the landing posture.
    #
    # 100 keeps the free window over takeoff discovery (measured at epoch
    # 100-108 across two runs) and then ramps across the region where the
    # policy is still moving.
    landing_min_tilt_ramp_start_epoch = 100.0
    landing_min_tilt_ramp_epochs = 200.0
    # Contact-tilt credit window. The old window ran 60 -> 75 deg, so the whole
    # physically reachable band scored below 0.09 of the multiplier and 7 deg
    # was indistinguishable from 31 deg.
    # Two tiers, because the two angles mean different things. 55 deg needs no
    # special manoeuvre -- full throttle holds altitude to ~60 deg -- but it is
    # the edge of what the gear can safely absorb, so it is adequate rather
    # than good and earns contact_tilt_safe_weight, not full credit. 65 deg is
    # past the altitude-hold limit and only reachable as a transit (a brief
    # climb, then a committed tuck), but it is markedly kinder to the hardware,
    # so the remaining credit pulls toward it.
    # The floor is where the geometry first permits a wheel contact at all;
    # below it the angle is never sampled at touchdown.
    contact_tilt_floor_rad = 30.0 * pi / 180.0
    contact_tilt_safe_rad = 55.0 * pi / 180.0
    contact_tilt_target_rad = 65.0 * pi / 180.0
    contact_tilt_safe_weight = 0.6
    # Tilt still gates contact quality multiplicatively, but a 0.025 floor also
    # erased the timing/speed/position gradients whenever tilt fell short.
    contact_tilt_multiplier_floor = 0.20
    # Per-hip bias is a designed attitude actuator (see morph_bias_start_stage),
    # so small spreads are free and only gross splits are penalised.
    hip_spread_deadband_rad = 9.0 * pi / 180.0
    # Per-step and squared, so the episode cost grows fast with the spread. At
    # -40 a 0.8 rad split cost ~198 over a 12 s episode and a 1.2 rad split
    # ~522, against a landing worth 100-300 and successful_takeoff_rew of 125:
    # large enough to teach "never move the hips differentially" in every mode
    # before landing had been discovered at all. At -8 the same splits cost ~40
    # and ~104, a correction rather than a prohibition.
    # Doubled to -16: ~79 and ~209 for those same splits. A moderate split is
    # still only a correction, but a gross one now costs about as much as the
    # landing it would have spoiled. This is close to the ceiling -- the static
    # guard on episode cost caps the scale near -20 -- so if hip disagreement is
    # still not being suppressed, narrow hip_spread_deadband_rad rather than
    # pushing the scale further, or the term starts prohibiting the differential
    # bias that morph_bias_start_stage deliberately unlocks.
    hip_spread_pen_scale = [-16.0, -16.0, -16.0]
    # Reference tracking fades over the final approach so the vehicle can trade
    # trajectory fidelity for a late, committed tuck.
    landing_tracking_fade_s = 1.2
    # Terminal vertical-speed limit. Was a hardcoded 2.0, which fired on ~48% of
    # episodes and was by a wide margin the dominant terminator -- free fall
    # reaches 2 m/s in 0.2 s, so the limit caught ordinary flight transients and,
    # worse, every exploratory throttle burst on the exact axis takeoff has to be
    # discovered along. Raised to 4.0: still well inside the envelope where the
    # airframe and gear survive an arrival, but no longer a wall across the
    # vertical exploration the takeoff discovery depends on.
    vertical_speed_termination_mps = 4.0
    too_fast_pen_scale = [-100.0, -100.0, -100.0]
    # Squared raw policy action on the authority-gated channels (tilt_balance,
    # thrust_center_xy), weighted by (1 - authority). While a channel's
    # authority is below 1 the plant ignores some or all of the command, so
    # parking at the rails is free habit-forming garbage -- the probe's
    # tilt_balance pinned at +/-1 is the signature. At authority 1 the weight
    # is zero: using a live actuator hard is never taxed.
    gated_action_saturation_pen_scale = [-0.5, -0.5, -0.5]
    timeout_pen = [-15.0, -15.0, -15.0]
    # Discovery shaping for the landing tuck, stage-zeroed like the takeoff guide:
    # live at stage 1, zero from stage 2 on where touchdown quality carries it.
    tuck_absolute_linear_rew_scale = [200.0, 0.0, 0.0]
    tuck_absolute_rew_cap = [360.0, 360.0, 360.0]
    # baseline + scale * tilt * mean(timing, speed, position). The 1200 scale
    # was set when the tilt multiplier was pinned near its 0.025 floor, so the
    # term paid ~130 in practice. With the window retargeted onto the reachable
    # band the multiplier now reaches ~1.0, a 10-30x jump that would swamp every
    # per-step term, so the scale comes down to keep the realised magnitude
    # where it was.
    # Stage 3 is raised well above the earlier stages. Trajectory tracking is a
    # per-step term, so it accrues over the whole episode while the landing pays
    # once; at 200 the long-run optimum drifts toward flying the reference
    # nicely and treating the touchdown as incidental. 600 restores the original
    # design point under the retargeted tilt curve -- roughly 0.5 on every
    # factor gives 100 + 600*0.5 = 400, which is what the 1200 scale was chosen
    # to produce back when the tilt multiplier could not exceed ~0.09.
    # Reduced from [200, 200, 600]: the policy was optimizing too heavily for
    # the contact reward at the expense of the rest of the objective.
    # Stage 2 raised 50% alongside successful_takeoff_rew: the tuck discovery
    # shaping is zeroed from stage 2 on, so the landing outcome carries it.
    # Stage 1 now uses the same 50 baseline / 450 quality scale as stage 3,
    # making the full landing worth 500 from the start of training.
    #
    # This is the defect run 17-49-41 measured. The baseline is paid for ANY
    # completed landing, so at stage 1 the old 100/150 split meant merely
    # getting four wheels down was worth 100 of a 250 maximum -- 40%,
    # guaranteed, for a landing of arbitrarily bad quality. Timing, tilt, speed
    # and position then competed for the remaining 60%, averaged three ways and
    # multiplied by a tilt multiplier that sat at 0.43 all run. The policy
    # settled at contact_in_acceptance_rew ~128, i.e. the baseline plus a
    # sliver: it was collecting the participation prize and ignoring quality.
    # landing_early_fraction plateaued at 0.60 against a 0.47 baseline, and
    # contact_tilt_multiplier never moved off 0.43.
    #
    # Both reward changes earlier in the session (timing floor 0.2 -> 0.05,
    # early sigma 0.20 -> 1.5 m) adjusted terms INSIDE the scale. Neither
    # touched the baseline, which is why the sigma fix produced a real
    # improvement to 0.41 by epoch 88 and then lost it the moment the tilt ramp
    # gave the policy something else to spend effort on: "land badly, collect
    # the baseline" stayed rational throughout.
    #
    # Stage 1 now carries the full landing objective early, while the doubled
    # baseline remains a minority of the total landing reward.
    contact_in_acceptance_rew_scale = [450.0, 337.5, 450.0]
    contact_in_acceptance_baseline_rew = [100.0, 75.0, 100.0]
    contact_in_acceptance_rew_cap = [11182.5, 11182.5, 11182.5]
    # Per-step cost of being on the ground before the landing reference gets
    # there, scaled by the remaining fraction of the descent. Sized against
    # what the behavior is worth: touching down at 60% early (the measured
    # plateau of run 17-49-41) on a ~4 s reference means ~2.4 s of sitting
    # early, and the deficit fraction averages ~0.3 over that, so at -60 the
    # habit costs roughly 43 -- comparable to the 25 baseline it would
    # otherwise be protecting, and well under the 225 a good landing pays. Big
    # enough to matter, too small to make the policy refuse to land.
    early_ground_pen_scale = [-60.0, -60.0, -60.0]
    # Half-width of the flat full-credit band around contact_speed_target_mps.
    # Narrowed from 0.40 together with the target: Gazebo touchdowns were arriving
    # harder than intended, and the old 0.80 m/s-wide band credited everything
    # from +0.15 down to -0.65 identically, so nothing inside it pulled toward a
    # softer arrival at all.
    contact_speed_rew_free_speed = [0.2, 0.2, 0.2]
    # Zero point pulled in 1.0 -> 0.6 (2026-08-19). The free BAND is deliberately
    # untouched: narrowing it to 0.35 previously drove the violent late flare
    # noted above, because it puts the speed term in direct conflict with tilt.
    # The slope outside the band was the real problem -- at zero=1.0 a 0.7 m/s
    # arrival still collected 63% credit, so nothing meaningful punished a hard
    # touchdown. At 0.6 the same arrival collects 27% and 0.8 m/s collects the
    # 0.025 floor, while everything inside 0.00..-0.40 m/s is credited exactly
    # as before. Measured motivation: hardware first-leg contact ~0.7 m/s.
    contact_speed_rew_zero_speed = [0.6, 0.6, 0.6]
    # Descent rate credited as ideal at touchdown, rather than zero. At the
    # landing tilt the vehicle cannot hold altitude at all -- throttle needed is
    # hover / cos(tilt), which saturates near 60 deg -- so scoring against a
    # zero-speed touchdown asks for a hover the posture forbids and puts the
    # speed term in direct conflict with the tilt term. Credit is measured as
    # the deviation from this descent rate in either direction.
    # Target and free band are equal, which places the flat band at exactly
    # 0.00 to -0.40 m/s: contact at 0.40 or slower is fully credited. Widened
    # from 0.35 (all stages -- the band was always meant to be global): 0.35
    # proved tight enough to make the speed factor fight the tilt objective
    # into a violent late flare. Still far below the old 0.80-wide band that
    # credited everything identically. If tilt at touchdown regresses further,
    # widen the free band before touching the target.
    contact_speed_target_mps = 0.2
    # Timing floor. Was 0.2 -- raised above the 0.025 the other factors use on
    # the grounds that touchdown timing is the factor least under the policy's
    # control once the tuck dictates the descent. Lowered to 0.05 because the
    # measured failure is the opposite one: the vehicle arrives far early, and
    # The baseline is now withheld for early touchdowns; the quality scale stays
    # active so the timing multiplier supplies the learning signal.
    contact_timing_baseline = 0.05
    # Exponential scale of the EARLY half of touchdown_timing, in metres of
    # remaining reference distance (the late half is in seconds and is
    # unchanged). The prior Gaussian was touchdown-precision scale and flattened
    # in its early tail; this exponential retains a nonzero gradient throughout
    # the landing segment.
    contact_timing_early_sigma_m = 0.5
    contact_reward_min_time_remaining_s = [0.50, 0.50, 0.50]
    # Squared discrete second difference of the policy action, reduced 2.5x
    # after the hardware-oriented sweep. A smooth ramp remains nearly free while
    # alternating command jitter is expensive. No dt^-2 conversion: this is a
    # fixed-50-Hz command-smoothness cost, not a physical jerk measurement.
    action_jerk_pen_scale = [-0.6, -0.6, -0.6]
    yaw_rate_pen_scale = [-0.6, -0.6, -0.6]
    yaw_angle_pen_scale = [-0.3, -0.6, -0.6]
    thrust_center_action_pen_scale = [0.0, 0.0, -0.025]
    thrust_center_loss_offset_pen_scale = [0.0, 0.0, -0.025]
    attitude_termination_angle = 40.0 * pi / 180.0
    attitude_termination_dwell_s = 0.15
    attitude_failure_penalty = -100.0
    # Continuous airborne attitude cost: -50/s at 30 deg, increasing with the
    # square of tilt so the policy gets a gradient well before termination.
    attitude_tilt_penalty_scale = -50.0 / (30.0 * pi / 180.0) ** 2
    # Past a few metres every trajectory term is flat or clamped to zero, so
    # there is no gradient pulling the vehicle back and the policy learns to
    # abandon the reference. Terminate instead of trying to price a region the
    # reward cannot express. Dwell mirrors the attitude failure treatment so a
    # brief transient overshoot does not end the episode.
    trajectory_deviation_dwell_s = 0.25
    # Deviation wall (see _trajectory_deviation_threshold): 8 m at epoch zero
    # tightening linearly to 2.5 m by epoch 400, independent of stage or
    # overlay state. Wide start + the dwell above absorb worst-case
    # actuator/initial-state draws; the observed-tau channel makes sustained
    # excursions a policy failure rather than an unavoidable accident.
    trajectory_deviation_wall_start = 8.0
    trajectory_deviation_wall_final = 2.5
    # 400 -> 300, for the same reason as the landing-tilt ramp above: a wall
    # that reaches its final 2.5 m at epoch 400 is still at ~3.9 m where
    # behavior flattens, so the policy is never actually held to the limit it
    # is supposed to be trained against. This ramp is absolute-epoch and
    # stage-independent, so it applies to the stage-1 run as written.
    trajectory_deviation_wall_ramp_epochs = 300.0
    trajectory_deviation_penalty = -50.0
    post_landing_takeoff_height = 0.05
    post_landing_takeoff_dwell_s = 0.25
    # Reduced by a third from 5.025 / 1.005: the landing posture the task asks
    # for sits close to the vehicle's thrust authority, so disturbance headroom
    # comes directly out of the margin available for the tuck. Briefly raised 50%
    # back to 5.025 and reverted -- at that level the impulse peak reached about
    # one vehicle weight once training_overlay_disturbance_multiplier (2.0 past
    # epoch 150) compounded onto it, on an airframe with a thrust-to-weight of
    # only 2.0.
    # These still compound with that overlay and with the per-episode bucket in
    # disturbance_batch_scale_multipliers, so the hard bucket past epoch 150 sees
    # 6.70, not 3.35. If a further reduction is wanted, lower that overlay
    # multiplier before these: it is what sets the peak.
    disturbance_force_scale = [3.35, 3.35, 3.35]
    disturbance_moment_scale = [3.35, 3.35, 3.35]
    disturbance_cts_force_scale = [0.67, 0.67, 0.67]
    disturbance_cts_moment_scale = [0.67, 0.67, 0.67]


class CombinedTask(BaseTask):
    """Independent drive/takeoff/flight/landing task with balanced phase resets."""

    DRIVE = 0
    TAKEOFF = 1
    FLIGHT = 2
    LANDING = 3
    TAKEOFF_ROUTE = 0
    LANDING_ROUTE = 1

    reward_keys = (
        "trajectory_pos_rew",
        "trajectory_vel_rew",
        "drive_translation_rew",
        "drive_yaw_rew",
        "position_progress_rew",
        "velocity_progress_rew",
        "yaw_progress_rew",
        "drive_config_deviation_penalty",
        "flight_config_rew",
        "flight_config_deviation_penalty",
        "flight_position_deviation_penalty",
        "flight_airborne_rew",
        "flight_accel_alignment_rew",
        "airborne_diagonal_imbalance_penalty",
        "successful_takeoff_rew",
        "takeoff_lift_rew",
        "untuck_progress_rew",
        "vertical_thrust_rew",
        "tuck_progress_rew",
        "early_ground_penalty",
        "early_touchdown_penalty",
        "invalid_contact_penalty",
        "action_jerk_penalty",
        "yaw_angle_penalty",
        "yaw_rate_penalty",
        "thrust_center_penalty",
        "hip_spread_penalty",
        "contact_in_acceptance_rew",
        "too_fast_penalty",
        "gated_action_saturation_penalty",
        "timeout_high_penalty",
        "attitude_tilt_penalty",
        "attitude_failure_penalty",
        "trajectory_deviation_penalty",
    )

    def __init__(self, env, cfg: CombinedTaskCfg):
        super().__init__(
            env,
            cfg,
            self.reward_keys,
            (
                # Log-only split of drive tuck quality by what follows the
                # drive. Episode logs are per-episode sums, and the branches
                # spend different amounts of time in drive, so each tuck sum is
                # paired with its own dwell time: tuck / time is the time
                # averaged score in [0, 1] and is directly comparable. Equal
                # ratios mean the policy cannot tell the branches apart; a high
                # pure / low takeoff split means it still can.
                "drive_tuck_pure_drive",
                "drive_tuck_takeoff_route",
                "drive_time_pure_drive",
                "drive_time_takeoff_route",
                # Same split by direction of travel. err / time is the mean
                # position error in metres while driving, so a healthy forward
                # figure beside a flat reverse one confirms that reverse is the
                # unlearned half rather than driving being broken generally.
                "drive_err_forward",
                "drive_err_reverse",
                "drive_vel_err_forward",
                "drive_vel_err_reverse",
                "drive_closing_speed_forward",
                "drive_closing_speed_reverse",
                "drive_displaced_time_forward",
                "drive_displaced_time_reverse",
                "drive_time_forward",
                "drive_time_reverse",
                # Standing wheel command while the channel is GATED OFF, and the
                # dwell it accrued over. gated / time is the mean |wheel action|
                # in the regime where the plant ignores it entirely, so a healthy
                # policy drives it to zero and only a habit keeps it high.
                #
                # This exists because the defect it measures was invisible until
                # a checkpoint was exported and probed offline: the ATMO stage-1
                # actor held drive near -0.7 with a stationary reference and
                # never once commanded forward, which put it off the pad before
                # the climb and cost every takeoff. Watching it here turns a
                # post-hoc export into a curve.
                "wheel_cmd_gated",
                "wheel_time_gated",
                "flight_pos_err",
                "flight_vel_err",
                "flight_closing_speed",
                "flight_displaced_time",
                "flight_time",
                "drive_turn_yaw_rate_err",
                "drive_turn_correct_direction",
                "drive_turn_time",
                "drive_err_stationary",
                "drive_err_stationary_positive",
                "drive_err_stationary_negative",
                "drive_err_straight",
                "drive_err_curved",
                "drive_time_stationary",
                "drive_time_stationary_positive",
                "drive_time_stationary_negative",
                "drive_time_straight",
                "drive_time_curved",
                "takeoff_timing_multiplier",
                "contact_tilt_multiplier",
                "final_contact_speed_multiplier",
                "final_contact_position_quality",
                "contact_timing_quality",
                "landing_early_fraction",
            ),
        )
        n = env.num_envs
        device = env.device
        self._curve_arc_samples = torch.linspace(0.0, 1.0, 33, device=device)
        env._desired_pos_w = torch.zeros(n, 3, device=device)
        env._virtual_xy_offset_w = torch.zeros(n, 3, device=device)
        env._combined_mode = torch.zeros(n, dtype=torch.long, device=device)
        env._combined_previous_position_cost = torch.zeros(n, device=device)
        env._combined_previous_velocity_cost = torch.zeros(n, device=device)
        env._combined_previous_yaw_quality = torch.zeros(n, device=device)
        env._combined_previous_root_velocity = torch.zeros(n, 3, device=device)
        env._combined_progress_regime = torch.full((n,), -1, dtype=torch.long, device=device)
        env._combined_progress_valid = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_route = torch.zeros(n, dtype=torch.long, device=device)
        env._combined_expected_landing = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_landing_to_takeoff = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_transition_takeoff_reached = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_vertical_trajectory = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_curve_active = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_drive_curve_active = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_pure_drive = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_pure_drive_stationary = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_stationary_initial_error_sign = torch.zeros(n, device=device)
        env._combined_pure_drive_straight = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_pure_drive_curved = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_curve_origin_w = torch.zeros(n, 3, device=device)
        env._combined_curve_heading = torch.zeros(n, device=device)
        env._combined_curve_speed = torch.zeros(n, device=device)
        env._combined_curve_lateral_amplitude = torch.zeros(n, device=device)
        env._combined_curve_lateral_wavenumber = torch.zeros(n, device=device)
        env._combined_curve_lateral_phase = torch.zeros(n, device=device)
        env._combined_curve_vertical_amplitude = torch.zeros(n, device=device)
        env._combined_curve_vertical_wavenumber = torch.zeros(n, device=device)
        env._combined_curve_vertical_phase = torch.zeros(n, device=device)
        env._combined_phase_start_elapsed = torch.zeros(n, 1, device=device)
        env._combined_phase_initial_time = torch.zeros(n, 1, device=device)
        env._combined_landing_transition_time = torch.zeros(n, device=device)
        env._combined_first_contact_time = torch.zeros(n, device=device)
        env._combined_first_contact_recorded = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_final_contact_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_final_contact_reference_distance = torch.zeros(n, device=device)
        env._combined_first_contact_xy_error = torch.zeros(n, device=device)
        env._combined_first_contact_tilt = torch.zeros(n, device=device)
        env._combined_spawn_pos_w = torch.zeros(n, 3, device=device)
        env._combined_drive_start_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_drive_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_drive_duration = torch.ones(n, 1, device=device)
        env._combined_takeoff_duration = torch.ones(n, 1, device=device)
        env._combined_flight_duration = torch.ones(n, 1, device=device)
        env._combined_landing_duration = torch.ones(n, 1, device=device)
        env._combined_liftoff_pos_w = torch.zeros(n, 3, device=device)
        env._combined_takeoff_end_pos_w = torch.zeros(n, 3, device=device)
        env._combined_takeoff_start_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_takeoff_end_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_landing_start_pos_w = torch.zeros(n, 3, device=device)
        env._combined_landing_pos_w = torch.zeros(n, 3, device=device)
        env._combined_landing_start_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_ground_velocity_w = torch.zeros(n, 3, device=device)
        env._combined_tuck_frontier = torch.zeros(n, device=device)
        env._combined_tuck_frontier_mode = torch.full((n,), -1, dtype=torch.long, device=device)
        env._combined_yaw = torch.zeros(n, device=device)
        env._combined_drive_yaw = torch.zeros(n, device=device)
        env._combined_flight_yaw = torch.zeros(n, device=device)
        env._combined_takeoff_start_morph = torch.full((n,), pi / 2, device=device)
        env._combined_airborne = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_takeoff_airborne_time = torch.full((n,), -1.0, device=device)
        env._combined_takeoff_timing_multiplier = torch.ones(n, device=device)
        env._combined_reset_throttle = torch.zeros(n, 1, device=device)
        env._combined_ep_contact_in_acceptance = torch.zeros(n, dtype=torch.bool, device=device)
        # Latched copy of takeoff_event, the env's own takeoff-success
        # condition (route + climb started + physically airborne + not
        # died). It fires once and is otherwise unrecorded. Write-only.
        env._combined_ep_took_off = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_post_landing_takeoff = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_post_landing_takeoff_dwell = torch.zeros(n, device=device)
        env._combined_touchdown_config_score = torch.zeros(n, device=device)
        env._combined_touchdown_config_recorded = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_nonfinite_state = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_trajectory_pos_error = torch.zeros(n, device=device)
        env._combined_attitude_failure_dwell = torch.zeros(n, device=device)
        env._combined_attitude_failure = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_early_touchdown = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_trajectory_deviation_dwell = torch.zeros(n, device=device)
        env._combined_trajectory_deviation_failure = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_morph_recovery_reset = torch.zeros(n, dtype=torch.bool, device=device)
        env._combined_thrust_center_recovery_reset = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_transition_update_step = -1
        self._termination_cache_step = -1
        self._termination_cache = None
        self._observation_context_cache = None
        self._morph_max_velocity_cache: float | None = None

    def _morph_max_velocity(self) -> float:
        """Slowest morph joint slew rate, cached to avoid a per-step device sync."""
        if self._morph_max_velocity_cache is None:
            group_name = self.env.vehicle.spec.morph_joint_group
            if group_name is None:
                self._morph_max_velocity_cache = 0.0
            else:
                runtime = self.env.vehicle.joint_groups[group_name]
                # Spec nominal, NOT torch.min(runtime.max_velocity): the runtime
                # tensor is per-env randomized at reset, so min-over-envs would
                # cache whatever the first draw happened to produce and the
                # timing baseline would differ run to run. With the prep window
                # covering the worst-case draw the baseline clamps to zero for
                # every start tilt anyway, so the nominal is exact.
                self._morph_max_velocity_cache = float(runtime.spec.max_velocity)
        return self._morph_max_velocity_cache

    def _takeoff_timing_multiplier(self, airborne_time: torch.Tensor) -> torch.Tensor:
        """Score liftoff speed relative to the actuator's feasible optimum.

        The baseline is the time the morph actuator needs to slew from the tilt
        held when the takeoff phase began down to the tilt where liftoff becomes
        possible. Scoring the excess over that baseline keeps the full [0, 1]
        range reachable without pre-untucking, and makes the score independent
        of the randomized trajectory duration.
        """
        env = self.env
        max_velocity = self._morph_max_velocity()
        if max_velocity <= 0.0:
            return torch.ones_like(airborne_time)
        slack = max(float(self.cfg.takeoff_timing_slack_s), 1e-3)
        # The prep window is announced slew time, so it comes off the baseline.
        # Once the window covers the full slew the baseline is zero for every
        # tilt, which is what makes untucking during it free rather than a way
        # to buy slack: a policy that ignores the window has no larger budget.
        min_time = torch.clamp(
            (env._combined_takeoff_start_morph - float(self.cfg.takeoff_liftoff_tilt_rad))
            / max_velocity
            - float(self.cfg.takeoff_prep_duration_s),
            min=0.0,
        )
        return torch.clamp(1.0 - (airborne_time - min_time) / slack, 0.0, 1.0)

    def episode_reward_done_mask(self, key: str, done: torch.Tensor) -> torch.Tensor:
        env = self.env
        if key == "contact_in_acceptance_rew":
            return done & env._combined_expected_landing
        if key == "takeoff_timing_multiplier":
            # Restrict to episodes that actually lifted off, so this reads as
            # timing quality rather than quality times takeoff rate. The rate is
            # then recoverable: successful_takeoff_rew stays unmasked and is now
            # baseline + scale * timing, so with B and S the stage values and Q
            # this masked ratio, the rate is successful_takeoff_rew / (B + S * Q).
            # _combined_takeoff_airborne_time resets to -1 and is only written
            # on a real takeoff event.
            return done & (env._combined_takeoff_airborne_time >= 0.0)
        if key in (
            "contact_tilt_multiplier",
            "final_contact_speed_multiplier",
            "final_contact_position_quality",
            "contact_timing_quality",
            "landing_early_fraction",
        ):
            # Same reasoning as takeoff_timing_multiplier. These are all logged
            # as landing_completed * value, so unmasked they read as quality
            # times landing rate and a rate change is indistinguishable from a
            # quality change -- a drop can mean fewer landings at the same tilt
            # just as easily as worse tilt. contact_in_acceptance_rew stays
            # masked to expected landings, so the rate is recoverable from it.
            return done & env._combined_ep_contact_in_acceptance
        if key in (
            "drive_err_forward",
            "drive_vel_err_forward",
            "drive_closing_speed_forward",
            "drive_displaced_time_forward",
            "drive_time_forward",
        ):
            return done & self._drive_reset() & (env._combined_curve_speed >= 0.0)
        if key in (
            "drive_err_reverse",
            "drive_vel_err_reverse",
            "drive_closing_speed_reverse",
            "drive_displaced_time_reverse",
            "drive_time_reverse",
        ):
            return done & self._drive_reset() & (env._combined_curve_speed < 0.0)
        if key in ("drive_tuck_pure_drive", "drive_time_pure_drive"):
            return done & env._combined_pure_drive
        if key in ("drive_tuck_takeoff_route", "drive_time_takeoff_route"):
            return done & self._reset_as_drive_then_takeoff()
        if key in ("drive_err_stationary", "drive_time_stationary"):
            return done & env._combined_pure_drive_stationary
        if key in ("drive_err_stationary_positive", "drive_time_stationary_positive"):
            return done & env._combined_pure_drive_stationary & (
                env._combined_stationary_initial_error_sign > 0.0
            )
        if key in ("drive_err_stationary_negative", "drive_time_stationary_negative"):
            return done & env._combined_pure_drive_stationary & (
                env._combined_stationary_initial_error_sign < 0.0
            )
        if key in ("drive_err_straight", "drive_time_straight"):
            return done & env._combined_pure_drive_straight
        if key in ("drive_err_curved", "drive_time_curved"):
            return done & env._combined_pure_drive_curved
        return done

    def _drive_reset(self) -> torch.Tensor:
        """Episodes that reset into drive mode, either pure or before a takeoff."""
        env = self.env
        return env._combined_drive_curve_active & (~env._combined_landing_to_takeoff)

    def _reset_as_drive_then_takeoff(self) -> torch.Tensor:
        """Episodes that reset into drive with a takeoff still to come.

        ``drive_curve_active`` is set at reset for drive-mode resets and for the
        landing-to-takeoff sequence, so excluding the latter leaves exactly the
        drive-mode resets, and excluding pure drive leaves those that take off.
        """
        env = self.env
        return (
            env._combined_drive_curve_active
            & (~env._combined_landing_to_takeoff)
            & (~env._combined_pure_drive)
        )

    def _disturbance_training_multiplier(self) -> float:
        return (
            float(self.cfg.training_overlay_disturbance_multiplier)
            if self.env._training_epoch() >= float(self.cfg.training_overlay_start_epoch)
            else 1.0
        )

    def disturbance_force_scale(self) -> float:
        return self.env.vehicle.nominal_total_kT() * self.env.cfg.disturbance_force_scale * self.stage_value(
            self.cfg.disturbance_force_scale, "disturbance_force_scale"
        ) * self._disturbance_training_multiplier()

    def disturbance_moment_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_rotor_moment_coeff()
            * self.env.cfg.disturbance_moment_scale
            * self.stage_value(self.cfg.disturbance_moment_scale, "disturbance_moment_scale")
            * self._disturbance_training_multiplier()
        )

    def disturbance_cts_force_scale(self) -> float:
        return self.env.vehicle.nominal_total_kT() * self.env.cfg.dist_force_cts_scale * self.stage_value(
            self.cfg.disturbance_cts_force_scale, "disturbance_cts_force_scale"
        ) * self._disturbance_training_multiplier()

    def disturbance_cts_moment_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_rotor_moment_coeff()
            * self.env.cfg.dist_moment_cts_scale
            * self.stage_value(self.cfg.disturbance_cts_moment_scale, "disturbance_cts_moment_scale")
            * self._disturbance_training_multiplier()
        )

    def _contact_state(self) -> dict[str, torch.Tensor]:
        env = self.env
        sensor = env.scene["contact_sensor"].data.current_contact_time
        valid_time = sensor[:, env._valid_contact_ids]
        invalid_time = sensor[:, env._invalid_contact_ids]
        valid_mask = valid_time > 0.0
        return {
            "valid_contact_time": valid_time,
            "valid_contact_mask": valid_mask,
            "any_valid_contacts": torch.any(valid_mask, dim=1),
            "valid_contact_count": torch.sum(valid_mask, dim=1),
            "invalid_contacts": torch.any(invalid_time > 0.0, dim=1),
        }

    def _joint_config_score(self, joint_config) -> torch.Tensor:
        env = self.env
        errors = []
        for group_name, _ in joint_config:
            if group_name not in env.vehicle.joint_groups:
                continue
            positions = env.vehicle.joint_group_positions(group_name)
            target = env.vehicle.joint_config_target_values(
                joint_config, group_name, positions
            )
            if target is not None:
                error = torch.abs(positions - target)
                if group_name == env.vehicle.spec.leg_joint_group:
                    error *= 2.0
                errors.append(error)
        if not errors:
            return torch.zeros(env.num_envs, device=env.device)
        worst_error = torch.max(torch.cat(errors, dim=1), dim=1).values
        return torch.clamp(1.0 - worst_error / (pi / 2.0), min=0.0, max=1.0)

    def reset_initial_state(self, env_ids: torch.Tensor, randomized: bool):
        self._observation_context_cache = None
        env = self.env
        count = len(env_ids)
        training_overlay = env._training_epoch() >= float(self.cfg.training_overlay_start_epoch)
        # Widens every reset error envelope below under the stage-3 hardening
        # overlay, and is exactly 1.0 whenever that overlay is inactive.
        error_scale = env.initial_error_multiplier()
        sequence_test = bool(getattr(env.cfg, "combined_sequence_test", False))
        if sequence_test:
            modes = torch.full((count,), self.FLIGHT, dtype=torch.long, device=env.device)
        elif randomized:
            if training_overlay:
                modes = torch.where(
                    torch.rand(count, device=env.device)
                    < float(self.cfg.training_overlay_drive_fraction),
                    torch.full((count,), self.DRIVE, dtype=torch.long, device=env.device),
                    torch.full((count,), self.FLIGHT, dtype=torch.long, device=env.device),
                )
            else:
                probabilities = torch.tensor(self.cfg.mode_probabilities, device=env.device)
                modes = torch.multinomial(probabilities / probabilities.sum(), count, replacement=True)
        else:
            modes = torch.full((count,), self.DRIVE, dtype=torch.long, device=env.device)
        landing_to_takeoff = torch.full(
            (count,), sequence_test, dtype=torch.bool, device=env.device
        )
        if randomized and not sequence_test:
            landing_to_takeoff = torch.rand(count, device=env.device) < float(
                self.cfg.landing_to_takeoff_fraction
            )
            modes[landing_to_takeoff] = self.FLIGHT
        routes = torch.where(
            modes == self.LANDING,
            torch.full_like(modes, self.LANDING_ROUTE),
            torch.full_like(modes, self.TAKEOFF_ROUTE),
        )
        pure_drive = torch.zeros(count, dtype=torch.bool, device=env.device)
        if randomized and not sequence_test:
            pure_drive = torch.rand(count, device=env.device) < float(
                self.cfg.pure_drive_fraction
            )
            # Pure DRIVE owns this reset; it cannot simultaneously be the
            # landing-to-takeoff diagnostic branch.
            landing_to_takeoff[pure_drive] = False
            modes[pure_drive] = self.DRIVE
        pure_drive_profile = torch.rand(count, device=env.device)
        stationary_pure_drive = pure_drive & (
            pure_drive_profile < float(self.cfg.pure_drive_stationary_fraction)
        )
        straight_pure_drive = pure_drive & (
            pure_drive_profile >= float(self.cfg.pure_drive_stationary_fraction)
        ) & (
            pure_drive_profile
            < float(self.cfg.pure_drive_stationary_fraction)
            + float(self.cfg.pure_drive_straight_fraction)
        )
        curved_pure_drive = pure_drive & (~stationary_pure_drive) & (~straight_pure_drive)
        pure_flight = torch.zeros(count, dtype=torch.bool, device=env.device)
        if randomized and not sequence_test:
            available = ~pure_drive
            conditional_fraction = float(self.cfg.pure_flight_fraction) / (
                1.0 - float(self.cfg.pure_drive_fraction)
            )
            pure_flight = available & (
                torch.rand(count, device=env.device) < conditional_fraction
            )
            landing_to_takeoff[pure_flight] = False
            modes[pure_flight] = self.FLIGHT
        routes[pure_drive] = self.LANDING_ROUTE
        env._combined_pure_drive[env_ids] = pure_drive
        env._combined_pure_drive_stationary[env_ids] = stationary_pure_drive
        env._combined_stationary_initial_error_sign[env_ids] = 0.0
        env._combined_pure_drive_straight[env_ids] = straight_pure_drive
        env._combined_pure_drive_curved[env_ids] = curved_pure_drive
        drive_curve_active = (modes == self.DRIVE) | landing_to_takeoff
        curve_active = drive_curve_active | pure_flight
        vertical_trajectory = ~(pure_drive | pure_flight)
        if randomized:
            vertical_trajectory &= (
                torch.rand(count, device=env.device) < float(self.cfg.vertical_trajectory_fraction)
            )
        vertical_trajectory[landing_to_takeoff] = False
        flight_reset = modes == self.FLIGHT
        routes[flight_reset] = torch.randint(0, 2, (int(flight_reset.sum()),), device=env.device)
        routes[pure_flight] = self.TAKEOFF_ROUTE
        if training_overlay:
            routes[flight_reset & ~pure_flight] = self.LANDING_ROUTE
        routes[landing_to_takeoff] = self.LANDING_ROUTE
        env._combined_route[env_ids] = routes
        env._combined_expected_landing[env_ids] = (
            (routes == self.LANDING_ROUTE) & (~pure_drive)
        )
        env._combined_landing_to_takeoff[env_ids] = landing_to_takeoff
        env._combined_drive_curve_active[env_ids] = drive_curve_active
        env._combined_transition_takeoff_reached[env_ids] = False
        env._combined_vertical_trajectory[env_ids] = vertical_trajectory

        origins = env._terrain.env_origins[env_ids]
        heading = torch.empty(count, device=env.device).uniform_(-pi, pi) if randomized else torch.zeros(count, device=env.device)
        direction = torch.stack((torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=1)
        drive_speed = (
            torch.empty(count, 1, device=env.device).uniform_(*self.cfg.drive_speed_range)
            if randomized
            else torch.ones(count, 1, device=env.device)
        )
        drive_speed[vertical_trajectory] = 0.0
        transition_sequence = sequence_test | landing_to_takeoff
        drive_speed[transition_sequence] = 0.25
        drive_start_velocity = drive_speed * direction
        # Flip the drive reference so the vehicle travels the other way along
        # the same heading. This is a TEST of ATMO_SPEC.forward_yaw_offset = pi
        # (vehicle_specs.py:285, ATMO only -- M4TII uses the default 0): the yaw
        # reference is `heading - forward_yaw_offset`, so ATMO is commanded to
        # face 180 deg from its direction of travel. If reversing the reference
        # makes the drive look right, the offset is the real defect and this
        # flag is not the fix.
        if os.environ.get("M4_BENCH_DRIVE_REVERSE") == "1":
            drive_start_velocity = -drive_start_velocity
        takeoff_start_velocity = drive_start_velocity
        drive_duration = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.drive_duration_range)
        takeoff_duration = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.takeoff_duration_range)
        flight_duration = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.flight_duration_range)
        if training_overlay:
            flight_duration.clamp_max_(4.0)
        # Keep the targeted landing-to-takeoff transition inside the existing
        # combined-task horizon: 0.5 s flight, 3 s landing, 1.5 s drive,
        # 3 s takeoff, then continued flight.
        transition_drive_duration_s = 1.5
        transition_takeoff_duration_s = 3.0
        ground_height = float(env.vehicle.spec.ground_reference_height)
        drive_reference_height = torch.full(
            (count,), ground_height, device=env.device
        )
        if randomized:
            height_jitter = float(self.cfg.drive_reference_height_randomization_m)
            drive_reference_height += torch.empty(
                count, device=env.device
            ).uniform_(-height_jitter, height_jitter)
        spawn = origins.clone()
        spawn[:, 2] += drive_reference_height
        liftoff = spawn + takeoff_start_velocity * drive_duration
        distance = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.flight_xy_distance_range)
        distance[vertical_trajectory] = 0.0
        height = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.flight_height_range)
        landing_duration = 3.0 + height
        if torch.any(landing_to_takeoff) and not sequence_test:
            landing_duration[landing_to_takeoff] = 3.0
            drive_duration[landing_to_takeoff] = transition_drive_duration_s
            takeoff_duration[landing_to_takeoff] = transition_takeoff_duration_s
            flight_duration[landing_to_takeoff] = 0.5
        if sequence_test:
            # Diagnostic profile: every reset is an unannounced landing-route
            # sequence. The observation still contains only the current mode.
            # These pins OVERWRITE the cfg-sampled durations three lines above,
            # so takeoff_duration_range / flight_duration_range have no effect
            # under the sequence test. M4_BENCH_PHASE_SCALE rescales the landing
            # and takeoff legs so the profile can be slowed without unpinning
            # the sequence. Note the trained landing is 3.0 + height, so the
            # pinned 3.0 is already SHORTER than anything training used.
            _scale = float(os.environ.get("M4_BENCH_PHASE_SCALE", "1.0"))
            flight_duration[:] = 0.5
            landing_duration[:] = 3.0 * _scale
            drive_duration[:] = 1.5
            takeoff_duration[:] = 3.0 * _scale

        curve_speed = torch.zeros(count, device=env.device)
        drive_curve_count = int(drive_curve_active.sum())
        if drive_curve_count > 0:
            drive_magnitude = torch.empty(drive_curve_count, device=env.device).uniform_(
                *self.cfg.pure_drive_speed_range
            )
            drive_sign = torch.where(
                torch.rand(drive_curve_count, device=env.device) < 0.5,
                -torch.ones_like(drive_magnitude),
                torch.ones_like(drive_magnitude),
            )
            curve_speed[drive_curve_active] = drive_sign * drive_magnitude
            curve_speed[stationary_pure_drive] = 0.0
        pure_flight_count = int(pure_flight.sum())
        if pure_flight_count > 0:
            curve_speed[pure_flight] = torch.empty(pure_flight_count, device=env.device).uniform_(
                *self.cfg.pure_flight_speed_range
            )
        lateral_amplitude = torch.zeros(count, device=env.device)
        lateral_wavelength = torch.ones(count, device=env.device)
        if drive_curve_count > 0:
            lateral_amplitude[drive_curve_active] = torch.empty(drive_curve_count, device=env.device).uniform_(
                *self.cfg.pure_drive_lateral_amplitude_range
            )
            lateral_wavelength[drive_curve_active] = torch.empty(drive_curve_count, device=env.device).uniform_(
                *self.cfg.pure_drive_lateral_wavelength_range
            )
            lateral_amplitude[stationary_pure_drive | straight_pure_drive] = 0.0
        if pure_flight_count > 0:
            lateral_amplitude[pure_flight] = torch.empty(pure_flight_count, device=env.device).uniform_(
                *self.cfg.pure_flight_lateral_amplitude_range
            )
            lateral_wavelength[pure_flight] = torch.empty(pure_flight_count, device=env.device).uniform_(
                *self.cfg.pure_flight_lateral_wavelength_range
            )
        vertical_amplitude = torch.zeros(count, device=env.device)
        vertical_wavelength = torch.ones(count, device=env.device)
        vertical_amplitude[pure_flight] = float(self.cfg.pure_flight_vertical_amplitude)
        if pure_flight_count > 0:
            vertical_wavelength[pure_flight] = torch.empty(pure_flight_count, device=env.device).uniform_(
                *self.cfg.pure_flight_vertical_wavelength_range
            )
        curve_origin = origins.clone()
        curve_origin[:, 2] += drive_reference_height
        curve_origin[pure_flight, 2] = origins[pure_flight, 2] + height[pure_flight, 0] + vertical_amplitude[pure_flight]
        env._combined_curve_active[env_ids] = curve_active
        env._combined_curve_origin_w[env_ids] = curve_origin
        env._combined_curve_heading[env_ids] = heading
        env._combined_curve_speed[env_ids] = curve_speed
        env._combined_curve_lateral_amplitude[env_ids] = lateral_amplitude
        env._combined_curve_lateral_wavenumber[env_ids] = 2.0 * pi / lateral_wavelength
        env._combined_curve_lateral_phase[env_ids] = torch.empty(count, device=env.device).uniform_(-pi, pi)
        env._combined_curve_vertical_amplitude[env_ids] = vertical_amplitude
        env._combined_curve_vertical_wavenumber[env_ids] = 2.0 * pi / vertical_wavelength
        env._combined_curve_vertical_phase[env_ids] = torch.empty(count, device=env.device).uniform_(-pi, pi)
        if drive_curve_count > 0:
            drive_curve_ids = env_ids[drive_curve_active]
            drive_curve_time = drive_duration[drive_curve_active]
            _, curve_start_velocity, _, _, _, _ = self._curve_reference(
                torch.zeros_like(drive_curve_time), drive_curve_ids
            )
            curve_end_pos, curve_end_velocity, _, _, _, _ = self._curve_reference(
                drive_curve_time, drive_curve_ids
            )
            liftoff[drive_curve_active] = curve_end_pos
            drive_start_velocity[drive_curve_active] = curve_start_velocity
            takeoff_start_velocity[drive_curve_active] = curve_end_velocity
        takeoff_heading = heading + torch.empty(count, device=env.device).uniform_(
            -self.cfg.takeoff_end_heading_offset,
            self.cfg.takeoff_end_heading_offset,
        )
        takeoff_direction = torch.stack(
            (torch.cos(takeoff_heading), torch.sin(takeoff_heading), torch.zeros_like(takeoff_heading)), dim=1
        )
        takeoff_end = liftoff + distance * takeoff_direction
        takeoff_end[:, 2:3] += height
        takeoff_end_speed = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.takeoff_end_speed_range)
        takeoff_end_speed[vertical_trajectory] = 0.0
        takeoff_end_speed[transition_sequence] = 0.25
        takeoff_end_velocity = takeoff_end_speed * takeoff_direction
        flight_end = takeoff_end + takeoff_end_velocity * flight_duration

        landing_heading = heading + torch.empty(count, device=env.device).uniform_(
            -self.cfg.takeoff_end_heading_offset,
            self.cfg.takeoff_end_heading_offset,
        )
        landing_direction = torch.stack(
            (torch.cos(landing_heading), torch.sin(landing_heading), torch.zeros_like(landing_heading)), dim=1
        )
        landing_start = origins - 0.5 * distance * landing_direction
        landing_start[:, 2:3] += height
        landing_pos = origins + 0.5 * distance * landing_direction
        landing_pos[:, 2] += ground_height
        landing_start_speed = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.takeoff_end_speed_range)
        ground_speed = torch.empty(count, 1, device=env.device).uniform_(*self.cfg.landing_end_speed_range)
        landing_start_speed[vertical_trajectory] = 0.0
        ground_speed[vertical_trajectory] = 0.0
        landing_start_speed[transition_sequence] = 0.25
        ground_speed[transition_sequence] = 0.25
        landing_start_velocity = landing_start_speed * direction
        ground_velocity = ground_speed * landing_direction
        takeoff_end_velocity[transition_sequence] = ground_velocity[transition_sequence]

        env._combined_spawn_pos_w[env_ids] = spawn
        env._combined_drive_start_velocity_w[env_ids] = drive_start_velocity
        env._combined_drive_velocity_w[env_ids] = takeoff_start_velocity
        env._combined_drive_duration[env_ids] = drive_duration
        env._combined_takeoff_duration[env_ids] = takeoff_duration
        env._combined_flight_duration[env_ids] = flight_duration
        env._combined_landing_duration[env_ids] = landing_duration
        env._combined_liftoff_pos_w[env_ids] = liftoff
        env._combined_takeoff_end_pos_w[env_ids] = takeoff_end
        env._combined_takeoff_start_velocity_w[env_ids] = takeoff_start_velocity
        env._combined_takeoff_end_velocity_w[env_ids] = takeoff_end_velocity
        env._combined_landing_start_pos_w[env_ids] = landing_start
        env._combined_landing_pos_w[env_ids] = landing_pos
        env._combined_landing_start_velocity_w[env_ids] = landing_start_velocity
        env._combined_ground_velocity_w[env_ids] = ground_velocity
        forward_yaw_offset = float(env.vehicle.spec.forward_yaw_offset)
        # Rotate ONLY the reference yaw. Changing spec.forward_yaw_offset also
        # feeds heading_yaw_from_quat (line ~2160), which rotates the entire
        # observation frame -- every position and velocity error then arrives
        # 180 deg out and the policy flips over immediately. This touches the
        # yaw TARGET alone; the observation frame is untouched.
        ref_yaw_rot = float(os.environ.get("M4_BENCH_REF_YAW_ROT", "0.0"))
        env._combined_drive_yaw[env_ids] = self._wrap_to_pi(
            torch.where(routes == self.TAKEOFF_ROUTE, heading, landing_heading)
            - forward_yaw_offset + ref_yaw_rot
        )
        env._combined_flight_yaw[env_ids] = self._wrap_to_pi(
            torch.where(routes == self.TAKEOFF_ROUTE, takeoff_heading, landing_heading)
            - forward_yaw_offset + ref_yaw_rot
        )

        phase_time = torch.zeros(count, 1, device=env.device)
        takeoff_mask = modes == self.TAKEOFF
        takeoff_flight = flight_reset & (routes == self.TAKEOFF_ROUTE)
        if torch.any(takeoff_flight):
            phase_time[takeoff_flight] = (
                torch.rand_like(phase_time[takeoff_flight]) * flight_duration[takeoff_flight]
            )
        landing_flight = flight_reset & (routes == self.LANDING_ROUTE)
        phase_time[landing_flight] = torch.rand_like(phase_time[landing_flight]) * flight_duration[landing_flight]
        phase_time[pure_flight] = 0.0
        landing_mask = modes == self.LANDING
        phase_time[landing_mask] = (
            0.10 * torch.rand_like(phase_time[landing_mask])
        ) * landing_duration[landing_mask]
        phase_time[landing_to_takeoff] = 0.0
        if sequence_test:
            phase_time[:] = 0.0
        env._combined_phase_start_elapsed[env_ids] = env._time_elapsed[env_ids].unsqueeze(1)
        env._combined_phase_initial_time[env_ids] = phase_time

        env._combined_mode[env_ids] = modes
        env._combined_yaw[env_ids] = torch.where(
            modes == self.DRIVE,
            env._combined_drive_yaw[env_ids],
            env._combined_flight_yaw[env_ids],
        )
        ref_pos, ref_vel, _ = self._reference_state(0.0, env_ids)
        ref_yaw, ref_yaw_rate, _ = self._reference_yaw_state(0.0, env_ids)
        root_state = env._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = ref_pos
        root_state[:, 7:10] = ref_vel
        root_state[:, 10:13] = 0.0
        root_state[curve_active, 12] = ref_yaw_rate[curve_active]
        root_state[:, :2] += (
            torch.empty(count, 2, device=env.device).uniform_(-0.15 * error_scale, 0.15 * error_scale)
            if randomized
            else 0.0
        )
        # Every grounded drive reset draws from the same error envelope, whether
        # or not a takeoff follows. Conditioning this on pure_drive alone leaked
        # the upcoming takeoff into the very first observation.
        drive_reset = modes == self.DRIVE
        if torch.any(drive_reset):
            drive_reset_count = int(drive_reset.sum())
            error_angle = torch.empty(drive_reset_count, device=env.device).uniform_(-pi, pi)
            max_position_error = min(
                float(self.cfg.pure_drive_position_error_max) * error_scale,
                0.8 * self._trajectory_deviation_threshold(),
            )
            error_radius = (
                torch.sqrt(torch.rand(drive_reset_count, device=env.device))
                * max_position_error
            )
            position_error = torch.stack(
                (error_radius * torch.cos(error_angle), error_radius * torch.sin(error_angle)), dim=1
            )
            root_state[drive_reset, :2] = ref_pos[drive_reset, :2] - position_error
            root_state[drive_reset, 7:10] += torch.empty(
                drive_reset_count, 3, device=env.device
            ).uniform_(
                -float(self.cfg.pure_drive_velocity_error_max) * error_scale,
                float(self.cfg.pure_drive_velocity_error_max) * error_scale,
            )
            root_state[drive_reset, 9] = 0.0
        roll = torch.zeros(count, device=env.device)
        pitch = torch.zeros(count, device=env.device)
        airborne_reset = (modes == self.FLIGHT) | (modes == self.LANDING)
        # Capped well inside the attitude termination angle: a reset that already
        # sits at the failure boundary trains nothing but an immediate death.
        attitude_range = min(
            float(self.cfg.hover_attitude_range) * error_scale,
            0.6 * float(self.cfg.attitude_termination_angle),
        )
        roll[airborne_reset] = torch.empty(int(airborne_reset.sum()), device=env.device).uniform_(
            -attitude_range, attitude_range
        )
        pitch[airborne_reset] = torch.empty(int(airborne_reset.sum()), device=env.device).uniform_(
            -attitude_range, attitude_range
        )
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, ref_yaw)
        if torch.any(drive_reset):
            drive_yaw_error = torch.empty(int(drive_reset.sum()), device=env.device).uniform_(
                -float(self.cfg.pure_drive_yaw_error_max) * error_scale,
                float(self.cfg.pure_drive_yaw_error_max) * error_scale,
            )
            root_state[drive_reset, 3:7] = quat_from_euler_xyz(
                roll[drive_reset],
                pitch[drive_reset],
                self._wrap_to_pi(ref_yaw[drive_reset] + drive_yaw_error),
            )
        if torch.any(stationary_pure_drive):
            stationary_count = int(stationary_pure_drive.sum())
            stationary_sign = torch.where(
                torch.rand(stationary_count, device=env.device) < 0.5,
                -torch.ones(stationary_count, device=env.device),
                torch.ones(stationary_count, device=env.device),
            )
            stationary_error = torch.empty(
                stationary_count, device=env.device
            ).uniform_(*self.cfg.stationary_drive_error_range_m)
            root_state[stationary_pure_drive, :2] = (
                ref_pos[stationary_pure_drive, :2]
                - stationary_sign[:, None]
                * stationary_error[:, None]
                * direction[stationary_pure_drive, :2]
            )
            root_state[stationary_pure_drive, 7:13] = 0.0
            root_state[stationary_pure_drive, 3:7] = quat_from_euler_xyz(
                roll[stationary_pure_drive],
                pitch[stationary_pure_drive],
                ref_yaw[stationary_pure_drive],
            )
            env._combined_stationary_initial_error_sign[
                env_ids[stationary_pure_drive]
            ] = stationary_sign
        root_state[~airborne_reset, 2] = origins[~airborne_reset, 2] + ground_height
        if torch.any(landing_to_takeoff):
            # Branch from the existing landing-state envelope with small
            # touchdown residuals. The policy must carry these through the
            # grounded drive instead of seeing a clean reset before takeoff.
            root_state[landing_to_takeoff, 7:10] += torch.empty(
                int(landing_to_takeoff.sum()), 3, device=env.device
            ).uniform_(-0.25 * error_scale, 0.25 * error_scale)
            root_state[landing_to_takeoff, 12] += torch.empty(
                int(landing_to_takeoff.sum()), device=env.device
            ).uniform_(-0.25 * error_scale, 0.25 * error_scale)

        joint_pos, joint_vel = env.vehicle.deterministic_joint_state(env_ids)
        ground_reset = (modes == self.DRIVE) | (modes == self.TAKEOFF)
        if torch.any(ground_reset) and env.vehicle.spec.landing_joint_config:
            ground_joint_pos = joint_pos[ground_reset].clone()
            ground_joint_vel = joint_vel[ground_reset].clone()
            env.vehicle.set_joint_config_state(
                ground_joint_pos,
                ground_joint_vel,
                env_ids[ground_reset],
                env.vehicle.spec.landing_joint_config,
                0.0,
            )
            joint_pos[ground_reset] = ground_joint_pos
            joint_vel[ground_reset] = ground_joint_vel
        airborne_reset_mask = ~ground_reset
        if torch.any(airborne_reset_mask) and env.vehicle.spec.hover_joint_config:
            airborne_joint_pos = joint_pos[airborne_reset_mask].clone()
            airborne_joint_vel = joint_vel[airborne_reset_mask].clone()
            env.vehicle.set_joint_config_state(
                airborne_joint_pos,
                airborne_joint_vel,
                env_ids[airborne_reset_mask],
                env.vehicle.spec.hover_joint_config,
                0.0,
            )
            joint_pos[airborne_reset_mask] = airborne_joint_pos
            joint_vel[airborne_reset_mask] = airborne_joint_vel
        morph = torch.full((count, 1), pi / 2, device=env.device)
        drive_mask = drive_reset
        # Pure drive used to start at exactly pi/2 while drive-then-takeoff
        # started below it, which made the observed tilt angle a near-perfect
        # predictor of the upcoming takeoff. Both draw from one envelope now.
        morph[drive_mask] = torch.empty(int(drive_mask.sum()), 1, device=env.device).uniform_(70.0 * pi / 180.0, pi / 2)
        morph[takeoff_mask] = torch.empty(int(takeoff_mask.sum()), 1, device=env.device).uniform_(
            45.0 * pi / 180.0,
            80.0 * pi / 180.0,
        )
        flight_mask = modes == self.FLIGHT
        morph[flight_mask] = torch.empty(int(flight_mask.sum()), 1, device=env.device).uniform_(0.0, 5.0 * pi / 180.0)
        landing_mask = modes == self.LANDING
        morph[landing_mask] = torch.empty(int(landing_mask.sum()), 1, device=env.device).uniform_(0.0, pi / 6.0)
        # A fraction of airborne resets start mid-morph instead of near the
        # flight tilt: the vehicle is already flying with its arms part-way to
        # the drive stop and has to fly out of it. Applied after the per-mode
        # envelopes so it overrides them for the selected envs only.
        posture_recovery = torch.zeros(count, dtype=torch.bool, device=env.device)
        if randomized:
            posture_low, posture_high = (
                float(value) for value in self.cfg.airborne_posture_recovery_range
            )
            if posture_high < posture_low:
                raise ValueError(
                    "airborne_posture_recovery_range must satisfy min <= max, "
                    f"got {(posture_low, posture_high)}"
                )
            posture_recovery = airborne_reset & (
                torch.rand(count, device=env.device)
                < float(self.cfg.airborne_posture_recovery_fraction)
            )
            if torch.any(posture_recovery):
                morph[posture_recovery] = torch.empty(
                    int(posture_recovery.sum()), 1, device=env.device
                ).uniform_(posture_low, posture_high)
        morph_state = morph
        morph_recovery = torch.zeros(count, dtype=torch.bool, device=env.device)
        if env.vehicle.spec.morph_joint_group is not None:
            runtime = env.vehicle.joint_groups[env.vehicle.spec.morph_joint_group]
            morph_state = morph.expand(-1, runtime.target_pos.shape[1]).clone()
            if randomized and "tilt_balance" in env.vehicle.action_schema.slices:
                morph_authority = env.action_authority(
                    env.cfg.morph_bias_start_stage,
                    env.cfg.morph_balance_authority_ramp_epochs,
                )
                if morph_authority > 0.0:
                    morph_recovery = torch.rand(count, device=env.device) < float(self.cfg.morph_balance_recovery_fraction)
                    perturbation = torch.empty_like(morph_state).uniform_(-1.0, 1.0)
                    perturbation -= torch.mean(perturbation, dim=1, keepdim=True)
                    perturbation *= float(self.cfg.morph_balance_recovery_max_angle) * morph_authority
                    morph_state[morph_recovery] = torch.clamp(
                        morph_state[morph_recovery] + perturbation[morph_recovery],
                        0.0,
                        pi / 2,
                    )
            env.vehicle.set_joint_group_state_values(
                joint_pos, joint_vel, env_ids, env.vehicle.spec.morph_joint_group, morph_state
            )
        thrust_center_recovery = torch.zeros(count, dtype=torch.bool, device=env.device)
        if randomized and env.vehicle.spec.leg_joint_group is not None and "thrust_center_xy" in env.vehicle.action_schema.slices:
            thrust_center_authority = env.action_authority(
                env.cfg.thrust_center_start_stage,
                env.cfg.thrust_center_authority_ramp_epochs,
            )
            if thrust_center_authority > 0.0:
                thrust_center_recovery = airborne_reset & (
                    torch.rand(count, device=env.device) < float(self.cfg.thrust_center_recovery_fraction)
                )
            if torch.any(thrust_center_recovery):
                recovery_joint_pos = joint_pos[thrust_center_recovery].clone()
                recovery_joint_vel = joint_vel[thrust_center_recovery].clone()
                env.vehicle.seed_thrust_center_joint_group(
                    recovery_joint_pos,
                    recovery_joint_vel,
                    env_ids[thrust_center_recovery],
                    (-1.0, 1.0),
                    (-1.0, 1.0),
                    float(self.cfg.thrust_center_recovery_max_angle) * thrust_center_authority,
                )
                joint_pos[thrust_center_recovery] = recovery_joint_pos
                joint_vel[thrust_center_recovery] = recovery_joint_vel
        env._combined_morph_recovery_reset[env_ids] = morph_recovery
        env._combined_thrust_center_recovery_reset[env_ids] = thrust_center_recovery
        env._combined_takeoff_start_morph[env_ids] = pi / 2
        mean_morph = torch.mean(morph_state, dim=1, keepdim=True)
        env._combined_takeoff_start_morph[env_ids[takeoff_mask]] = mean_morph[takeoff_mask, 0]
        reset_actions = env._reset_policy_actions[env_ids]
        reset_actions.fill_(-1.0)
        action_terms = env.vehicle.action_schema.split(reset_actions)
        for name in ("roll_pitch_yaw", "wheel_speed", "tilt_balance", "thrust_center_xy"):
            if name in action_terms:
                action_terms[name].zero_()
        ref_accel = self._reference_state(0.0, env_ids)[2]
        reset_throttle = torch.zeros(count, 1, device=env.device)
        hover_throttle = env.vehicle.hover_collective_throttle()
        reset_throttle[takeoff_mask] = hover_throttle * torch.cos(mean_morph[takeoff_mask])
        warm_start = takeoff_mask | airborne_reset
        reset_throttle[airborne_reset] = (
            hover_throttle
            * (1.0 + ref_accel[airborne_reset, 2:3] / 9.81)
            / torch.cos(mean_morph[airborne_reset])
        ).clamp(0.05, 0.95)
        action_terms["lift"][warm_start] = 2.0 * reset_throttle[warm_start] - 1.0
        env._combined_reset_throttle[env_ids] = reset_throttle
        env._desired_pos_w[env_ids] = torch.where(
            (routes == self.TAKEOFF_ROUTE).unsqueeze(1), flight_end, landing_pos
        )
        env._virtual_xy_offset_w[env_ids] = 0.0
        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)

    def _phase_time(self, time_offset_s=0.0, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        env = self.env
        ids = env._robot._ALL_INDICES if env_ids is None else env_ids
        time = (
            env._time_elapsed[ids].unsqueeze(1)
            - env._combined_phase_start_elapsed[ids]
            + env._combined_phase_initial_time[ids]
        )
        if isinstance(time_offset_s, torch.Tensor):
            value = time_offset_s.to(env.device)
            if value.ndim == 1:
                value = value.unsqueeze(1)
            return time + value[ids] if value.shape[0] == env.num_envs else time + value
        return time + max(float(time_offset_s), 0.0)

    def _curve_reference(self, time: torch.Tensor, ids: torch.Tensor):
        env = self.env
        scalar_time = time[:, 0]
        speed = env._combined_curve_speed[ids]
        arc_distance = speed * scalar_time
        heading = env._combined_curve_heading[ids]
        forward = torch.stack((torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=1)
        lateral = torch.stack((-torch.sin(heading), torch.cos(heading), torch.zeros_like(heading)), dim=1)

        lateral_amplitude = env._combined_curve_lateral_amplitude[ids]
        lateral_wavenumber = env._combined_curve_lateral_wavenumber[ids]
        lateral_phase = env._combined_curve_lateral_phase[ids]
        vertical_amplitude = env._combined_curve_vertical_amplitude[ids]
        vertical_wavenumber = env._combined_curve_vertical_wavenumber[ids]
        vertical_phase = env._combined_curve_vertical_phase[ids]

        # Invert centerline arc length so curve speed is the true 3D reference speed.
        metric_rms = torch.sqrt(
            1.0
            + 0.5 * torch.square(lateral_amplitude * lateral_wavenumber)
            + 0.5 * torch.square(vertical_amplitude * vertical_wavenumber)
        )
        distance = arc_distance / metric_rms
        samples = self._curve_arc_samples
        for _ in range(5):
            sample_distance = distance.unsqueeze(1) * samples.unsqueeze(0)
            sample_lateral_slope = (lateral_amplitude * lateral_wavenumber).unsqueeze(1) * torch.cos(
                lateral_wavenumber.unsqueeze(1) * sample_distance + lateral_phase.unsqueeze(1)
            )
            sample_vertical_slope = (vertical_amplitude * vertical_wavenumber).unsqueeze(1) * torch.cos(
                vertical_wavenumber.unsqueeze(1) * sample_distance + vertical_phase.unsqueeze(1)
            )
            sample_metric = torch.sqrt(
                1.0 + torch.square(sample_lateral_slope) + torch.square(sample_vertical_slope)
            )
            arc_length = distance / 96.0 * (
                sample_metric[:, 0]
                + sample_metric[:, -1]
                + 4.0 * sample_metric[:, 1:-1:2].sum(dim=1)
                + 2.0 * sample_metric[:, 2:-1:2].sum(dim=1)
            )
            distance_metric = sample_metric[:, -1]
            distance -= (arc_length - arc_distance) / distance_metric

        lateral_angle = lateral_wavenumber * distance + lateral_phase
        lateral_offset = lateral_amplitude * (torch.sin(lateral_angle) - torch.sin(lateral_phase))
        lateral_slope = lateral_amplitude * lateral_wavenumber * torch.cos(lateral_angle)
        lateral_curvature = -lateral_amplitude * torch.square(lateral_wavenumber) * torch.sin(lateral_angle)
        lateral_curvature_rate = (
            -lateral_amplitude * torch.pow(lateral_wavenumber, 3) * torch.cos(lateral_angle)
        )
        vertical_angle = vertical_wavenumber * distance + vertical_phase
        vertical_offset = vertical_amplitude * torch.sin(vertical_angle)
        vertical_slope = vertical_amplitude * vertical_wavenumber * torch.cos(vertical_angle)
        vertical_curvature = -vertical_amplitude * torch.square(vertical_wavenumber) * torch.sin(vertical_angle)

        up = torch.zeros_like(forward)
        up[:, 2] = 1.0
        position = (
            env._combined_curve_origin_w[ids]
            + forward * distance.unsqueeze(1)
            + lateral * lateral_offset.unsqueeze(1)
            + up * vertical_offset.unsqueeze(1)
        )
        tangent = forward + lateral * lateral_slope.unsqueeze(1) + up * vertical_slope.unsqueeze(1)
        distance_metric = torch.sqrt(
            1.0 + torch.square(lateral_slope) + torch.square(vertical_slope)
        )
        distance_rate = speed / distance_metric
        distance_metric_rate = (
            lateral_slope * lateral_curvature + vertical_slope * vertical_curvature
        ) / distance_metric
        distance_accel = -torch.square(speed) * distance_metric_rate / torch.pow(distance_metric, 3)
        velocity = distance_rate.unsqueeze(1) * tangent
        acceleration = (
            torch.square(distance_rate).unsqueeze(1)
            * (lateral * lateral_curvature.unsqueeze(1) + up * vertical_curvature.unsqueeze(1))
            + distance_accel.unsqueeze(1) * tangent
        )

        horizontal_tangent = forward + lateral * lateral_slope.unsqueeze(1)
        yaw = self._wrap_to_pi(
            torch.atan2(horizontal_tangent[:, 1], horizontal_tangent[:, 0])
            - float(env.vehicle.spec.forward_yaw_offset)
        )
        tangent_norm_sq = 1.0 + torch.square(lateral_slope)
        yaw_distance_rate = lateral_curvature / tangent_norm_sq
        yaw_rate = yaw_distance_rate * distance_rate
        yaw_accel = (
            (
                lateral_curvature_rate / tangent_norm_sq
                - 2.0
                * lateral_slope
                * torch.square(lateral_curvature)
                / torch.square(tangent_norm_sq)
            )
            * torch.square(distance_rate)
            + yaw_distance_rate * distance_accel
        )
        return position, velocity, acceleration, yaw, yaw_rate, yaw_accel

    def _reference_yaw_state(self, time_offset_s=0.0, env_ids: torch.Tensor | None = None):
        env = self.env
        ids = env._robot._ALL_INDICES if env_ids is None else env_ids
        time = self._phase_time(time_offset_s, ids)
        mode = env._combined_mode[ids]
        drive_curve_active = env._combined_drive_curve_active[ids]
        curve_active = env._combined_curve_active[ids] & (
            (drive_curve_active & (mode == self.DRIVE))
            | (~drive_curve_active & (mode == self.FLIGHT))
        )
        yaw = env._combined_yaw[ids].clone()
        yaw_rate = torch.zeros_like(yaw)
        yaw_accel = torch.zeros_like(yaw)
        _, _, _, curve_yaw, curve_yaw_rate, curve_yaw_accel = self._curve_reference(
            time[curve_active], ids[curve_active]
        )
        yaw[curve_active] = curve_yaw
        yaw_rate[curve_active] = curve_yaw_rate
        yaw_accel[curve_active] = curve_yaw_accel
        return yaw, yaw_rate, yaw_accel

    def _reference_state(self, time_offset_s=0.0, env_ids: torch.Tensor | None = None):
        env = self.env
        ids = env._robot._ALL_INDICES if env_ids is None else env_ids
        time = self._phase_time(time_offset_s, ids)
        mode = env._combined_mode[ids].unsqueeze(1)
        takeoff_route = (env._combined_route[ids] == self.TAKEOFF_ROUTE).unsqueeze(1)
        drive_duration = env._combined_drive_duration[ids]
        takeoff_duration = env._combined_takeoff_duration[ids]
        flight_duration = env._combined_flight_duration[ids]
        landing_duration = env._combined_landing_duration[ids]
        zero = torch.zeros_like(env._combined_drive_velocity_w[ids])

        drive_time = torch.minimum(torch.clamp(time, min=0.0), drive_duration)
        takeoff_drive_velocity = env._combined_drive_velocity_w[ids]
        takeoff_drive_pos = env._combined_spawn_pos_w[ids] + takeoff_drive_velocity * drive_time
        takeoff_drive_vel = takeoff_drive_velocity
        takeoff_drive_accel = zero
        drive_overrun = torch.clamp(time - drive_duration, min=0.0)
        takeoff_drive_pos = takeoff_drive_pos + takeoff_drive_velocity * drive_overrun
        landing_drive_pos = env._combined_landing_pos_w[ids] + env._combined_ground_velocity_w[ids] * torch.clamp(time, min=0.0)
        drive_pos = torch.where(takeoff_route, takeoff_drive_pos, landing_drive_pos)
        drive_vel = torch.where(takeoff_route, takeoff_drive_vel, env._combined_ground_velocity_w[ids])
        drive_accel = torch.where(takeoff_route, takeoff_drive_accel, zero)

        takeoff_time = torch.minimum(torch.clamp(time, min=0.0), takeoff_duration)
        takeoff_pos, takeoff_vel, takeoff_accel = self._trajectory_segment(
            env._combined_liftoff_pos_w[ids],
            env._combined_takeoff_end_pos_w[ids],
            env._combined_takeoff_start_velocity_w[ids],
            env._combined_takeoff_end_velocity_w[ids],
            takeoff_time,
            takeoff_duration,
        )
        takeoff_overrun = torch.clamp(time - takeoff_duration, min=0.0)
        takeoff_pos = takeoff_pos + env._combined_takeoff_end_velocity_w[ids] * takeoff_overrun
        takeoff_vel = torch.where(
            time <= takeoff_duration, takeoff_vel, env._combined_takeoff_end_velocity_w[ids]
        )
        takeoff_accel = torch.where(time <= takeoff_duration, takeoff_accel, zero)
        # Prep window: keep rolling along the ground at the drive exit velocity.
        # At time == -prep this lands exactly on the pre-shift liftoff point, so
        # position and velocity stay continuous across the drive handover.
        takeoff_prep = time < 0.0
        takeoff_start_velocity = env._combined_takeoff_start_velocity_w[ids]
        takeoff_pos = torch.where(
            takeoff_prep,
            env._combined_liftoff_pos_w[ids] + takeoff_start_velocity * time,
            takeoff_pos,
        )
        takeoff_vel = torch.where(takeoff_prep, takeoff_start_velocity, takeoff_vel)
        takeoff_accel = torch.where(takeoff_prep, zero, takeoff_accel)

        takeoff_flight_time = torch.clamp(time, min=0.0)
        takeoff_flight_pos = (
            env._combined_takeoff_end_pos_w[ids]
            + env._combined_takeoff_end_velocity_w[ids] * takeoff_flight_time
        )
        takeoff_flight_vel = env._combined_takeoff_end_velocity_w[ids]
        takeoff_flight_accel = zero
        landing_flight_time = torch.minimum(torch.clamp(time, min=0.0), flight_duration)
        landing_flight_pos = env._combined_landing_start_pos_w[ids] - env._combined_landing_start_velocity_w[ids] * (
            flight_duration - landing_flight_time
        )
        landing_flight_pos = landing_flight_pos + env._combined_landing_start_velocity_w[ids] * torch.clamp(
            time - flight_duration, min=0.0
        )
        landing_flight_vel = env._combined_landing_start_velocity_w[ids]
        flight_pos = torch.where(takeoff_route, takeoff_flight_pos, landing_flight_pos)
        flight_vel = torch.where(takeoff_route, takeoff_flight_vel, landing_flight_vel)
        flight_accel = torch.where(takeoff_route, takeoff_flight_accel, zero)

        landing_time = torch.minimum(torch.clamp(time, min=0.0), landing_duration)
        landing_pos, landing_vel, landing_accel = self._trajectory_segment(
            env._combined_landing_start_pos_w[ids],
            env._combined_landing_pos_w[ids],
            env._combined_landing_start_velocity_w[ids],
            env._combined_ground_velocity_w[ids],
            landing_time,
            landing_duration,
        )
        landing_overrun = torch.clamp(time - landing_duration, min=0.0)
        landing_pos = landing_pos + env._combined_ground_velocity_w[ids] * landing_overrun
        landing_vel = torch.where(time <= landing_duration, landing_vel, env._combined_ground_velocity_w[ids])
        landing_accel = torch.where(time <= landing_duration, landing_accel, zero)

        ref_pos = torch.where(mode == self.DRIVE, drive_pos, takeoff_pos)
        ref_vel = torch.where(mode == self.DRIVE, drive_vel, takeoff_vel)
        ref_accel = torch.where(mode == self.DRIVE, drive_accel, takeoff_accel)
        ref_pos = torch.where(mode == self.FLIGHT, flight_pos, ref_pos)
        ref_vel = torch.where(mode == self.FLIGHT, flight_vel, ref_vel)
        ref_accel = torch.where(mode == self.FLIGHT, flight_accel, ref_accel)
        ref_pos = torch.where(mode == self.LANDING, landing_pos, ref_pos)
        ref_vel = torch.where(mode == self.LANDING, landing_vel, ref_vel)
        ref_accel = torch.where(mode == self.LANDING, landing_accel, ref_accel)
        drive_curve_active = env._combined_drive_curve_active[ids]
        curve_active = env._combined_curve_active[ids] & (
            (drive_curve_active & (mode[:, 0] == self.DRIVE))
            | (~drive_curve_active & (mode[:, 0] == self.FLIGHT))
        )
        curve_pos, curve_vel, curve_accel, _, _, _ = self._curve_reference(
            time[curve_active], ids[curve_active]
        )
        ref_pos[curve_active] = curve_pos
        ref_vel[curve_active] = curve_vel
        ref_accel[curve_active] = curve_accel
        return (
            torch.nan_to_num(ref_pos, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(ref_vel, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(ref_accel, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def reference_state(self, time_offset_s=0.0):
        return self._reference_state(time_offset_s)

    def _update_mode(self):
        env = self.env
        if self._last_transition_update_step == env._global_env_step:
            return
        self._last_transition_update_step = env._global_env_step
        time = self._phase_time()[:, 0]
        mode = env._combined_mode
        takeoff_route = env._combined_route == self.TAKEOFF_ROUTE
        landing_route = ~takeoff_route
        active = ~(
            (takeoff_route & (mode == self.FLIGHT))
            | (
                landing_route
                & (mode == self.DRIVE)
                & (~env._combined_landing_to_takeoff)
            )
        )
        contacts = self._contact_state()
        invalid_contact = contacts["invalid_contacts"]

        drive_overrun = torch.clamp(time - env._combined_drive_duration[:, 0], min=0.0)
        transition_drive_to_takeoff = (
            active
            & landing_route
            & env._combined_landing_to_takeoff
            & (mode == self.DRIVE)
            & (time >= env._combined_drive_duration[:, 0])
        )
        drive_to_takeoff = (
            active
            & (takeoff_route | transition_drive_to_takeoff)
            & (mode == self.DRIVE)
            & (time >= env._combined_drive_duration[:, 0])
        )

        takeoff_to_flight = (
            active
            & takeoff_route
            & (mode == self.TAKEOFF)
            & (time >= env._combined_takeoff_duration[:, 0])
        )

        flight_overrun = torch.clamp(time - env._combined_flight_duration[:, 0], min=0.0)
        flight_to_landing = (
            active
            & landing_route
            & (mode == self.FLIGHT)
            & (time >= env._combined_flight_duration[:, 0])
        )

        landing_overrun = torch.clamp(time - env._combined_landing_duration[:, 0], min=0.0)
        first_contact = (
            active
            & landing_route
            & (mode == self.LANDING)
            & contacts["any_valid_contacts"]
            & (~env._combined_first_contact_recorded)
        )
        env._combined_first_contact_time[first_contact] = time[first_contact]
        env._combined_first_contact_recorded[first_contact] = True
        # Every landing quality metric is sampled here, at the first wheel down,
        # not when the fourth completes the set. Latching at completion let the
        # policy hold one wheel clear, trim speed, posture and position against
        # the remaining supports, and then place the last wheel at a moment of
        # its own choosing - the metrics measured a staged pose rather than an
        # arrival. Completion still requires all four wheels; it just no longer
        # decides what the landing scores.
        if torch.any(first_contact):
            first_contact_ids = env._robot._ALL_INDICES[first_contact]
            env._combined_final_contact_velocity_w[first_contact] = torch.nan_to_num(
                env._robot.data.root_com_lin_vel_w[first_contact],
                nan=0.0,
                posinf=1e3,
                neginf=-1e3,
            )
            first_contact_ref_pos, _, _ = self._reference_state(env_ids=first_contact_ids)
            reference_distance = torch.linalg.norm(
                first_contact_ref_pos - env._combined_landing_pos_w[first_contact_ids],
                dim=1,
            )
            env._combined_final_contact_reference_distance[first_contact_ids] = torch.where(
                time[first_contact] < env._combined_landing_duration[first_contact_ids, 0],
                reference_distance,
                torch.zeros_like(reference_distance),
            )
            env._combined_first_contact_xy_error[first_contact_ids] = torch.linalg.norm(
                first_contact_ref_pos[:, :2]
                - env._robot.data.root_link_pos_w[first_contact_ids, :2],
                dim=1,
            )
            env._combined_first_contact_tilt[first_contact] = torch.min(
                torch.nan_to_num(env.vehicle.morph_joint_positions()[first_contact], nan=0.0),
                dim=1,
            ).values
        landing_to_drive = (
            active
            & landing_route
            & (mode == self.LANDING)
            & (contacts["valid_contact_count"] >= len(env._valid_contact_ids))
            & (~invalid_contact)
            & self._landing_posture_ok()
        )
        # The landing clock now gates EVERY landing, not just the scripted
        # landing-to-takeoff sequence. Previously the `~landing_to_takeoff` term
        # made this clause vacuously true for ordinary landings, so contact plus
        # posture alone could end the phase at any moment -- and a policy that
        # dropped early got to skip the rest of the descent profile.
        landing_to_drive &= time >= (
            env._combined_landing_duration[:, 0]
            - float(self.cfg.landing_early_tolerance_s)
        )

        transitioned = drive_to_takeoff | takeoff_to_flight | flight_to_landing | landing_to_drive
        drive_velocity = torch.where(
            takeoff_route.unsqueeze(1),
            env._combined_drive_velocity_w,
            env._combined_ground_velocity_w,
        )
        drive_shift = drive_velocity * drive_overrun.unsqueeze(1)
        if torch.any(transition_drive_to_takeoff):
            transition_ids = env._robot._ALL_INDICES[transition_drive_to_takeoff]
            touchdown_pos = (
                env._combined_landing_pos_w[transition_ids]
                + env._combined_ground_velocity_w[transition_ids]
                * time[transition_drive_to_takeoff].unsqueeze(1)
            )
            takeoff_height = (
                env._combined_takeoff_end_pos_w[transition_ids, 2]
                - env._combined_liftoff_pos_w[transition_ids, 2]
            )
            drive_velocity = env._combined_ground_velocity_w[transition_ids]
            drive_duration = env._combined_drive_duration[transition_ids]
            liftoff_pos = touchdown_pos + drive_velocity * drive_duration
            drive_curve_transition = env._combined_drive_curve_active[transition_drive_to_takeoff]
            if torch.any(drive_curve_transition):
                curve_ids = transition_ids[drive_curve_transition]
                curve_time = drive_duration[drive_curve_transition]
                _, curve_start_velocity, _, _, _, _ = self._curve_reference(
                    torch.zeros_like(curve_time), curve_ids
                )
                curve_end_pos, curve_end_velocity, _, _, _, _ = self._curve_reference(
                    curve_time, curve_ids
                )
                liftoff_pos[drive_curve_transition] = curve_end_pos
                drive_velocity[drive_curve_transition] = curve_end_velocity
                drive_start_velocity = env._combined_drive_start_velocity_w[transition_ids].clone()
                drive_start_velocity[drive_curve_transition] = curve_start_velocity
            else:
                drive_start_velocity = drive_velocity
            env._combined_spawn_pos_w[transition_ids] = touchdown_pos
            env._combined_drive_start_velocity_w[transition_ids] = drive_start_velocity
            env._combined_drive_velocity_w[transition_ids] = drive_velocity
            env._combined_liftoff_pos_w[transition_ids] = liftoff_pos
            env._combined_takeoff_start_velocity_w[transition_ids] = drive_velocity
            env._combined_takeoff_end_pos_w[transition_ids] = (
                liftoff_pos
                + 0.5
                * env._combined_takeoff_duration[transition_ids]
                * (
                    drive_velocity
                    + env._combined_takeoff_end_velocity_w[transition_ids]
                )
            )
            env._combined_takeoff_end_pos_w[transition_ids, 2] = (
                liftoff_pos[:, 2] + takeoff_height
            )
            env._desired_pos_w[transition_ids] = (
                env._combined_takeoff_end_pos_w[transition_ids]
                + env._combined_takeoff_end_velocity_w[transition_ids]
                * env._combined_flight_duration[transition_ids]
            )
            env._combined_route[transition_ids] = self.TAKEOFF_ROUTE
            env._combined_transition_takeoff_reached[transition_ids] = True
        normal_drive_to_takeoff = drive_to_takeoff & (~transition_drive_to_takeoff)
        env._combined_liftoff_pos_w[normal_drive_to_takeoff] += drive_shift[normal_drive_to_takeoff]
        env._combined_takeoff_end_pos_w[normal_drive_to_takeoff] += drive_shift[normal_drive_to_takeoff]
        env._desired_pos_w[normal_drive_to_takeoff] += drive_shift[normal_drive_to_takeoff]
        # Translate the whole takeoff segment by the ground roll of the prep
        # window. Shifting start and end together preserves its geometry, so the
        # same climb is simply performed from further along the drive.
        prep_duration = float(self.cfg.takeoff_prep_duration_s)
        if prep_duration > 0.0 and torch.any(drive_to_takeoff):
            prep_shift = env._combined_takeoff_start_velocity_w * prep_duration
            env._combined_liftoff_pos_w[drive_to_takeoff] += prep_shift[drive_to_takeoff]
            env._combined_takeoff_end_pos_w[drive_to_takeoff] += prep_shift[drive_to_takeoff]
            env._desired_pos_w[drive_to_takeoff] += prep_shift[drive_to_takeoff]
        flight_shift = env._combined_landing_start_velocity_w * flight_overrun.unsqueeze(1)
        env._combined_landing_start_pos_w[flight_to_landing] += flight_shift[flight_to_landing]
        env._combined_landing_pos_w[flight_to_landing] += flight_shift[flight_to_landing]
        env._desired_pos_w[flight_to_landing] += flight_shift[flight_to_landing]
        transition_flight_to_landing = flight_to_landing & env._combined_landing_to_takeoff
        env._combined_flight_duration[transition_flight_to_landing] = 2.0
        env._combined_landing_transition_time[landing_to_drive] = time[landing_to_drive]
        # Re-anchor the DRIVE reference to where the LANDING reference actually
        # IS at this instant, for every landing.
        #
        # The drive segment is built as landing_pos + ground_velocity * t, and
        # landing_pos was the PLANNED touchdown point. So the moment the modes
        # swapped, the reference snapped from wherever the descent had reached
        # to wherever the plan said it should have ended -- the teleport into
        # the driving segment. Re-anchoring makes position continuous across the
        # handover by construction rather than by luck.
        #
        # Computed BEFORE the mode assignments below, because _reference_state
        # reads _combined_mode to choose which segment to evaluate.
        if torch.any(landing_to_drive):
            landing_to_drive_ids = env._robot._ALL_INDICES[landing_to_drive]
            env._combined_landing_pos_w[landing_to_drive_ids] = self._reference_state(
                env_ids=landing_to_drive_ids
            )[0]
        env._combined_takeoff_start_morph[drive_to_takeoff] = torch.mean(
            torch.clamp(env.vehicle.morph_joint_positions()[drive_to_takeoff], 0.0, pi / 2), dim=1
        )
        env._combined_mode[drive_to_takeoff] = self.TAKEOFF
        env._combined_mode[takeoff_to_flight] = self.FLIGHT
        env._combined_mode[flight_to_landing] = self.LANDING
        env._combined_mode[landing_to_drive] = self.DRIVE
        env._combined_yaw[drive_to_takeoff] = env._combined_flight_yaw[drive_to_takeoff]
        env._combined_yaw[landing_to_drive] = env._combined_drive_yaw[landing_to_drive]
        env._combined_phase_start_elapsed[transitioned, 0] = env._time_elapsed[transitioned]
        env._combined_phase_initial_time[transitioned, 0] = 0.0
        # Negative phase time is the prep window: the takeoff clock reaches zero
        # when the climb starts, so every downstream time comparison (segment
        # progress, takeoff_to_flight, liftoff timing) accounts for it already.
        env._combined_phase_initial_time[drive_to_takeoff, 0] = -prep_duration
        env._combined_phase_initial_time[takeoff_to_flight, 0] = torch.clamp(
            time[takeoff_to_flight] - env._combined_takeoff_duration[takeoff_to_flight, 0], min=0.0
        )
        env._combined_phase_initial_time[landing_to_drive, 0] = landing_overrun[landing_to_drive]
        transition_landing_to_drive = landing_to_drive & env._combined_landing_to_takeoff
        if torch.any(transition_landing_to_drive):
            transition_ids = env._robot._ALL_INDICES[transition_landing_to_drive]
            env._combined_airborne[transition_landing_to_drive] = False
            # The overrun shift that used to live here is gone: _reference_state
            # already carries landing_pos + ground_velocity * landing_overrun,
            # so applying it again on top of the re-anchor above would advance
            # this subset's reference twice.
            env._combined_curve_origin_w[transition_ids] = env._combined_landing_pos_w[transition_ids]
            env._combined_phase_initial_time[transition_landing_to_drive, 0] = 0.0

    def task_observation(self) -> torch.Tensor:
        """Mode one-hot plus signed seconds to the current phase's event.

        The scalar is negative before the event and positive after, and it is
        exactly the quantity the timing rewards are built from: for takeoff the
        phase clock is measured from the climb start, so it counts down through
        the prep window and then becomes the lateness the timing multiplier
        penalises; for landing it is the offset from the expected touchdown,
        whose positive part is the lateness in ``touchdown_timing``. The critic
        can therefore reconstruct both multipliers rather than infer them.

        Drive and flight read zero. Their scheduled event is the transition
        itself, and announcing that would reveal the route before the mode
        one-hot does, which is the leak this task deliberately avoids.
        """
        self._update_mode()
        env = self.env
        mode = env._combined_mode
        one_hot = torch.nn.functional.one_hot(mode, num_classes=4).float()
        phase_time = self._phase_time()[:, 0]
        time_to_event = torch.where(
            mode == self.TAKEOFF, phase_time, torch.zeros_like(phase_time)
        )
        time_to_event = torch.where(
            mode == self.LANDING,
            phase_time - env._combined_landing_duration[:, 0],
            time_to_event,
        )
        clip = float(self.cfg.phase_event_time_clip_s)
        time_to_event = torch.nan_to_num(time_to_event, nan=0.0).clamp(-clip, clip)

        # The driving-frame position error lives in reference_pos_error now that
        # every vector observation is heading frame, so it is not repeated here.
        # This channel carries only commanded quantities, which is also why it
        # is correct for it to bypass observation noise: the mode and the phase
        # schedule are known exactly on hardware, unlike estimator-derived state.
        return torch.cat((one_hot, time_to_event.unsqueeze(1)), dim=1)

    def _landing_posture_ok(self) -> torch.Tensor:
        """Require every hip near drive posture before a landing can complete.

        Stage 1 ramps the threshold from zero to the configured value. Stages 2
        and 3 keep the full threshold.
        """
        env = self.env
        threshold = self._landing_min_tilt_threshold()
        if threshold <= 0.0:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        tilt = env.vehicle.joint_group_positions("morph_tilt")
        return torch.nan_to_num(tilt, nan=0.0).min(dim=1).values >= threshold

    def _landing_tracking_fade(self, phase_time: torch.Tensor) -> torch.Tensor:
        """Ramp reference tracking down to zero over the final approach.

        One outside LANDING and early in it, falling linearly to zero across
        the last ``landing_tracking_fade_s`` of the landing reference. The same
        factor suppresses the trajectory-deviation termination, so a deliberate
        hop cannot end the episode before the reward can teach it.

        ``phase_time`` must be the per-env vector, i.e. ``_phase_time()[:, 0]``.
        ``_phase_time`` returns (num_envs, 1), and subtracting that from the
        (num_envs,) duration broadcasts into an (num_envs, num_envs) outer
        product instead of raising.
        """
        env = self.env
        if phase_time.dim() != 1:
            raise ValueError(
                f"phase_time must be 1-D per-env, got shape {tuple(phase_time.shape)}"
            )
        fade_s = float(self.cfg.landing_tracking_fade_s)
        if fade_s <= 0.0:
            return torch.ones_like(phase_time)
        remaining = env._combined_landing_duration[:, 0] - phase_time
        fade = torch.clamp(remaining / fade_s, 0.0, 1.0)
        return torch.where(
            env._combined_mode == self.LANDING, fade, torch.ones_like(fade)
        )

    def _trajectory_deviation_threshold(self) -> float:
        """Deviation limit: one absolute-epoch ramp, active from epoch zero.

        Opens at trajectory_deviation_wall_start (8 m) and tightens linearly to
        trajectory_deviation_wall_final (2.5 m) by
        trajectory_deviation_wall_ramp_epochs (400), independent of curriculum
        stage or the stage-3 hardening overlay. The wide start replaces the old
        disable-until-epoch-250 grace period: a from-scratch policy fits inside
        8 m while it learns to track, and the drawn actuator lag is now part of
        the observation, so a sustained excursion is a policy failure rather
        than the honest consequence of an unobservable draw. The dwell
        requirement below still forgives transient spikes.
        """
        env = self.env
        start = float(self.cfg.trajectory_deviation_wall_start)
        final = float(self.cfg.trajectory_deviation_wall_final)
        ramp = env._epoch_ramp(0.0, float(self.cfg.trajectory_deviation_wall_ramp_epochs))
        return start + (final - start) * ramp

    def _landing_min_tilt_threshold(self) -> float:
        env = self.env
        target = float(self.cfg.landing_min_tilt_rad)
        if target <= 0.0:
            return 0.0
        stage = int(env.cfg.curriculum_stage)
        if stage == 1:
            return target * env._epoch_ramp(
                float(self.cfg.landing_min_tilt_ramp_start_epoch),
                float(self.cfg.landing_min_tilt_ramp_start_epoch)
                + float(self.cfg.landing_min_tilt_ramp_epochs),
            )
        return target

    def rotor_thrust_gate(self) -> torch.Tensor | None:
        """Per-phase rotor authority. Zero in DRIVE; tilt limited elsewhere.

        The old DRIVE-only test left the rotors live wherever the mode was not
        DRIVE, including parked on the wheels in LANDING and through the TAKEOFF
        prep window. At high tilt the rotor axes point sideways: they cannot
        lift, but they yaw and shove happily, and the 2026-08-16 policy learned
        to steer on the ground with them. Gated rather than penalised, because
        it is not a manoeuvre the vehicle can actually perform.

        TAKEOFF rotor authority rises linearly from zero at 65 deg to full at
        45 deg. LANDING and FLIGHT are deliberately UNLIMITED: landing keeps
        thrust through the flare, and the airborne posture recovery resets spawn
        between 40 deg and pi/2 specifically to train recovery from a stalled
        morph.
        """
        if not bool(getattr(self.env.cfg, "drive_zero_thrust", False)):
            return None
        env = self.env
        mode = env._combined_mode
        live = (mode != self.DRIVE).to(dtype=torch.float32)
        morph = env.vehicle.morph_joint_positions()
        if morph.numel():
            # Worst (least tucked) joint, matching every other tilt criterion.
            tilt = torch.min(torch.clamp(morph, 0.0, pi / 2), dim=1).values
            full_tilt = float(self.cfg.takeoff_thrust_full_tilt_rad)
            zero_tilt = float(self.cfg.takeoff_thrust_zero_tilt_rad)
            takeoff_gate = torch.clamp(
                (zero_tilt - tilt) / max(zero_tilt - full_tilt, 1e-6),
                0.0,
                1.0,
            )
            live = live * torch.where(mode == self.TAKEOFF, takeoff_gate, torch.ones_like(tilt))
        return live.unsqueeze(1)

    def wheel_speed_gate(self) -> torch.Tensor | None:
        """Deployment parity: the wheel speed target is zeroed while airborne.

        The takeoff prep window counts as grounded. The mode has already flipped
        to TAKEOFF there, but the reference is still rolling along the ground,
        so cutting the wheels would make that reference impossible to follow and
        the announced window useless.
        """
        if not bool(getattr(self.env.cfg, "wheels_ground_only", False)):
            return None
        mode = self.env._combined_mode
        grounded = (
            (mode == self.DRIVE)
            | (mode == self.LANDING)
            | ((mode == self.TAKEOFF) & (self._phase_time()[:, 0] < 0.0))
        )
        return grounded.to(dtype=torch.float32).unsqueeze(1)

    def observation_value(self, source: str) -> torch.Tensor | None:
        ctx = self._observation_context()
        values = {
            "task_ref_accel_w": ctx["ref_accel_w"],
            "task_ref_pos_error_w": ctx["ref_pos_error_w"],
            "task_vel_error_w": ctx["vel_error_w"],
            "task_ref_yaw_accel": ctx["yaw_accel"],
            "task_yaw_error": ctx["yaw_error"],
            "task_ref_yaw_rate_error": ctx["yaw_rate_error"],
        }
        return values.get(source)

    def _observation_context(self):
        env = self.env
        if self._observation_context_cache is not None and self._observation_context_cache["step"] == env._global_env_step:
            return self._observation_context_cache
        ref_pos, ref_vel, ref_accel = self._reference_state()
        ref_yaw, ref_yaw_rate, ref_yaw_accel = self._reference_yaw_state()
        yaw_error = self._wrap_to_pi(ref_yaw - self._yaw_from_quat(env._robot.data.root_link_quat_w)).unsqueeze(1)
        # Heading frame, matching the root observations: yaw rotated, z world.
        heading_yaw = heading_yaw_from_quat(
            env._robot.data.root_link_quat_w, env.vehicle.spec.forward_yaw_offset
        )
        self._observation_context_cache = {
            "step": env._global_env_step,
            "ref_accel_w": to_heading_frame(ref_accel, heading_yaw),
            "ref_pos_error_w": to_heading_frame(
                ref_pos - env._robot.data.root_link_pos_w, heading_yaw
            ),
            "vel_error_w": to_heading_frame(
                ref_vel - env._robot.data.root_com_lin_vel_w, heading_yaw
            ),
            "yaw_accel": ref_yaw_accel.unsqueeze(1),
            "yaw_error": yaw_error,
            "yaw_rate_error": ref_yaw_rate.unsqueeze(1) - env._robot.data.root_com_ang_vel_w[:, 2:3],
        }
        return self._observation_context_cache

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        self._update_mode()
        termination = self._termination_masks()
        died = termination["died"]
        time_out = termination["time_out"]
        ref_pos, ref_vel, _ = self._reference_state()
        ref_yaw, ref_yaw_rate, _ = self._reference_yaw_state()
        phase_time = self._phase_time()[:, 0]
        discovery_reward_active = float(
            env._training_epoch() < float(self.cfg.discovery_reward_end_epoch)
        )
        root_pos = torch.nan_to_num(env._robot.data.root_link_pos_w, nan=0.0, posinf=1e6, neginf=-1e6)
        root_vel = torch.nan_to_num(env._robot.data.root_com_lin_vel_w, nan=0.0, posinf=1e3, neginf=-1e3)
        yaw_error = self._wrap_to_pi(self._yaw_from_quat(env._robot.data.root_link_quat_w) - ref_yaw)
        position_delta = ref_pos - root_pos
        position_distance = torch.linalg.norm(position_delta, dim=1)
        drive_reference_active = (env._combined_mode == self.DRIVE) | (
            (env._combined_mode == self.TAKEOFF) & (phase_time < 0.0)
        )
        trajectory_position_delta = position_delta.clone()
        trajectory_position_delta[drive_reference_active, 2] = 0.0
        trajectory_position_distance = torch.linalg.norm(
            trajectory_position_delta, dim=1
        )
        # Translation and yaw are separate objectives. Combining metres and
        # radians hid which one the policy corrected and counted yaw twice.
        pos_error = trajectory_position_distance
        # Termination remains based on Cartesian distance. Folding yaw in would
        # let a radian of heading error end an episode that is on position,
        # which at a 1 m threshold is a much tighter coupling than intended.
        # Unlike the reward, termination retains the Z error so the height
        # randomization cannot hide a gross vertical excursion.
        env._combined_trajectory_pos_error[:] = position_distance.detach()
        reference_velocity_delta = ref_vel - root_vel
        reference_velocity_delta[drive_reference_active, 2] = 0.0
        vel_error = torch.linalg.norm(reference_velocity_delta, dim=1)
        near_position = torch.clamp(1.0 - pos_error / 0.30, 0.0, 1.0)
        near_position_score = 100.0 * torch.square(near_position) * (3.0 - 2.0 * near_position)
        pos_score = (
            50.0 / (1.0 + 2.0 * pos_error)
            + 25.0 / (1.0 + 10.0 * pos_error)
            + near_position_score
        )
        vel_score = 25.0 / (1.0 + 2.0 * vel_error)
        yaw_abs_error = torch.abs(yaw_error)
        yaw_rate_error = env._robot.data.root_com_ang_vel_w[:, 2] - ref_yaw_rate
        drive_yaw_score = 25.0 / (1.0 + 2.0 * yaw_abs_error)
        drive_yaw_rate_score = 10.0 / (1.0 + 2.0 * torch.abs(yaw_rate_error))
        contact = env.scene["contact_sensor"].data.current_contact_time
        valid_contact = torch.any(contact[:, env._valid_contact_ids] > 0.0, dim=1)
        invalid_contact = torch.any(contact[:, env._invalid_contact_ids] > 0.0, dim=1)
        # Fold in the early-touchdown failure the termination path adds, so the
        # invalid-contact PENALTY fires for it too and not just the death. This
        # local recompute reads raw contact bodies and cannot see it otherwise,
        # which would leave an early landing terminating for free.
        invalid_contact = invalid_contact | termination["invalid_contacts"]
        contact_state = self._contact_state()
        valid_contact_count = contact_state["valid_contact_count"]
        enough_contacts = valid_contact_count >= len(env._valid_contact_ids)
        landing_route = (env._combined_route == self.LANDING_ROUTE) & (~env._combined_pure_drive)
        airborne = ~valid_contact
        root_quat = torch.nan_to_num(
            env._robot.data.root_link_quat_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        body_up_z = 1.0 - 2.0 * (
            torch.square(root_quat[:, 1]) + torch.square(root_quat[:, 2])
        )
        attitude_tilt = torch.acos(body_up_z.clamp(-1.0, 1.0))
        attitude_tilt_penalty = (
            torch.square(attitude_tilt)
            * float(self.cfg.attitude_tilt_penalty_scale)
            * airborne.float()
            * env.step_dt
        )
        flight_modes = (env._combined_mode == self.FLIGHT) | (env._combined_mode == self.LANDING)
        morph = env.vehicle.morph_joint_positions()
        # Worst hip, not the mean. Under the mean a [0, 0, 90, 90] split scored
        # exactly as well as a uniform 45 deg tuck, so nothing in the reward
        # distinguished a symmetric tuck from a lopsided one and the frontier
        # ratcheted on an average the landing gate could never satisfy. Every
        # other tilt criterion (_landing_posture_ok, _combined_first_contact_tilt)
        # already uses the minimum.
        tuck_completion = torch.min(
            torch.clamp(morph / (pi / 2), 0.0, 1.0), dim=1
        ).values
        ground_leg_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        leg_group = env.vehicle.spec.leg_joint_group
        if leg_group is not None:
            leg_positions = torch.nan_to_num(
                env.vehicle.joint_group_positions(leg_group),
                nan=pi / 2,
                posinf=pi / 2,
                neginf=0.0,
            )
            ground_leg_valid = torch.all(leg_positions <= pi / 4, dim=1)
        ground_leg_reward_valid = (
            ~(
                ((env._combined_mode == self.LANDING) | (env._combined_mode == self.DRIVE))
                & (~ground_leg_valid)
            )
        ).float()
        clearance = root_pos[:, 2] - env._combined_liftoff_pos_w[:, 2]
        # Gated out of DRIVE mode. env._combined_airborne latches this with |=
        # and never clears within an episode, while takeoff_event requires
        # ~env._combined_airborne -- so a 10 cm hop over a bump during the drive
        # phase, with the wheels momentarily clear, silently made the takeoff
        # bonus unreachable for the rest of that episode. Nothing legitimately
        # counts as airborne while the reference is still rolling on the ground.
        physically_airborne = (
            airborne & (clearance > 0.10) & (env._combined_mode != self.DRIVE)
        )
        # phase_time is negative during the takeoff prep window. Gating on it
        # keeps a liftoff that happens while the reference is still on the
        # ground from claiming the bonus at a full timing multiplier.
        takeoff_climb_started = phase_time >= 0.0
        takeoff_event = (
            (env._combined_route == self.TAKEOFF_ROUTE)
            & (env._combined_mode != self.DRIVE)
            & takeoff_climb_started
            & physically_airborne
            & (~env._combined_airborne)
            & (~died)
        )
        takeoff_airborne_time = torch.clamp(phase_time, min=0.0)
        takeoff_timing_multiplier = self._takeoff_timing_multiplier(takeoff_airborne_time)
        takeoff_height = (
            env._combined_takeoff_end_pos_w[:, 2] - env._combined_liftoff_pos_w[:, 2]
        ).clamp_min(0.10)
        takeoff_lift_fraction = torch.clamp(clearance / takeoff_height, 0.0, 1.0)
        takeoff_lift_rew = (
            takeoff_lift_fraction
            * (env._combined_mode == self.TAKEOFF).float()
            * takeoff_climb_started.float()
            * (env._combined_route == self.TAKEOFF_ROUTE).float()
            * self.stage_value(self.cfg.takeoff_lift_rew_scale, "takeoff_lift_rew_scale")
            * discovery_reward_active
            * env.step_dt
        )
        env._combined_takeoff_airborne_time[takeoff_event] = takeoff_airborne_time[takeoff_event]
        env._combined_takeoff_timing_multiplier[takeoff_event] = takeoff_timing_multiplier[takeoff_event]
        env._combined_airborne |= physically_airborne
        env._combined_ep_took_off |= takeoff_event
        landing_done = landing_route & (env._combined_mode == self.DRIVE)
        # A COMPLETED landing: four wheels, valid posture, in time. This is the
        # set the DIAGNOSTICS are masked to, and it deliberately ignores whether
        # the arrival was early -- otherwise landing_early_fraction would only
        # ever see on-time landings, read ~0, and stop being able to report the
        # very failure it exists to measure.
        landing_completed = (
            landing_route
            & landing_done
            & enough_contacts
            & ground_leg_valid
            & (~invalid_contact)
            & (~env._combined_ep_contact_in_acceptance)
            & ((env.cfg.episode_length_s - env._time_elapsed) > self.stage_value(self.cfg.contact_reward_min_time_remaining_s, "contact_reward_min_time_remaining_s"))
        )
        # Was the touchdown early? Read from the LATCHED first-contact time, not
        # from a live condition: by the time the landing completes the mode has
        # already flipped to DRIVE, so the per-step early_touchdown flag has
        # stopped firing. first_contact_time is stamped at the first wheel down
        # and survives to here.
        touchdown_was_early = env._combined_first_contact_recorded & (
            env._combined_first_contact_time
            < env._combined_landing_duration[:, 0]
            - float(self.cfg.landing_early_tolerance_s)
        )
        # An early landing keeps the quality scale but loses the baseline. No
        # terminal penalty is added, so the timing curve supplies the gradient
        # instead of turning the whole landing reward into a binary gate.
        env._combined_ep_contact_in_acceptance |= landing_completed
        has_landed = env._combined_ep_contact_in_acceptance | (landing_route & landing_done & enough_contacts)
        vertical_speed = torch.abs(root_vel[:, 2])
        contact_speed_free = float(
            self.stage_value(self.cfg.contact_speed_rew_free_speed, "contact_speed_rew_free_speed")
        )
        contact_speed_zero = max(
            float(self.stage_value(self.cfg.contact_speed_rew_zero_speed, "contact_speed_rew_zero_speed")),
            contact_speed_free + 1e-6,
        )
        morph_action = torch.zeros(env.num_envs, device=env.device)
        if "tilt_mean" in env.vehicle.action_schema.slices:
            morph_action = torch.nan_to_num(
                env.vehicle.action_term_values("tilt_mean", filtered=True).mean(dim=1),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).clamp(-1.0, 1.0)
        mode = env._combined_mode
        mode_changed = mode != env._combined_tuck_frontier_mode
        env._combined_tuck_frontier[mode_changed] = tuck_completion[mode_changed]
        env._combined_tuck_frontier_mode[mode_changed] = mode[mode_changed]
        landing_frontier = (
            (mode == self.LANDING)
            & (tuck_completion >= env._combined_tuck_frontier - 1e-4)
        )
        takeoff_frontier = (
            (mode == self.TAKEOFF)
            & (tuck_completion <= env._combined_tuck_frontier + 1e-4)
        )
        env._combined_tuck_frontier[mode == self.LANDING] = torch.maximum(
            env._combined_tuck_frontier[mode == self.LANDING],
            tuck_completion[mode == self.LANDING],
        )
        env._combined_tuck_frontier[mode == self.TAKEOFF] = torch.minimum(
            env._combined_tuck_frontier[mode == self.TAKEOFF],
            tuck_completion[mode == self.TAKEOFF],
        )
        # Weight the tuck by descent progress: the same tuck command is worth
        # half as much at the landing start as it is at the ground. This makes
        # lower tucks better without removing the initiation gradient entirely.
        # Both shaping terms are stage-zeroed: live for discovery at stages
        # 1-2, zero at stage 3 where the outcome rewards carry the incentive.
        landing_height_progress = torch.clamp(
            (
                env._combined_landing_start_pos_w[:, 2]
                - root_pos[:, 2]
            )
            / (
                env._combined_landing_start_pos_w[:, 2]
                - env._combined_landing_pos_w[:, 2]
            ).clamp_min(1e-6),
            0.0,
            1.0,
        )
        tuck_height_weight = 0.5 + 0.5 * landing_height_progress
        tuck_progress_rew = (
            torch.clamp(morph_action, min=0.0)
            * tuck_height_weight
            * landing_frontier.float()
            * (~valid_contact).float()
            * (~has_landed).float()
            * discovery_reward_active
        )
        # Dense per-step cost for sitting on the ground before the reference
        # arrives. Everything else prices earliness ONCE, at touchdown, as one
        # of three averaged factors behind a tilt multiplier -- so a policy that
        # drops early and waits pays a single bounded fee and then rides out the
        # rest of the descent for free. This charges it for every step it is
        # down early, which is the quantity actually being minimised.
        #
        # Scaled by HOW early, not just that it is early, so there is a gradient
        # pointing at the schedule from anywhere in the descent rather than a
        # step at the tolerance boundary. Uses the same
        # landing_early_tolerance_s the mode gate uses, so the reward and the
        # transition agree on what "early" means.
        landing_clock_deficit = torch.clamp(
            env._combined_landing_duration[:, 0]
            - float(self.cfg.landing_early_tolerance_s)
            - phase_time,
            min=0.0,
        ) / env._combined_landing_duration[:, 0].clamp_min(1e-6)
        early_ground_penalty = (
            (env._combined_mode == self.LANDING).float()
            * landing_route.float()
            * valid_contact.float()
            * landing_clock_deficit
            * self.stage_value(self.cfg.early_ground_pen_scale, "early_ground_pen_scale")
            * env.step_dt
        )
        # Takeoff guide, part 1: measured untuck progress, not the untuck
        # action. State-based pay cannot be farmed by oscillating the command,
        # and the frontier gate still stops paying the moment the vehicle
        # re-tucks past its own best.
        untuck_progress_rew = (
            (1.0 - tuck_completion)
            * takeoff_frontier.float()
            * self.stage_value(self.cfg.untuck_progress_rew_scale, "untuck_progress_rew_scale")
            * discovery_reward_active
            * env.step_dt
        )
        # How early the wheels actually arrived, as a fraction of the landing
        # reference. 0.0 is on schedule or later, 1.0 is touchdown at the very
        # start of the descent.
        #
        # This exists because the claim it measures had never been measured.
        # "The vehicle reaches the ground around 45% early" travelled for two
        # sessions as a code comment and drove two failed attempts at weighting
        # the tuck reward, with no logged quantity behind it.
        #
        # It is also the only way to read this run. Both 2026-08-16 reward
        # changes -- the timing floor 0.2 -> 0.05 and the early sigma 0.20 ->
        # 1.5 m -- RAISE contact_in_acceptance_rew for an early landing, because
        # a term that was saturated at exactly zero now returns something. That
        # curve will step up whether or not the landing got better, so it cannot
        # be used to judge the fix. This can.
        landing_early_fraction = torch.clamp(
            1.0
            - env._combined_first_contact_time
            / env._combined_landing_duration[:, 0].clamp_min(1e-6),
            0.0,
            1.0,
        )
        landing_event_time = torch.where(
            env._combined_first_contact_recorded,
            env._combined_first_contact_time,
            phase_time,
        )
        early_contact_excess = torch.clamp(
            env._combined_final_contact_reference_distance - 0.20,
            min=0.0,
        )
        late_contact_time = torch.clamp(
            landing_event_time - env._combined_landing_duration[:, 0],
            min=0.0,
        )
        overlay_active = env.final_stage_overlay_active()
        # The early curve is exponential rather than Gaussian so it remains
        # informative in the tail where the policy actually lands. The 0.20 m
        # free band still absorbs ordinary touchdown scatter, while the scale
        # gives a strong slope immediately outside it.
        early_sigma = float(self.cfg.contact_timing_early_sigma_m)
        touchdown_timing = torch.exp(
            -early_contact_excess / early_sigma
            - torch.square(late_contact_time / (0.50 if overlay_active else 0.25))
        )
        final_contact_velocity = env._combined_final_contact_velocity_w
        # Deviation from the target descent rate, not from zero: world +z is up,
        # so a touchdown at the target has velocity -target and scores best,
        # while both a hover and a hard arrival are penalised.
        final_contact_speed = torch.abs(
            final_contact_velocity[:, 2] + float(self.cfg.contact_speed_target_mps)
        )
        final_contact_speed_multiplier = 0.025 + 0.975 * torch.clamp(
            (contact_speed_zero - final_contact_speed)
            / (contact_speed_zero - contact_speed_free),
            0.0,
            1.0,
        )
        landing_config_score = self._joint_config_score(env.vehicle.spec.landing_joint_config)
        drive_config_score = torch.square(torch.square(landing_config_score))
        first_contact = (
            (env._combined_mode == self.LANDING)
            & torch.any(contact_state["valid_contact_mask"], dim=1)
            & (~env._combined_touchdown_config_recorded)
        )
        env._combined_touchdown_config_score = torch.where(
            first_contact,
            landing_config_score.detach() * ground_leg_valid.float(),
            env._combined_touchdown_config_score,
        )
        env._combined_touchdown_config_recorded |= (env._combined_mode == self.LANDING) & torch.any(
            contact_state["valid_contact_mask"], dim=1
        )
        touchdown_config_reward_valid = env._combined_touchdown_config_score > 0.0
        timing_baseline = float(self.cfg.contact_timing_baseline)
        contact_timing_quality = timing_baseline + (1.0 - timing_baseline) * touchdown_timing
        final_contact_xy_error = env._combined_first_contact_xy_error
        # The old window saturated at 1.60 m, so a 2 m touchdown and a 10 m one
        # scored the same and nothing rewarded landing nearer the reference.
        final_contact_position_quality = 0.025 + 0.975 * (
            1.0 - torch.clamp((final_contact_xy_error - 0.60) / 2.0, 0.0, 1.0)
        )
        tilt_floor = float(self.cfg.contact_tilt_floor_rad)
        tilt_safe = float(self.cfg.contact_tilt_safe_rad)
        tilt_target = float(self.cfg.contact_tilt_target_rad)
        safe_weight = float(self.cfg.contact_tilt_safe_weight)
        # Linear within each tier. Smoothstep's derivative vanishes at both
        # ends, which puts the weakest pull exactly at the tier boundaries and
        # at the target -- the places the tuck has to be dragged through.
        contact_tilt_progress = torch.clamp(
            (env._combined_first_contact_tilt - tilt_floor) / max(tilt_safe - tilt_floor, 1e-6),
            0.0,
            1.0,
        )
        contact_tilt_reach = torch.clamp(
            (env._combined_first_contact_tilt - tilt_safe) / max(tilt_target - tilt_safe, 1e-6),
            0.0,
            1.0,
        )
        contact_tilt_quality = (
            safe_weight * contact_tilt_progress + (1.0 - safe_weight) * contact_tilt_reach
        )
        tilt_floor_weight = float(self.cfg.contact_tilt_multiplier_floor)
        contact_tilt_multiplier = tilt_floor_weight + (1.0 - tilt_floor_weight) * contact_tilt_quality
        # Tilt gates the variable reward multiplicatively: touching down without
        # the arms tucked is the critical failure, not a quality shortfall to be
        # traded against the others. Timing, speed and position are averaged
        # instead of multiplied, so the quality scale remains informative even
        # when one factor is poor. An early landing keeps that scale but loses
        # the unconditional baseline, making timing quality the gradient rather
        # than a hard all-or-nothing reward mask.
        contact_quality_score = (
            contact_tilt_multiplier
            * (
                contact_timing_quality
                + final_contact_speed_multiplier
                + final_contact_position_quality
            )
            / 3.0
        )
        landing_baseline_reward = (
            (~touchdown_was_early).float()
            * self.stage_value(
                self.cfg.contact_in_acceptance_baseline_rew,
                "contact_in_acceptance_baseline_rew",
            )
        )
        contact_quality = landing_completed.float() * touchdown_config_reward_valid.float() * (
            landing_baseline_reward
            + self.stage_value(self.cfg.contact_in_acceptance_rew_scale, "contact_in_acceptance_rew_scale")
            * contact_quality_score
        )
        # Invalid contact is now unconditionally terminal, so the one-shot
        # penalty applies to every occurrence; the dt-scaled terms only ever
        # accrue for the single step on which the episode dies.
        invalid_penalty = (
            invalid_contact.float()
            * self.stage_value(self.cfg.post_landing_invalid_contact_pen, "post_landing_invalid_contact_pen")
            + invalid_contact.float()
            * self.stage_value(self.cfg.invalid_contact_pen, "invalid_contact_pen")
            * env.step_dt
        )
        invalid_contact_rate_penalty = (
            invalid_contact.float()
            * self.stage_value(self.cfg.invalid_contact_rate_pen, "invalid_contact_rate_pen")
            * env.step_dt
        )
        filtered_rotor = torch.nan_to_num(env.vehicle.rotor_action_values(filtered=True), nan=0.0, posinf=1.0, neginf=0.0)
        rotor_thrust = env.kT * filtered_rotor
        total_rotor_thrust = torch.sum(rotor_thrust, dim=1).clamp_min(1e-6)
        # Takeoff guide, part 2: the world-up component of delivered thrust as
        # a fraction of nominal max. Each rotor's thrust axis tilts with its
        # hip, so the vertical share is thrust * cos(tilt) projected through
        # the chassis attitude -- near zero while tucked, which couples this
        # term to the untuck guide and teaches the sequence untuck-then-push.
        # Paid only until liftoff; airborne, takeoff_lift_rew and trajectory
        # tracking take over so full throttle is not subsidised in the air.
        guide_quat = torch.nan_to_num(
            env._robot.data.root_link_quat_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        guide_body_up_z = 1.0 - 2.0 * (
            torch.square(guide_quat[:, 1]) + torch.square(guide_quat[:, 2])
        )
        vertical_thrust_fraction = torch.clamp(
            torch.sum(rotor_thrust * torch.cos(morph.clamp(0.0, pi / 2)), dim=1)
            * guide_body_up_z.clamp(0.0, 1.0)
            / max(float(env.vehicle.nominal_total_kT()), 1e-6),
            0.0,
            1.0,
        )
        vertical_thrust_rew = (
            vertical_thrust_fraction
            * (env._combined_mode == self.TAKEOFF).float()
            * (~physically_airborne).float()
            * self.stage_value(self.cfg.vertical_thrust_rew_scale, "vertical_thrust_rew_scale")
            * discovery_reward_active
            * env.step_dt
        )
        positive_spin_thrust = torch.sum(rotor_thrust[:, env.vehicle.spin_direction > 0.0], dim=1)
        negative_spin_thrust = torch.sum(rotor_thrust[:, env.vehicle.spin_direction < 0.0], dim=1)
        # Linear, not squared. Squaring flattens the penalty exactly where the
        # drift starts: a 55/45 split scores 0.010 squared against 0.100 linear,
        # so there was almost no restoring gradient until the imbalance was
        # already large. Linear keeps constant pull back toward an even split.
        diagonal_imbalance = torch.abs(
            (positive_spin_thrust - negative_spin_thrust) / total_rotor_thrust
        )
        thrust_center_action = torch.zeros(env.num_envs, device=env.device)
        action_name = next((name for name in ("thrust_center_xy", "thrust_center") if name in env.vehicle.action_schema.slices), None)
        if action_name is not None:
            thrust_center_action = torch.sum(torch.square(env.vehicle.action_schema.split(env._actions)[action_name]), dim=1)
        thrust_center_penalty = thrust_center_action * self.stage_value(self.cfg.thrust_center_action_pen_scale, "thrust_center_action_pen_scale") + torch.linalg.norm(env.vehicle.thrust_center_offset_body(), dim=1) * self.stage_value(self.cfg.thrust_center_loss_offset_pen_scale, "thrust_center_loss_offset_pen_scale")
        gated_saturation = torch.zeros(env.num_envs, device=env.device)
        for term_name, start_stage, ramp_epochs in (
            ("tilt_balance", env.cfg.morph_bias_start_stage, env.cfg.morph_balance_authority_ramp_epochs),
            ("thrust_center_xy", env.cfg.thrust_center_start_stage, env.cfg.thrust_center_authority_ramp_epochs),
        ):
            term_slice = env.vehicle.action_schema.slices.get(term_name)
            if term_slice is None:
                continue
            dead_fraction = 1.0 - float(env.action_authority(start_stage, ramp_epochs))
            if dead_fraction <= 0.0:
                continue
            gated_saturation += dead_fraction * torch.sum(
                torch.square(env._policy_actions[:, term_slice]), dim=1
            )
        # The same argument, for the channels gated by MODE rather than by
        # curriculum authority. wheel_speed_gate() zeroes the wheel target
        # whenever the vehicle is not on the ground, so while airborne the
        # channel has no effect on the plant and no cost, and every value in it
        # is equally optimal -- which is exactly the condition the authority
        # loop above was written to tax.
        #
        # It was written for tilt_balance and thrust_center_xy, and ATMO_SPEC
        # has neither, so for ATMO the whole penalty was identically zero and
        # wheel_speed was free to park at the rails. It does: the exported
        # stage-1 actor holds drive near -0.7 with a stationary reference, then
        # carries that habit into DRIVE and drives itself off the pad before the
        # climb starts.
        wheel_slice = env.vehicle.action_schema.slices.get("wheel_speed")
        wheel_gate = self.wheel_speed_gate()
        if wheel_slice is not None and wheel_gate is not None:
            airborne_fraction = 1.0 - wheel_gate.reshape(env.num_envs)
            gated_saturation = gated_saturation + airborne_fraction * torch.sum(
                torch.square(env._policy_actions[:, wheel_slice]), dim=1
            )
        action_jerk_penalty = (
            env.vehicle.action_jerk()
            * self.stage_value(self.cfg.action_jerk_pen_scale, "action_jerk_pen_scale")
            * torch.where(env._combined_mode == self.FLIGHT, 0.2, 1.0)
            * env.step_dt
        )
        hover_config_score = self._joint_config_score(env.vehicle.spec.hover_joint_config)
        too_fast_penalty = termination["too_fast_vertical"].float() * self.stage_value(
            self.cfg.too_fast_pen_scale,
            "too_fast_pen_scale",
        )
        timeout_high = time_out & (~has_landed) & landing_route
        landing_reference_expired = (env._combined_mode == self.LANDING) & (
            phase_time >= env._combined_landing_duration[:, 0]
        )
        # Fade reference tracking over the final approach. The landing reference
        # is a monotone descent, so the late committed tuck -- which needs a
        # brief upward push to buy time at high tilt -- reads as a deviation and
        # was penalised exactly when it was the right move. Touchdown position
        # and timing are still anchored by final_contact_position_quality and
        # contact_timing_quality, so nothing here removes the landing target.
        landing_tracking_fade = self._landing_tracking_fade(phase_time)
        # Reference tracking is turned DOWN through the two transition phases.
        #
        # Takeoff and landing are the phases where what matters is completing
        # the objective safely -- get off the ground, get down in one piece --
        # not flying a pretty line. The vehicle can follow the reference; it
        # needs room to be creative about how. Holding it to a straight descent
        # while it also has to tuck (and lose lift doing so) prices the
        # manoeuvre it must perform in order to land at all.
        #
        # FLIGHT and DRIVE keep full authority: that is where tracking IS the
        # objective, and where the reward reads as the pose+yaw controller it
        # was designed to be.
        transition_authority = float(self.cfg.trajectory_transition_authority)
        trajectory_phase_authority = torch.where(
            (env._combined_mode == self.TAKEOFF) | (env._combined_mode == self.LANDING),
            torch.full_like(landing_tracking_fade, transition_authority),
            torch.ones_like(landing_tracking_fade),
        )
        trajectory_reward_active = (
            (~landing_reference_expired).float()
            * (~drive_reference_active).float()
            * landing_tracking_fade
            * trajectory_phase_authority
        )
        drive_reward_active = drive_reference_active.float()
        drive_translation_rew = (
            pos_score
            * self.stage_value(self.cfg.drive_position_rew_scale, "drive_position_rew_scale")
            + vel_score
            * self.stage_value(self.cfg.drive_velocity_rew_scale, "drive_velocity_rew_scale")
        ) * drive_reward_active * env.step_dt
        drive_yaw_rew = (
            drive_yaw_score
            * self.stage_value(self.cfg.drive_yaw_rew_scale, "drive_yaw_rew_scale")
            + drive_yaw_rate_score
            * self.stage_value(self.cfg.drive_yaw_rate_rew_scale, "drive_yaw_rate_rew_scale")
        ) * drive_reward_active * env.step_dt
        position_huber = torch.where(
            pos_error <= 1.0,
            0.5 * torch.square(pos_error),
            pos_error - 0.5,
        )
        velocity_huber = torch.where(
            vel_error <= 2.0,
            0.5 * torch.square(vel_error),
            2.0 * (vel_error - 1.0),
        )
        # The same phase authority applies to the COST the progress terms are
        # built from, or the change is half-done: position_progress_rew and
        # velocity_progress_rew are cost deltas and carry most of the tracking
        # shaping, so leaving them at full strength would keep the pressure the
        # trajectory reward just gave up. progress_valid already requires the
        # regime to be unchanged step-to-step, so scaling per mode cannot
        # manufacture a spurious delta at a transition.
        position_cost = torch.where(
            drive_reference_active,
            25.0
            * position_huber
            * self.stage_value(self.cfg.drive_position_rew_scale, "drive_position_rew_scale"),
            25.0
            * position_huber
            * self.stage_value(self.cfg.trajectory_pos_rew_scale, "trajectory_pos_rew_scale")
            * trajectory_phase_authority,
        )
        velocity_cost = torch.where(
            drive_reference_active,
            10.0
            * velocity_huber
            * self.stage_value(self.cfg.drive_velocity_rew_scale, "drive_velocity_rew_scale"),
            10.0
            * velocity_huber
            * self.stage_value(self.cfg.trajectory_vel_rew_scale, "trajectory_vel_rew_scale")
            * trajectory_phase_authority,
        )
        yaw_quality = (
            drive_yaw_score
            * self.stage_value(self.cfg.drive_yaw_rew_scale, "drive_yaw_rew_scale")
            + drive_yaw_rate_score
            * self.stage_value(self.cfg.drive_yaw_rate_rew_scale, "drive_yaw_rate_rew_scale")
        )
        progress_regime = 2 * env._combined_mode + drive_reference_active.long()
        progress_active = (~landing_reference_expired) & (landing_tracking_fade >= 1.0)
        progress_valid = (
            env._combined_progress_valid
            & progress_active
            & (env._combined_progress_regime == progress_regime)
        )
        position_progress_rew = (
            env._combined_previous_position_cost - position_cost
        ) * progress_valid.float()
        velocity_progress_rew = (
            env._combined_previous_velocity_cost - velocity_cost
        ) * progress_valid.float()
        yaw_progress_rew = (
            yaw_quality - env._combined_previous_yaw_quality
        ) * progress_valid.float()
        env._combined_previous_position_cost[:] = position_cost.detach()
        env._combined_previous_velocity_cost[:] = velocity_cost.detach()
        env._combined_previous_yaw_quality[:] = yaw_quality.detach()
        env._combined_progress_regime[:] = progress_regime
        env._combined_progress_valid[:] = progress_active
        # Mean |wheel action| while the wheel channel is gated off, as a dwell
        # pair. Uses the same gate the penalty uses, so the diagnostic and the
        # incentive can never disagree about which steps count.
        wheel_slice_log = env.vehicle.action_schema.slices.get("wheel_speed")
        wheel_gate_log = self.wheel_speed_gate()
        if wheel_slice_log is not None and wheel_gate_log is not None:
            gated_dwell = (1.0 - wheel_gate_log.reshape(env.num_envs)) * env.step_dt
            wheel_cmd_gated = (
                torch.mean(
                    torch.abs(env._policy_actions[:, wheel_slice_log]), dim=1
                )
                * gated_dwell
            )
        else:
            gated_dwell = torch.zeros(env.num_envs, device=env.device)
            wheel_cmd_gated = torch.zeros(env.num_envs, device=env.device)
        drive_dwell = (env._combined_mode == self.DRIVE).float() * env.step_dt
        drive_tuck_dwell = drive_config_score * drive_dwell
        drive_error_dwell = position_distance * drive_dwell
        linear_velocity_error = torch.linalg.norm(reference_velocity_delta, dim=1)
        drive_velocity_error_dwell = linear_velocity_error * drive_dwell
        displaced = trajectory_position_distance > 0.10
        displaced_dwell = displaced.float() * env.step_dt
        closing_speed = -torch.sum(
            trajectory_position_delta * reference_velocity_delta, dim=1
        ) / trajectory_position_distance.clamp_min(1e-6)
        drive_displaced_dwell = displaced_dwell * (env._combined_mode == self.DRIVE).float()
        drive_closing_speed_dwell = closing_speed * drive_displaced_dwell
        flight_dwell = (env._combined_mode == self.FLIGHT).float() * env.step_dt
        flight_displaced_dwell = displaced_dwell * (env._combined_mode == self.FLIGHT).float()
        flight_position_error_dwell = trajectory_position_distance * flight_dwell
        flight_velocity_error_dwell = linear_velocity_error * flight_dwell
        flight_closing_speed_dwell = closing_speed * flight_displaced_dwell
        flight_active = (env._combined_mode == self.FLIGHT) & (~died)
        actual_acceleration = (
            root_vel - env._combined_previous_root_velocity
        ) / env.step_dt
        correction = trajectory_position_delta + 0.5 * reference_velocity_delta
        correction_direction = correction / torch.linalg.norm(
            correction, dim=1, keepdim=True
        ).clamp_min(1e-6)
        corrective_acceleration = torch.sum(
            actual_acceleration * correction_direction, dim=1
        )
        flight_accel_alignment_rew = (
            torch.tanh(torch.clamp(corrective_acceleration, min=0.0) / 2.0)
            * flight_active.float()
            * self.stage_value(
                self.cfg.flight_accel_alignment_rew_scale,
                "flight_accel_alignment_rew_scale",
            )
            * env.step_dt
        )
        flight_airborne_rew = (
            (flight_active & physically_airborne).float()
            * self.stage_value(
                self.cfg.flight_airborne_rew_scale,
                "flight_airborne_rew_scale",
            )
            * env.step_dt
        )
        env._combined_previous_root_velocity[:] = root_vel.detach()
        drive_turn_dwell = (
            (env._combined_mode == self.DRIVE) & (torch.abs(ref_yaw_rate) > 0.05)
        ).float() * env.step_dt
        drive_turn_yaw_rate_error_dwell = (
            torch.abs(env._robot.data.root_com_ang_vel_w[:, 2] - ref_yaw_rate)
            * drive_turn_dwell
        )
        drive_turn_correct_direction_dwell = (
            (env._robot.data.root_com_ang_vel_w[:, 2] * ref_yaw_rate > 0.0).float()
            * drive_turn_dwell
        )
        drive_joint_errors = [torch.abs(morph - pi / 2.0)]
        if leg_group is not None:
            drive_joint_errors.append(torch.abs(leg_positions))
        drive_config_deviation = torch.max(
            torch.cat(drive_joint_errors, dim=1), dim=1
        ).values / (pi / 2.0)
        # Ramp the drive config requirement in over drive_config_settle_s after
        # a landing hands over, rather than stepping to full strength the
        # instant the mode flips.
        #
        # phase_start_elapsed is reset at every transition, so this is time
        # since entering DRIVE. Applied only on the landing route: a DRIVE
        # phase reached any other way has not just finished a descent and needs
        # no grace.
        settle_s = float(self.cfg.drive_config_settle_s)
        if settle_s > 0.0:
            time_in_drive = env._time_elapsed - env._combined_phase_start_elapsed[:, 0]
            settle_gain = torch.clamp(time_in_drive / settle_s, 0.0, 1.0)
            drive_config_authority = torch.where(
                landing_route, settle_gain, torch.ones_like(settle_gain)
            )
        else:
            drive_config_authority = torch.ones(env.num_envs, device=env.device)
        drive_config_deviation_penalty = (
            drive_config_deviation
            * (env._combined_mode == self.DRIVE).float()
            * drive_config_authority
            * self.stage_value(
                self.cfg.drive_config_deviation_pen_scale,
                "drive_config_deviation_pen_scale",
            )
            * env.step_dt
        )
        flight_config_deviation_penalty = (
            (1.0 - hover_config_score)
            * (env._combined_mode == self.FLIGHT).float()
            * self.stage_value(
                self.cfg.flight_config_deviation_pen_scale,
                "flight_config_deviation_pen_scale",
            )
            * env.step_dt
        )
        flight_position_deviation_penalty = (
            torch.clamp(torch.square(pos_error), max=16.0)
            * (env._combined_mode == self.FLIGHT).float()
            * self.stage_value(
                self.cfg.flight_position_deviation_pen_scale,
                "flight_position_deviation_pen_scale",
            )
            * env.step_dt
        )
        rewards = {
            "trajectory_pos_rew": pos_score * trajectory_reward_active * self.stage_value(self.cfg.trajectory_pos_rew_scale, "trajectory_pos_rew_scale") * env.step_dt,
            "trajectory_vel_rew": vel_score * trajectory_reward_active * self.stage_value(self.cfg.trajectory_vel_rew_scale, "trajectory_vel_rew_scale") * env.step_dt,
            "drive_translation_rew": drive_translation_rew,
            "drive_yaw_rew": drive_yaw_rew,
            "position_progress_rew": position_progress_rew,
            "velocity_progress_rew": velocity_progress_rew,
            "yaw_progress_rew": yaw_progress_rew,
            "drive_config_deviation_penalty": drive_config_deviation_penalty,
            "flight_position_deviation_penalty": flight_position_deviation_penalty,
            "flight_airborne_rew": flight_airborne_rew,
            "flight_accel_alignment_rew": flight_accel_alignment_rew,
            # Unscaled tuck score and dwell time, split by branch via the done
            # mask. Divide tuck by time for the comparable [0, 1] average.
            "drive_tuck_pure_drive": drive_tuck_dwell,
            "drive_tuck_takeoff_route": drive_tuck_dwell,
            "drive_time_pure_drive": drive_dwell,
            "drive_time_takeoff_route": drive_dwell,
            "wheel_cmd_gated": wheel_cmd_gated,
            "wheel_time_gated": gated_dwell,
            "drive_err_forward": drive_error_dwell,
            "drive_err_reverse": drive_error_dwell,
            "drive_vel_err_forward": drive_velocity_error_dwell,
            "drive_vel_err_reverse": drive_velocity_error_dwell,
            "drive_closing_speed_forward": drive_closing_speed_dwell,
            "drive_closing_speed_reverse": drive_closing_speed_dwell,
            "drive_displaced_time_forward": drive_displaced_dwell,
            "drive_displaced_time_reverse": drive_displaced_dwell,
            "drive_time_forward": drive_dwell,
            "drive_time_reverse": drive_dwell,
            "flight_pos_err": flight_position_error_dwell,
            "flight_vel_err": flight_velocity_error_dwell,
            "flight_closing_speed": flight_closing_speed_dwell,
            "flight_displaced_time": flight_displaced_dwell,
            "flight_time": flight_dwell,
            "drive_turn_yaw_rate_err": drive_turn_yaw_rate_error_dwell,
            "drive_turn_correct_direction": drive_turn_correct_direction_dwell,
            "drive_turn_time": drive_turn_dwell,
            "drive_err_stationary": drive_error_dwell,
            "drive_err_stationary_positive": drive_error_dwell,
            "drive_err_stationary_negative": drive_error_dwell,
            "drive_err_straight": drive_error_dwell,
            "drive_err_curved": drive_error_dwell,
            "drive_time_stationary": drive_dwell,
            "drive_time_stationary_positive": drive_dwell,
            "drive_time_stationary_negative": drive_dwell,
            "drive_time_straight": drive_dwell,
            "drive_time_curved": drive_dwell,
            "flight_config_rew": hover_config_score
            * (env._combined_mode == self.FLIGHT).float()
            * self.stage_value(self.cfg.flight_config_rew_scale, "flight_config_rew_scale")
            * env.step_dt,
            "flight_config_deviation_penalty": flight_config_deviation_penalty,
            "airborne_diagonal_imbalance_penalty": diagonal_imbalance
            * physically_airborne.float()
            * self.stage_value(
                self.cfg.airborne_diagonal_imbalance_pen_scale,
                "airborne_diagonal_imbalance_pen_scale",
            )
            * env.step_dt,
            "successful_takeoff_rew": takeoff_event.float()
            * (
                self.stage_value(
                    self.cfg.successful_takeoff_baseline_rew,
                    "successful_takeoff_baseline_rew",
                )
                + takeoff_timing_multiplier
                * self.stage_value(self.cfg.successful_takeoff_rew, "successful_takeoff_rew")
            ),
            "takeoff_lift_rew": takeoff_lift_rew,
            "untuck_progress_rew": untuck_progress_rew,
            "vertical_thrust_rew": vertical_thrust_rew,
            "tuck_progress_rew": torch.clamp(tuck_progress_rew * self.stage_value(self.cfg.tuck_absolute_linear_rew_scale, "tuck_absolute_linear_rew_scale"), -self.stage_value(self.cfg.tuck_absolute_rew_cap, "tuck_absolute_rew_cap"), self.stage_value(self.cfg.tuck_absolute_rew_cap, "tuck_absolute_rew_cap")) * env.step_dt,

            "takeoff_timing_multiplier": takeoff_event.float() * takeoff_timing_multiplier,
            "contact_tilt_multiplier": landing_completed.float() * contact_tilt_multiplier,
            "final_contact_speed_multiplier": landing_completed.float() * final_contact_speed_multiplier,
            "final_contact_position_quality": landing_completed.float() * final_contact_position_quality,
            "contact_timing_quality": landing_completed.float() * contact_timing_quality,
            # Masked to real landings like its neighbours, so a change in the
            # logged mean is a change in EARLINESS and not in landing rate.
            "landing_early_fraction": landing_completed.float() * landing_early_fraction,
            "early_ground_penalty": early_ground_penalty,
            "early_touchdown_penalty": termination["early_touchdown"].float()
            * self.stage_value(self.cfg.early_touchdown_pen, "early_touchdown_pen"),
            "invalid_contact_penalty": invalid_penalty + invalid_contact_rate_penalty,
            "action_jerk_penalty": action_jerk_penalty,
            "yaw_angle_penalty": torch.square(yaw_abs_error)
            * (~drive_reference_active).float()
            * self.stage_value(self.cfg.yaw_angle_pen_scale, "yaw_angle_pen_scale")
            * env.step_dt,
            "yaw_rate_penalty": torch.square(yaw_rate_error)
            * (~drive_reference_active).float()
            * self.stage_value(self.cfg.yaw_rate_pen_scale, "yaw_rate_pen_scale")
            * env.step_dt,
            "thrust_center_penalty": thrust_center_penalty * env.step_dt,
            # Active in every mode. Squared beyond a deadband so per-hip bias
            # stays available as an attitude actuator while a gross split --
            # the 7 deg / 55 deg spread that left the worst hip far short of the
            # landing gate -- is what actually costs.
            "hip_spread_penalty": torch.square(
                torch.clamp(
                    morph.max(dim=1).values
                    - morph.min(dim=1).values
                    - float(self.cfg.hip_spread_deadband_rad),
                    min=0.0,
                )
            )
            * self.stage_value(self.cfg.hip_spread_pen_scale, "hip_spread_pen_scale")
            * env.step_dt,
            "contact_in_acceptance_rew": torch.clamp(contact_quality, max=self.stage_value(self.cfg.contact_in_acceptance_rew_cap, "contact_in_acceptance_rew_cap")),
            "too_fast_penalty": too_fast_penalty,
            "gated_action_saturation_penalty": gated_saturation
            * self.stage_value(
                self.cfg.gated_action_saturation_pen_scale,
                "gated_action_saturation_pen_scale",
            )
            * env.step_dt,
            "timeout_high_penalty": timeout_high.float() * self.stage_value(self.cfg.timeout_pen, "timeout_pen"),
            "attitude_tilt_penalty": attitude_tilt_penalty,
            "attitude_failure_penalty": termination["attitude_failure"].float()
            * float(self.cfg.attitude_failure_penalty),
            "trajectory_deviation_penalty": termination["trajectory_deviation_failure"].float()
            * float(self.cfg.trajectory_deviation_penalty),
        }
        rewards = {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in rewards.items()}
        self.log_rewards(rewards, died | time_out)
        landing_takeoff_reward_keys = {
            "successful_takeoff_rew",
            "takeoff_lift_rew",
            "untuck_progress_rew",
            "vertical_thrust_rew",
            "tuck_progress_rew",
            "early_ground_penalty",
            "early_touchdown_penalty",
            "contact_in_acceptance_rew",
        }
        training_rewards = {
            key: value if key in landing_takeoff_reward_keys else 0.5 * value
            for key, value in rewards.items()
        }
        return self.reward_mixer.sum(training_rewards)

    def _termination_masks(self):
        env = self.env
        if self._termination_cache_step == env._global_env_step and self._termination_cache is not None:
            return self._termination_cache
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        root_pos = env._robot.data.root_link_pos_w
        root_vel = env._robot.data.root_com_lin_vel_w
        root_quat = env._robot.data.root_link_quat_w
        root_ang = env._robot.data.root_com_ang_vel_w
        nonfinite = (
            ~torch.isfinite(root_pos).all(dim=1)
            | ~torch.isfinite(root_quat).all(dim=1)
            | ~torch.isfinite(root_vel).all(dim=1)
            | ~torch.isfinite(root_ang).all(dim=1)
        )
        env._combined_nonfinite_state[:] = nonfinite
        safe_vz = torch.nan_to_num(root_vel[:, 2], nan=0.0, posinf=1e6, neginf=-1e6)
        landing_route = (env._combined_route == self.LANDING_ROUTE) & (~env._combined_pure_drive)
        too_fast = torch.abs(safe_vz) > float(self.cfg.vertical_speed_termination_mps)
        contact_state = self._contact_state()
        invalid = contact_state["invalid_contacts"]
        # Early touchdown is a PENALTY, not a death.
        #
        # It was terminal for one run (2026-08-16_20-45-26) and that run
        # collapsed: 133 epochs, zero landings, zero takeoffs, the policy
        # specialised into a pure ground vehicle. Making the end of the landing
        # route lethal removed the payoff for ever leaving the ground, and
        # combined with a cheaper drive config penalty the optimum became
        # "never fly". The gate was too blunt an instrument for a behavior the
        # vehicle has not learned to avoid yet.
        #
        # The flag stays so the condition can be recovered, but it now drives a
        # one-shot penalty term instead of the invalid-contact path.
        phase_time_term = self._phase_time()[:, 0]
        early_touchdown = (
            landing_route
            & (env._combined_mode == self.LANDING)
            & contact_state["any_valid_contacts"]
            # Must have actually FLOWN. Mode transitions are time-based, so a
            # vehicle that never took off is sitting on the ground when LANDING
            # begins at phase_time 0 and would otherwise read as the earliest
            # possible touchdown. That is "never left the ground", a different
            # failure with its own terms, and charging it here is what made the
            # terminal version punish the wrong thing.
            & env._combined_airborne
            & (
                phase_time_term
                < env._combined_landing_duration[:, 0]
                - float(self.cfg.landing_early_tolerance_s)
            )
        )
        env._combined_early_touchdown[:] = early_touchdown
        if bool(getattr(self.cfg, "landing_early_termination", False)):
            invalid = invalid | early_touchdown
        airborne = ~contact_state["any_valid_contacts"]
        body_up_z = 1.0 - 2.0 * (torch.square(root_quat[:, 1]) + torch.square(root_quat[:, 2]))
        extreme_attitude = airborne & (body_up_z < cos(float(self.cfg.attitude_termination_angle)))
        env._combined_attitude_failure_dwell[:] = torch.where(
            extreme_attitude,
            env._combined_attitude_failure_dwell + env.step_dt,
            torch.zeros_like(env._combined_attitude_failure_dwell),
        )
        attitude_failure = env._combined_attitude_failure_dwell >= float(self.cfg.attitude_termination_dwell_s)
        env._combined_attitude_failure[:] = attitude_failure
        # _combined_trajectory_pos_error is written later in get_rewards, so
        # this reads the previous step's error. The dwell requirement spans
        # several steps, which makes the one-step lag immaterial, and the reset
        # hook zeroes the buffer so a stale error cannot terminate a fresh
        # episode on its first step.
        # Suppressed wherever reference tracking has faded out: the final-approach
        # hop is a deliberate departure from the descent profile, and terminating
        # on it would train the policy never to attempt the manoeuvre.
        gross_deviation = (
            env._combined_trajectory_pos_error > self._trajectory_deviation_threshold()
        ) & (self._landing_tracking_fade(self._phase_time()[:, 0]) > 0.0)
        env._combined_trajectory_deviation_dwell[:] = torch.where(
            gross_deviation,
            env._combined_trajectory_deviation_dwell + env.step_dt,
            torch.zeros_like(env._combined_trajectory_deviation_dwell),
        )
        trajectory_deviation_failure = env._combined_trajectory_deviation_dwell >= float(
            self.cfg.trajectory_deviation_dwell_s
        )
        env._combined_trajectory_deviation_failure[:] = trajectory_deviation_failure
        post_takeoff_condition = (
            env._combined_ep_contact_in_acceptance
            & (~env._combined_landing_to_takeoff)
            & (~contact_state["any_valid_contacts"])
            & (
                torch.nan_to_num(root_pos[:, 2], nan=0.0)
                > env._combined_landing_pos_w[:, 2]
                + float(self.cfg.post_landing_takeoff_height)
            )
        )
        env._combined_post_landing_takeoff_dwell[:] = torch.where(
            post_takeoff_condition,
            env._combined_post_landing_takeoff_dwell + env.step_dt,
            torch.zeros_like(env._combined_post_landing_takeoff_dwell),
        )
        post_takeoff = (
            env._combined_post_landing_takeoff_dwell >= float(self.cfg.post_landing_takeoff_dwell_s)
        )
        env._combined_post_landing_takeoff[:] = post_takeoff
        # Any invalid-body contact is death. The gear collider resize has been
        # applied to the training asset, so this is a real, reachable event --
        # scraping the frame or gear is never acceptable, before or after an
        # accepted touchdown.
        terminating_invalid = invalid
        died = (
            nonfinite
            | too_fast
            | attitude_failure
            | trajectory_deviation_failure
            | post_takeoff
            | terminating_invalid
        )
        if not env.cfg.terminate:
            died = torch.zeros_like(died)
        self._termination_cache = {
            "died": died,
            "time_out": time_out,
            "too_fast_vertical": too_fast,
            "attitude_failure": attitude_failure,
            "trajectory_deviation_failure": trajectory_deviation_failure,
            "nonfinite_state": nonfinite,
            "invalid_contacts": invalid,
            "terminating_invalid_contact": terminating_invalid,
            "early_touchdown": early_touchdown,
        }
        self._termination_cache_step = env._global_env_step
        return self._termination_cache

    def get_dones(self):
        termination = self._termination_masks()
        return termination["died"], termination["time_out"]

    def reset_episode_state(self, env_ids: torch.Tensor):
        self._observation_context_cache = None
        self._termination_cache_step = -1
        self._termination_cache = None
        env = self.env
        landing_drive = (
            (env._combined_route[env_ids] == self.LANDING_ROUTE)
            & (env._combined_mode[env_ids] == self.DRIVE)
            & (~env._combined_pure_drive[env_ids])
        )
        env._combined_airborne[env_ids] = (env._combined_mode[env_ids] == self.FLIGHT) | (env._combined_mode[env_ids] == self.LANDING)
        env._combined_ep_contact_in_acceptance[env_ids] = landing_drive
        env._combined_ep_took_off[env_ids] = False
        env._combined_post_landing_takeoff[env_ids] = False
        env._combined_post_landing_takeoff_dwell[env_ids] = 0.0
        env._combined_touchdown_config_score[env_ids] = landing_drive.float()
        env._combined_touchdown_config_recorded[env_ids] = landing_drive
        env._combined_nonfinite_state[env_ids] = False
        env._combined_attitude_failure_dwell[env_ids] = 0.0
        env._combined_attitude_failure[env_ids] = False
        env._combined_early_touchdown[env_ids] = False
        env._combined_trajectory_pos_error[env_ids] = 0.0
        env._combined_trajectory_deviation_dwell[env_ids] = 0.0
        env._combined_trajectory_deviation_failure[env_ids] = False
        env._combined_previous_position_cost[env_ids] = 0.0
        env._combined_previous_velocity_cost[env_ids] = 0.0
        env._combined_previous_yaw_quality[env_ids] = 0.0
        env._combined_previous_root_velocity[env_ids] = torch.nan_to_num(
            env._robot.data.root_com_lin_vel_w[env_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        env._combined_progress_valid[env_ids] = False
        env._combined_progress_regime[env_ids] = -1
        env._combined_landing_transition_time[env_ids] = 0.0
        env._combined_takeoff_airborne_time[env_ids] = -1.0
        env._combined_takeoff_timing_multiplier[env_ids] = 1.0
        morph = env.vehicle.morph_joint_positions()[env_ids]
        # Must match the aggregation used for tuck_completion in get_rewards.
        env._combined_tuck_frontier[env_ids] = torch.min(
            torch.clamp(morph / (pi / 2), 0.0, 1.0), dim=1
        ).values
        env._combined_tuck_frontier_mode[env_ids] = env._combined_mode[env_ids]
        env._combined_first_contact_time[env_ids] = 0.0
        env._combined_first_contact_recorded[env_ids] = landing_drive
        env._combined_final_contact_velocity_w[env_ids] = 0.0
        env._combined_final_contact_reference_distance[env_ids] = 0.0
        env._combined_first_contact_xy_error[env_ids] = 0.0
        env._combined_first_contact_tilt[env_ids] = 0.0
        throttle = env._combined_reset_throttle[env_ids]
        env.normalized_rotor_thrust[env_ids] = throttle
        env.previous_normalized_rotor_thrust[env_ids] = throttle
        env.normalized_rotor_thrust_filtered[env_ids] = throttle
        env.previous_normalized_rotor_thrust_filtered[env_ids] = throttle
        landing_drive_ids = env_ids[landing_drive]
        if len(landing_drive_ids) > 0:
            env._disturbance_force[landing_drive_ids] = 0.0
            env._disturbance_moment[landing_drive_ids] = 0.0
            env._disturbance_force_cts[landing_drive_ids] = 0.0
            env._disturbance_moment_cts[landing_drive_ids] = 0.0
            env._disturbance_batch_scale[landing_drive_ids] = 0.0
            env._thrust_loss_batch_scale[landing_drive_ids] = 0.0
            env._push_active[landing_drive_ids] = False
            env.kT[landing_drive_ids] = env.vehicle._nominal_kT_tensor
            env.kM[landing_drive_ids] = env.vehicle._nominal_kM_tensor
            env.normalized_rotor_thrust_filtered[landing_drive_ids] = 0.0
            env.previous_normalized_rotor_thrust_filtered[landing_drive_ids] = 0.0

    def _trajectory_segment(self, start, end, start_velocity, end_velocity, time, duration):
        """Landing task acceleration/cruise/deceleration trajectory composition."""
        duration = duration.clamp_min(self.env.step_dt)
        accel_duration = 0.2 * duration + 0.2
        decel_duration = 0.2 * duration + 0.5
        cruise_duration = torch.clamp(duration - accel_duration - decel_duration, min=self.env.step_dt)
        cruise_velocity = (
            end
            - start
            - 0.5 * accel_duration * start_velocity
            - 0.5 * decel_duration * end_velocity
        ) / (0.5 * accel_duration + cruise_duration + 0.5 * decel_duration)
        accel_end = start + 0.5 * accel_duration * (start_velocity + cruise_velocity)
        decel_start = accel_end + cruise_duration * cruise_velocity
        accel_time = torch.minimum(torch.clamp(time, min=0.0), accel_duration)
        cruise_time = torch.minimum(torch.clamp(time - accel_duration, min=0.0), cruise_duration)
        decel_time = torch.minimum(
            torch.clamp(time - accel_duration - cruise_duration, min=0.0), decel_duration
        )
        accel_pos, accel_vel, accel_accel = self._seventh_order_segment(
            start, accel_end, start_velocity, cruise_velocity, accel_time, accel_duration
        )
        cruise_pos = accel_end + cruise_velocity * cruise_time
        cruise_vel = cruise_velocity.expand_as(cruise_pos)
        cruise_accel = torch.zeros_like(cruise_pos)
        decel_pos, decel_vel, decel_accel = self._seventh_order_segment(
            decel_start, end, cruise_velocity, end_velocity, decel_time, decel_duration
        )
        return (
            torch.where(time <= accel_duration, accel_pos, torch.where(time <= accel_duration + cruise_duration, cruise_pos, decel_pos)),
            torch.where(time <= accel_duration, accel_vel, torch.where(time <= accel_duration + cruise_duration, cruise_vel, decel_vel)),
            torch.where(time <= accel_duration, accel_accel, torch.where(time <= accel_duration + cruise_duration, cruise_accel, decel_accel)),
        )

    @staticmethod
    def _seventh_order_segment(start, end, start_velocity, end_velocity, time, duration):
        duration = duration.clamp_min(1e-6)
        tau = torch.clamp(time / duration, 0.0, 1.0)
        tau2, tau3 = tau * tau, tau * tau * tau
        tau4, tau5 = tau2 * tau2, tau2 * tau3
        tau6, tau7 = tau5 * tau, tau5 * tau2
        shape = 35.0 * tau4 - 84.0 * tau5 + 70.0 * tau6 - 20.0 * tau7
        shape_rate = 140.0 * tau3 - 420.0 * tau4 + 420.0 * tau5 - 140.0 * tau6
        shape_accel = 420.0 * tau2 - 1680.0 * tau3 + 2100.0 * tau4 - 840.0 * tau5
        start_shape = tau - 20.0 * tau4 + 45.0 * tau5 - 36.0 * tau6 + 10.0 * tau7
        start_rate = 1.0 - 80.0 * tau3 + 225.0 * tau4 - 216.0 * tau5 + 70.0 * tau6
        start_accel = -240.0 * tau2 + 900.0 * tau3 - 1080.0 * tau4 + 420.0 * tau5
        end_shape = -15.0 * tau4 + 39.0 * tau5 - 34.0 * tau6 + 10.0 * tau7
        end_rate = -60.0 * tau3 + 195.0 * tau4 - 204.0 * tau5 + 70.0 * tau6
        end_accel = -180.0 * tau2 + 780.0 * tau3 - 1020.0 * tau4 + 420.0 * tau5
        delta = end - start
        return (
            start + delta * shape + start_velocity * duration * start_shape + end_velocity * duration * end_shape,
            delta * shape_rate / duration + start_velocity * start_rate + end_velocity * end_rate,
            delta * shape_accel / torch.square(duration) + start_velocity * start_accel / duration + end_velocity * end_accel / duration,
        )

    @staticmethod
    def _yaw_from_quat(quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
