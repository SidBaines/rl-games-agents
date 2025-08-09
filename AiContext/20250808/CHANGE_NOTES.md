# Changes on 2025-08-08

- Added periodic rollout video logging to Weights & Biases, configurable via `rollout_log_freq`, `rollout_max_steps`, and `rollout_fps` in training configs.
- Instrumented training with timing metrics:
  - Logs `time/train_chunk_s`, `time/train_per_step_ms` after each chunk.
  - Logs `time/eval_total_s`, `time/eval_avg_episode_ms`, `time/eval_avg_step_ms` during evaluation.
  - Logs `time/rollout_log_s` when rollout video is captured.
  - Logs `time/checkpoint_save_s` and `time/curriculum_state_save_s` for persistence.
- Added optional CPU profiling (`--profile`, `--profile-output` or YAML `training.profile`, `training.profile_output`). Profile dumps to `profiles/*.prof`.
- Updated `scripts/train.py` and `scripts/train_curriculum.py` to wire new options.
- Updated `configs/ricochet_robots/*_curriculum.yaml` with commented examples for rollout logging and profiling.
- Added `RGBArrayEncoder` for Ricochet Robots and wired it into imports and encoder factories (config `encoder.type: rgb`, optional `encoder.scale`).

Impact:
- Easier diagnosis of training slowdown causes (per-step time, eval overhead, render overhead).
- Optional cProfile allows deep analysis when needed.
- New RGB observation option enables training directly on rendered images (CHW float32 in [0,1]). 