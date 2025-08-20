# 2025-08-20 Changes

- Unify Weights & Biases logging step across training and evaluation to prevent non-monotonic step warnings.
  - Edited `rl_board_games/training/trainer.py`:
    - Added `_get_log_step(fallback_step)` to prefer SB3 model's `num_timesteps`.
    - Updated `wandb.log(...)` calls in training loop, evaluation, rollout, and timing logs to use the unified step.
    - Evaluation now logs with the SB3 step instead of the trainer's chunk boundary, eliminating step-regression warnings.

- Fix PPO CNN crash with small CHW observations when using Stable-Baselines3 default NatureCNN.
  - Added `SmallBoardCNN` features extractor at `rl_board_games/agents/sb3/feature_extractors.py` tailored for 16x16 boards.
  - Updated `scripts/train_curriculum.py`:
    - Resolve string `features_extractor_class` from YAML to actual class via `importlib`.
    - Default to `SmallBoardCNN` when using `CnnPolicy` unless overridden.
    - Ensure `normalize_images=False` for planar/rgb encoders.
  - Updated config `configs/ricochet_robots/ppo_cnn_curriculum.yaml` to specify the extractor and its `features_dim`.
