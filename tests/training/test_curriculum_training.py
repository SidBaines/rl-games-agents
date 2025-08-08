import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from rl_board_games.core.curriculum import DifficultyLookup, CurriculumLevel
from rl_board_games.curricula.ricochet_robots_curriculum import RicochetRobotsCurriculum
from rl_board_games.training.curriculum_env import CurriculumRicochetRobotsEnv
from rl_board_games.training.trainer import Trainer
from rl_board_games.games.ricochet_robots import FlatArrayEncoder
from rl_board_games.core.persistence import CheckpointManager


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self):
        self.model = Mock()
        self.model.learning_rate = 0.001
        self.model.gamma = 0.99
    
    def learn(self, total_timesteps, callback=None, reset_num_timesteps=True):
        pass
    
    def predict(self, obs, deterministic=False):
        return 0, None  # Always return action 0
    
    def save(self, path):
        pass
    
    def get_model(self):
        return self.model


class TestCurriculumTraining:
    """Test curriculum training integration."""
    
    def test_curriculum_env_creation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create simple curriculum
            levels = [
                CurriculumLevel(
                    name="Easy",
                    min_solve_length=1,
                    max_solve_length=3,
                    success_threshold=0.8,
                    board_size=4,
                    num_robots=2,
                    max_walls=5,
                    episodes_per_evaluation=10
                )
            ]
            
            lookup = DifficultyLookup(tmp_dir)
            curriculum = RicochetRobotsCurriculum(
                levels=levels,
                difficulty_lookup=lookup,
                evaluation_episodes=10
            )
            
            encoder = FlatArrayEncoder()
            env = CurriculumRicochetRobotsEnv(
                curriculum=curriculum,
                encoder=encoder,
                max_episode_steps=20
            )
            
            # Test environment creation
            assert env.curriculum == curriculum
            assert env.max_episode_steps == 20
            assert env.ricochet_game.num_robots == 2
            assert env.ricochet_game.board.width == 4
    
    def test_curriculum_env_reset_and_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create curriculum with lookup data
            lookup = DifficultyLookup(tmp_dir)
            lookup.add_difficulty(42, 4, 2, 2)  # Easy difficulty
            
            levels = [
                CurriculumLevel(
                    name="Easy",
                    min_solve_length=1,
                    max_solve_length=3,
                    success_threshold=0.8,
                    board_size=4,
                    num_robots=2,
                    max_walls=5,
                    episodes_per_evaluation=10
                )
            ]
            
            curriculum = RicochetRobotsCurriculum(
                levels=levels,
                difficulty_lookup=lookup,
                evaluation_episodes=10
            )
            
            encoder = FlatArrayEncoder()
            env = CurriculumRicochetRobotsEnv(
                curriculum=curriculum,
                encoder=encoder,
                max_episode_steps=20
            )
            
            # Test reset
            obs, info = env.reset()
            assert obs is not None
            assert "curriculum_level" in info or len(info) == 0  # May be empty initially
            
            # Test step
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
    
    def test_curriculum_trainer_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create simple curriculum
            lookup = DifficultyLookup(tmp_dir)
            lookup.add_difficulty(42, 4, 2, 2)  # Easy difficulty
            
            levels = [
                CurriculumLevel(
                    name="Easy",
                    min_solve_length=1,
                    max_solve_length=3,
                    success_threshold=0.8,
                    board_size=4,
                    num_robots=2,
                    max_walls=5,
                    episodes_per_evaluation=5
                )
            ]
            
            curriculum = RicochetRobotsCurriculum(
                levels=levels,
                difficulty_lookup=lookup,
                evaluation_episodes=5
            )
            
            encoder = FlatArrayEncoder()
            env = CurriculumRicochetRobotsEnv(
                curriculum=curriculum,
                encoder=encoder,
                max_episode_steps=10
            )
            
            # Create mock agent
            agent = MockAgent()
            
            # Create checkpoint manager
            checkpoint_manager = CheckpointManager(tmp_dir)
            
            # Create trainer
            trainer = Trainer(
                agent=agent,
                env=env,
                curriculum=curriculum,
                checkpoint_manager=checkpoint_manager,
                use_wandb=False,
                curriculum_update_freq=10
            )
            
            # Test short training run
            trainer.train(
                total_timesteps=100,
                eval_freq=50,
                save_freq=100,
                eval_episodes=3
            )
            
            # Verify curriculum was used
            assert curriculum.state.total_episodes > 0
            assert curriculum.state.current_level == 0  # Should still be at first level
    
    def test_curriculum_metrics_collection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create curriculum
            lookup = DifficultyLookup(tmp_dir)
            levels = [
                CurriculumLevel(
                    name="Easy",
                    min_solve_length=1,
                    max_solve_length=3,
                    success_threshold=0.8,
                    board_size=4,
                    num_robots=2,
                    max_walls=5,
                    episodes_per_evaluation=10
                )
            ]
            
            curriculum = RicochetRobotsCurriculum(
                levels=levels,
                difficulty_lookup=lookup,
                evaluation_episodes=10
            )
            
            encoder = FlatArrayEncoder()
            env = CurriculumRicochetRobotsEnv(
                curriculum=curriculum,
                encoder=encoder,
                max_episode_steps=20
            )
            
            # Test metrics collection
            metrics = env.get_curriculum_metrics()
            
            assert "curriculum/current_level" in metrics
            assert "curriculum/level_name" in metrics
            assert "curriculum/success_rate" in metrics
            assert "curriculum/episodes_completed" in metrics
            assert "curriculum/total_episodes" in metrics
            assert "curriculum/current_board_size" in metrics
            assert "curriculum/current_num_robots" in metrics
            
            # Test level info
            level_info = env.get_current_level_info()
            assert level_info["name"] == "Easy"
            assert level_info["board_size"] == 4
            assert level_info["num_robots"] == 2
    
    def test_curriculum_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create curriculum
            lookup = DifficultyLookup(tmp_dir)
            levels = [
                CurriculumLevel(
                    name="Easy",
                    min_solve_length=1,
                    max_solve_length=3,
                    success_threshold=0.8,
                    board_size=4,
                    num_robots=2,
                    max_walls=5,
                    episodes_per_evaluation=10
                )
            ]
            
            curriculum = RicochetRobotsCurriculum(
                levels=levels,
                difficulty_lookup=lookup,
                evaluation_episodes=10
            )
            
            encoder = FlatArrayEncoder()
            env = CurriculumRicochetRobotsEnv(
                curriculum=curriculum,
                encoder=encoder,
                max_episode_steps=20
            )
            
            # Record some results
            curriculum.record_episode_result(True)
            curriculum.record_episode_result(False)
            
            original_state = curriculum.get_state()
            
            # Save state
            state_file = Path(tmp_dir) / "curriculum_state.json"
            env.save_curriculum_state(str(state_file))
            
            # Create new environment and load state
            new_curriculum = RicochetRobotsCurriculum(
                levels=levels,
                difficulty_lookup=lookup,
                evaluation_episodes=10
            )
            
            new_env = CurriculumRicochetRobotsEnv(
                curriculum=new_curriculum,
                encoder=encoder,
                max_episode_steps=20
            )
            
            new_env.load_curriculum_state(str(state_file))
            
            # Verify state was loaded
            loaded_state = new_curriculum.get_state()
            assert loaded_state.current_level == original_state.current_level
            assert loaded_state.episodes_completed == original_state.episodes_completed
            assert loaded_state.success_rate == original_state.success_rate