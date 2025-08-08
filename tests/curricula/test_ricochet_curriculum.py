import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from rl_board_games.core.curriculum import (
    CurriculumLevel, 
    CurriculumState, 
    DifficultyLookup, 
    ProgressiveCurriculum
)
from rl_board_games.curricula.ricochet_robots_curriculum import RicochetRobotsCurriculum
from rl_board_games.games.ricochet_robots import RicochetRobotsGame


class TestDifficultyLookup:
    """Test difficulty lookup functionality."""
    
    def test_save_and_load_lookup_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lookup = DifficultyLookup(tmp_dir)
            
            # Test data
            test_data = {1: 5, 2: 3, 3: 8, 4: 2}
            
            # Save lookup table
            lookup.save_lookup_table(8, 3, test_data)
            
            # Load lookup table
            loaded_data = lookup.load_lookup_table(8, 3)
            
            assert loaded_data == test_data
    
    def test_get_difficulty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lookup = DifficultyLookup(tmp_dir)
            
            # Test data
            test_data = {42: 7, 123: 4, 456: 12}
            lookup.save_lookup_table(6, 2, test_data)
            
            # Test existing seed
            assert lookup.get_difficulty(42, 6, 2) == 7
            assert lookup.get_difficulty(123, 6, 2) == 4
            
            # Test non-existing seed
            assert lookup.get_difficulty(999, 6, 2) is None
    
    def test_add_difficulty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lookup = DifficultyLookup(tmp_dir)
            
            # Add some difficulties
            lookup.add_difficulty(100, 4, 2, 3)
            lookup.add_difficulty(200, 4, 2, 8)
            
            # Verify they were added
            assert lookup.get_difficulty(100, 4, 2) == 3
            assert lookup.get_difficulty(200, 4, 2) == 8


class TestCurriculumLevel:
    """Test curriculum level data structure."""
    
    def test_curriculum_level_creation(self):
        level = CurriculumLevel(
            name="Test Level",
            min_solve_length=3,
            max_solve_length=7,
            success_threshold=0.8,
            board_size=8,
            num_robots=3,
            max_walls=15
        )
        
        assert level.name == "Test Level"
        assert level.min_solve_length == 3
        assert level.max_solve_length == 7
        assert level.success_threshold == 0.8
        assert level.board_size == 8
        assert level.num_robots == 3
        assert level.max_walls == 15
        assert level.episodes_per_evaluation == 100  # default value


class TestCurriculumState:
    """Test curriculum state tracking."""
    
    def test_curriculum_state_serialization(self):
        state = CurriculumState(
            current_level=2,
            episodes_completed=150,
            success_rate=0.75,
            total_episodes=300,
            level_episodes=50
        )
        
        # Test serialization
        state_dict = state.to_dict()
        expected = {
            "current_level": 2,
            "episodes_completed": 150,
            "success_rate": 0.75,
            "total_episodes": 300,
            "level_episodes": 50
        }
        assert state_dict == expected
        
        # Test deserialization
        recovered_state = CurriculumState.from_dict(state_dict)
        assert recovered_state.current_level == 2
        assert recovered_state.episodes_completed == 150
        assert recovered_state.success_rate == 0.75
        assert recovered_state.total_episodes == 300
        assert recovered_state.level_episodes == 50


class TestRicochetRobotsCurriculum:
    """Test Ricochet Robots curriculum implementation."""
    
    def test_default_levels_creation(self):
        curriculum = RicochetRobotsCurriculum()
        
        # Should have 5 default levels
        assert len(curriculum.levels) == 5
        
        # Check level names
        expected_names = ["Easy", "Medium", "Hard", "Expert", "Master"]
        actual_names = [level.name for level in curriculum.levels]
        assert actual_names == expected_names
        
        # Check progressive difficulty
        for i in range(len(curriculum.levels) - 1):
            current_level = curriculum.levels[i]
            next_level = curriculum.levels[i + 1]
            
            # Board size should not decrease
            assert next_level.board_size >= current_level.board_size
            
            # Success threshold should not increase
            assert next_level.success_threshold <= current_level.success_threshold
    
    def test_curriculum_progression(self):
        # Create a simple curriculum with 2 levels
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
            ),
            CurriculumLevel(
                name="Hard",
                min_solve_length=5,
                max_solve_length=10,
                success_threshold=0.6,
                board_size=8,
                num_robots=3,
                max_walls=15,
                episodes_per_evaluation=10
            )
        ]
        
        curriculum = RicochetRobotsCurriculum(levels=levels, evaluation_episodes=10)
        
        # Should start at level 0
        assert curriculum.state.current_level == 0
        assert curriculum.get_current_level().name == "Easy"
        
        # Record successful episodes
        for _ in range(10):
            curriculum.record_episode_result(True)
        
        # Should advance to level 1
        assert curriculum.state.current_level == 1
        assert curriculum.get_current_level().name == "Hard"
        
        # Success rate should be 1.0
        assert curriculum.state.success_rate == 1.0
    
    def test_curriculum_no_progression_low_success(self):
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
            ),
            CurriculumLevel(
                name="Hard",
                min_solve_length=5,
                max_solve_length=10,
                success_threshold=0.6,
                board_size=8,
                num_robots=3,
                max_walls=15,
                episodes_per_evaluation=10
            )
        ]
        
        curriculum = RicochetRobotsCurriculum(levels=levels, evaluation_episodes=10)
        
        # Record mostly failed episodes (success rate = 0.3)
        for _ in range(7):
            curriculum.record_episode_result(False)
        for _ in range(3):
            curriculum.record_episode_result(True)
        
        # Should stay at level 0
        assert curriculum.state.current_level == 0
        assert curriculum.get_current_level().name == "Easy"
        assert curriculum.state.success_rate == 0.3
    
    def test_game_generation(self):
        # Create curriculum with mock difficulty lookup
        with tempfile.TemporaryDirectory() as tmp_dir:
            lookup = DifficultyLookup(tmp_dir)
            
            # Add some test difficulty data
            lookup.add_difficulty(42, 4, 2, 2)  # Easy difficulty
            lookup.add_difficulty(123, 4, 2, 3)  # Easy difficulty
            
            curriculum = RicochetRobotsCurriculum(difficulty_lookup=lookup)
            
            # Generate a game
            game_iter = iter(curriculum)
            game = next(game_iter)
            
            assert isinstance(game, RicochetRobotsGame)
            assert game.board.width == 4  # Easy level board size
            assert game.num_robots == 2  # Easy level robot count
    
    def test_curriculum_state_save_load(self):
        curriculum = RicochetRobotsCurriculum()
        
        # Record some episodes
        for _ in range(5):
            curriculum.record_episode_result(True)
        
        original_state = curriculum.get_state()
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        
        curriculum.save_state(state_file)
        
        # Create new curriculum and load state
        new_curriculum = RicochetRobotsCurriculum()
        new_curriculum.load_state(state_file)
        
        loaded_state = new_curriculum.get_state()
        
        # States should match
        assert loaded_state.current_level == original_state.current_level
        assert loaded_state.episodes_completed == original_state.episodes_completed
        assert loaded_state.success_rate == original_state.success_rate
        assert loaded_state.total_episodes == original_state.total_episodes
        assert loaded_state.level_episodes == original_state.level_episodes
        
        # Clean up
        Path(state_file).unlink()
    
    def test_curriculum_metrics(self):
        curriculum = RicochetRobotsCurriculum()
        
        # Record some episodes
        for _ in range(3):
            curriculum.record_episode_result(True)
        for _ in range(2):
            curriculum.record_episode_result(False)
        
        metrics = curriculum.get_metrics()
        
        # Check expected metrics
        assert "curriculum/current_level" in metrics
        assert "curriculum/level_name" in metrics
        assert "curriculum/success_rate" in metrics
        assert "curriculum/episodes_completed" in metrics
        assert "curriculum/total_episodes" in metrics
        assert "curriculum/level_episodes" in metrics
        assert "curriculum/target_success_rate" in metrics
        assert "curriculum/current_board_size" in metrics
        assert "curriculum/current_num_robots" in metrics
        
        # Check some values
        assert metrics["curriculum/current_level"] == 0
        assert metrics["curriculum/level_name"] == "Easy"
        assert metrics["curriculum/success_rate"] == 0.6  # 3/5
        assert metrics["curriculum/episodes_completed"] == 5
        assert metrics["curriculum/current_board_size"] == 4
        assert metrics["curriculum/current_num_robots"] == 2
    
    def test_get_seeds_for_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lookup = DifficultyLookup(tmp_dir)
            
            # Add test data for Easy level (1-3 moves, 4x4 board, 2 robots)
            lookup.add_difficulty(1, 4, 2, 2)   # Easy
            lookup.add_difficulty(2, 4, 2, 3)   # Easy
            lookup.add_difficulty(3, 4, 2, 5)   # Not easy (too hard)
            lookup.add_difficulty(4, 4, 2, 1)   # Easy
            
            curriculum = RicochetRobotsCurriculum(difficulty_lookup=lookup)
            
            # Get seeds for Easy level
            seeds = curriculum.get_seeds_for_level("Easy", count=2)
            
            # Should return valid seeds (1, 2, 4 are valid for Easy level)
            assert len(seeds) <= 2
            for seed in seeds:
                difficulty = lookup.get_difficulty(seed, 4, 2)
                assert 1 <= difficulty <= 3