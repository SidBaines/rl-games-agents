import pytest
import numpy as np

from rl_board_games.games.ricochet_robots import RicochetRobotsGame, AStarSolver
from rl_board_games.games.ricochet_robots.board import Board, NORTH, EAST, SOUTH, WEST


def test_slide_until_wall():
    game = RicochetRobotsGame()
    state = game.reset(seed=42)
    # Pick first robot, move north; ensure it slides to boundary row 0
    robot_idx = 0
    x, y = state.robots[robot_idx]
    new_state, _, _, _ = game.step((robot_idx, NORTH))
    nx, ny = new_state.robots[robot_idx]
    assert ny == 0  # should slide to top edge due to boundary wall
    # Moving north again should not change position
    newer_state, _, _, _ = game.step((robot_idx, NORTH))
    nnx, nny = newer_state.robots[robot_idx]
    assert (nnx, nny) == (nx, ny)


def test_legal_actions_nonempty():
    game = RicochetRobotsGame()
    state = game.reset(seed=123)
    actions = game.legal_actions(state)
    assert len(actions) > 0
    # Ensure each returned action actually changes position
    for robot_idx, direction in actions:
        robots = list(state.robots)
        x, y = robots[robot_idx]
        nx, ny = game.board.next_position(x, y, direction, robots)
        assert (nx, ny) != (x, y)


def test_astar_solver_reaches_goal():
    # Create a simple 5x5 board with no internal walls
    board = Board.empty(5)
    # Place two robots, one at (0,0), one at (4,4), goal at (4,0), goal_robot=0
    robots = ((0, 0), (4, 4))
    goal = (4, 0)
    goal_robot = 0
    from rl_board_games.games.ricochet_robots.game import RRGameState
    state = RRGameState(robots=robots, goal=goal, goal_robot=goal_robot, move_count=0, board=board)
    game = RicochetRobotsGame(board=board, num_robots=2)
    solver = AStarSolver(game, max_depth=10)
    actions = solver.solve(state)
    # The optimal solution is: move robot 0 east to (4,0) in one move
    assert len(actions) == 1
    assert actions[0] == (0, EAST)
    # Apply actions sequentially; should reach terminal
    current_state = state
    game._state = current_state
    done = False
    for act in actions:
        current_state, _, done, _ = game.step(act)
    assert done or current_state.is_terminal or len(actions) == 0 