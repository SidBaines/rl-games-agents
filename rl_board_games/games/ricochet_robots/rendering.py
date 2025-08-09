from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .board import Board, NORTH, EAST, SOUTH, WEST


def render_rgb(
    board: Board,
    robots: Sequence[Tuple[int, int]],
    goal: Tuple[int, int],
    goal_robot: int,
    scale: int = 20,
) -> np.ndarray:
    """
    Render the given board and state as an HWC uint8 RGB image.

    - walls: black lines
    - robots: solid colored squares
    - goal: colored border matching the goal robot

    Parameters
    ----------
    board: Board
        The board object for wall queries and size
    robots: Sequence[(x, y)]
        Robot positions
    goal: (x, y)
        Goal cell position
    goal_robot: int
        Index of the robot that must reach the goal
    scale: int
        Pixels per cell (min 8 for proper border drawing)
    """
    scale = max(8, int(scale))
    h, w = board.height * scale, board.width * scale
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    # Draw walls (black)
    for y in range(board.height):
        for x in range(board.width):
            cx, cy = x * scale, y * scale
            if board.has_wall(x, y, NORTH):
                img[cy, cx : cx + scale, :] = 0
            if board.has_wall(x, y, WEST):
                img[cy : cy + scale, cx, :] = 0
    # Draw outer south/east walls
    for x in range(board.width):
        if board.has_wall(x, board.height - 1, SOUTH):
            cy = h - 1
            cx = x * scale
            img[cy, cx : cx + scale, :] = 0
    for y in range(board.height):
        if board.has_wall(board.width - 1, y, EAST):
            cx = w - 1
            cy = y * scale
            img[cy : cy + scale, cx, :] = 0

    # Colors for robots and goal
    colors = np.array([[255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]], dtype=np.uint8)

    # Goal border
    gx, gy = goal
    gx_pix, gy_pix = gx * scale, gy * scale
    goal_color = colors[goal_robot % len(colors)]
    thickness = 2
    pad = 3
    img[gy_pix + pad : gy_pix + pad + thickness, gx_pix + pad : gx_pix + scale - pad, :] = goal_color
    img[gy_pix + scale - pad - thickness : gy_pix + scale - pad, gx_pix + pad : gx_pix + scale - pad, :] = goal_color
    img[gy_pix + pad : gy_pix + scale - pad, gx_pix + pad : gx_pix + pad + thickness, :] = goal_color
    img[gy_pix + pad : gy_pix + scale - pad, gx_pix + scale - pad - thickness : gx_pix + scale - pad, :] = goal_color

    # Robots
    for idx, (x, y) in enumerate(robots):
        cx, cy = x * scale, y * scale
        color = colors[idx % len(colors)]
        img[cy + pad : cy + scale - pad, cx + pad : cx + scale - pad, :] = color

    return img 