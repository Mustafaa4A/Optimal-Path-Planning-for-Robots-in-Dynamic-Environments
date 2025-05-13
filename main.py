"""
CS1001 Advanced Algorithms Project: Optimal Path Planning for Robots in Dynamic Environments

This module implements an A* path planning algorithm for robots in dynamic environments
with multiple targets. It includes visualization capabilities and performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import heapq
import itertools
import time
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict
from matplotlib.colors import ListedColormap

# Constants
ROWS, COLS = 12, 24
BORDER_THICKNESS = 1
BORDER_VALUE = 2

# Color definitions
COLORS = {
    'clear': '#FFFFFF',
    'wall': '#00FF00',
    'path': '#F39C12',
    'explored': '#F9E79F',
    'start': 'blue',
    'target': 'red'
}

@dataclass
class Node:
    """Represents a node in the A* search algorithm."""
    pos: Tuple[int, int]
    g: int  # Cost from start to current node
    h: int  # Heuristic cost from current to goal
    parent: Optional['Node'] = None

    @property
    def f(self) -> int:
        """Total cost (f = g + h)."""
        return self.g + self.h

    def __lt__(self, other: 'Node') -> bool:
        """Comparison for priority queue ordering."""
        return self.f < other.f

class PathPlanner:
    """Main class for path planning functionality."""

    def __init__(self, maze: np.ndarray):
        """Initialize the path planner with a maze."""
        self.maze = maze
        self.start = tuple(map(int, np.argwhere(maze == -1)[0]))
        self.target_points = self._generate_targets()
        self.maze_with_border = self._add_border()
        self.start_offset = (self.start[0] + BORDER_THICKNESS, self.start[1] + BORDER_THICKNESS)
        self.target_points_offset = [(r + BORDER_THICKNESS, c + BORDER_THICKNESS) 
                                   for r, c in self.target_points]
        self.maze_targets = self._prepare_maze_targets()

    def _generate_targets(self) -> List[Tuple[int, int]]:
        """Generate random target points in the maze."""
        empty_cells = list(zip(*np.where(self.maze == 0)))
        np.random.seed(142)
        num_targets = np.random.randint(2, 5)
        possible_targets = [cell for cell in empty_cells if cell != self.start]
        target_indices = np.random.choice(len(possible_targets), num_targets, replace=False)
        return [possible_targets[i] for i in target_indices]

    def _add_border(self) -> np.ndarray:
        """Add a border around the maze."""
        maze_with_border = np.full(
            (ROWS + 2 * BORDER_THICKNESS, COLS + 2 * BORDER_THICKNESS),
            BORDER_VALUE,
            dtype=int
        )
        maze_with_border[
            BORDER_THICKNESS:ROWS + BORDER_THICKNESS,
            BORDER_THICKNESS:COLS + BORDER_THICKNESS
        ] = self.maze
        return maze_with_border

    def _prepare_maze_targets(self) -> np.ndarray:
        """Prepare the maze with target points."""
        maze_targets = self.maze_with_border.copy()
        for r, c in self.target_points_offset:
            maze_targets[r, c] = 9
        return maze_targets

    def a_star(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], int, Set[Tuple[int, int]]]:
        """Implement A* pathfinding algorithm."""
        heap = [Node(start, 0, abs(start[0]-end[0]) + abs(start[1]-end[1]), None)]
        visited = set()
        nodes_expanded = 0

        while heap:
            current = heapq.heappop(heap)
            r, c = current.pos

            if current.pos in visited:
                continue

            visited.add(current.pos)
            nodes_expanded += 1

            if current.pos == end:
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                path.reverse()
                return path, nodes_expanded, visited

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < ROWS + 2 * BORDER_THICKNESS and 
                    0 <= nc < COLS + 2 * BORDER_THICKNESS and 
                    maze[nr, nc] in [0, 9] and 
                    (nr, nc) not in visited):
                    h = abs(nr - end[0]) + abs(nc - end[1])
                    heapq.heappush(heap, Node((nr, nc), current.g + 1, h, current))

        return None, nodes_expanded, set()

    def total_path_length(self, path: List[Tuple[int, int]]) -> int:
        """Calculate the total length of a path."""
        if not path or len(path) < 2:
            return 0
        return sum(abs(path[i][0] - path[i-1][0]) + abs(path[i][1] - path[i-1][1]) 
                  for i in range(1, len(path)))

    def find_optimal_path(self) -> Tuple[List[Tuple[int, int]], int, int, Set[Tuple[int, int]]]:
        """Find the optimal path visiting all targets."""
        all_orders = list(itertools.permutations(self.target_points_offset))
        shortest_path = []
        shortest_length = float('inf')
        shortest_nodes_expanded = 0
        shortest_visited = set()
        start_time = time.time()

        for order in all_orders:
            curr_pos = self.start_offset
            order_path = []
            order_length = 0
            order_nodes_expanded = 0
            order_visited = set()

            for tgt in order:
                maze_tmp = self.maze_targets.copy()
                for t in self.target_points_offset:
                    maze_tmp[t] = 9 if t != tgt else 0

                path_, nodes_expanded, visited_order = self.a_star(maze_tmp, curr_pos, tgt)
                if path_ is None:
                    break

                order_path.extend(path_[1:] if order_path else path_)
                order_length += self.total_path_length(path_)
                order_nodes_expanded += nodes_expanded
                order_visited.update(visited_order)
                curr_pos = tgt
            else:
                if order_length < shortest_length:
                    shortest_path = order_path
                    shortest_length = order_length
                    shortest_nodes_expanded = order_nodes_expanded
                    shortest_visited = order_visited

        end_time = time.time()
        return shortest_path, shortest_length, shortest_nodes_expanded, shortest_visited

class Visualizer:
    """Class for handling visualization of the path planning results."""

    def __init__(self, planner: PathPlanner):
        """Initialize the visualizer with a path planner."""
        self.planner = planner
        self.fig, self.ax = plt.subplots()
        self.custom_cmap = ListedColormap([
            COLORS['clear'],  # 0: clear space
            COLORS['wall'],   # 1: wall
            COLORS['wall'],   # 2: border
            COLORS['path'],   # 3: path
            COLORS['explored']  # 4: explored
        ])
        self.img = self.ax.imshow(
            np.where(self.planner.maze_targets == -1, 0,
                    np.where(self.planner.maze_targets == 9, 0, self.planner.maze_targets)),
            cmap=self.custom_cmap,
            vmin=0,
            vmax=4
        )

    def create_legend_patches(self) -> List[mpatches.Patch]:
        """Create legend patches for the visualization."""
        return [
            mpatches.Patch(color=COLORS['clear'], label='Clear space'),
            mpatches.Patch(color=COLORS['wall'], label='Obstacle'),
            mpatches.Patch(color=COLORS['path'], label='Path'),
            mpatches.Patch(color=COLORS['explored'], label='Explored'),
            mpatches.Patch(color=COLORS['start'], label='Origin'),
            mpatches.Patch(color=COLORS['target'], label='Goal'),
        ]

    def update_animation(self, frame: int, shortest_path: List[Tuple[int, int]], 
                        shortest_visited: Set[Tuple[int, int]]) -> List:
        """Update function for the animation."""
        self.img.set_data(np.where(self.planner.maze_targets == -1, 0,
                                 np.where(self.planner.maze_targets == 9, 0, self.planner.maze_targets)))

        if frame > 0 and shortest_visited:
            explored_y, explored_x = zip(*shortest_visited)
            self.ax.scatter(explored_x, explored_y, c=COLORS['explored'], s=15, marker='.', zorder=3)

        if frame > 0:
            y, x = zip(*shortest_path[:frame+1])
            self.ax.plot(x, y, color=COLORS['path'], linewidth=2.5, zorder=4)

        self.ax.scatter(self.planner.start_offset[1], self.planner.start_offset[0],
                       c=COLORS['start'], s=180, marker='*', label='Origin',
                       edgecolors='black', zorder=5)

        for t in self.planner.target_points_offset:
            self.ax.scatter(t[1], t[0], c=COLORS['target'], s=180, marker='o',
                           label='Goal', edgecolors='black', zorder=5)

        if frame == 0:
            self.ax.legend(handles=self.create_legend_patches(), loc='upper right',
                          fontsize=8, framealpha=0.9)

        return [self.img]

    def save_visualization(self, shortest_path: List[Tuple[int, int]], 
                          shortest_visited: Set[Tuple[int, int]]) -> None:
        """Save the visualization as GIF and PNG."""
        anim = animation.FuncAnimation(
            self.fig, self.update_animation,
            fargs=(shortest_path, shortest_visited),
            frames=len(shortest_path),
            interval=120,
            repeat=False
        )

        plt.title("Optimal Path Planning for Robots in Dynamic Environments")
        plt.axis('off')
        plt.tight_layout()
        anim.save('a_star_target.gif', writer='pillow', fps=8)
        self.fig.savefig('a_star_target_result.png', bbox_inches='tight')

def main():
    """Main function to run the path planning algorithm."""
    # Initialize maze
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 9, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    # Create path planner and find optimal path
    planner = PathPlanner(maze)
    shortest_path, shortest_length, shortest_nodes_expanded, shortest_visited = planner.find_optimal_path()

    # Create visualizer and save results
    visualizer = Visualizer(planner)
    visualizer.save_visualization(shortest_path, shortest_visited)

    # Save statistics to file
    with open('path_planning_output.txt', 'w') as f:
        f.write("Optimal Path Planning for Robots in Dynamic Environments\n")
        f.write("====================================================\n\n")
        f.write(f"Start Position: {planner.start}\n\n")
        f.write(f"Target Order: \n{tuple(tuple(map(int, pos)) for pos in planner.target_points)}\n\n")
        f.write(f"Total Path Length: {shortest_length}\n\n")
        f.write("Optimal Path:\n")
        f.write(f"{[tuple(map(int, pos)) for pos in shortest_path]}\n\n")
        f.write("Performance Metrics:\n")
        f.write("------------------\n")
        f.write(f"Nodes expanded: {shortest_nodes_expanded}\n")
        f.write(f"Number of targets: {len(planner.target_points)}\n\n")
        f.write("Target Positions:\n")
        f.write(f"{[tuple(map(int, pos)) for pos in planner.target_points]}\n")
        f.write("\nExplored Points (Visited Nodes):\n")
        f.write(f"{[tuple(map(int, pos)) for pos in shortest_visited]}\n")

if __name__ == "__main__":
    main()

