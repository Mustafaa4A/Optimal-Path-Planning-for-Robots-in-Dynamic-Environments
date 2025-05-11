"""
CS1001 Advanced Algorithms Project: Optimal Path Planning for Robots in Dynamic Environments

For detailed documentation, features, and usage instructions, please refer to README.md
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import heapq
import itertools
import time
from matplotlib.colors import ListedColormap

ROWS, COLS = 12, 24

# Maze definition (1: wall, 0: space, -1: start, 9: target)
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

# Find start position
start = tuple(map(int, np.argwhere(maze == -1)[0]))

# Generate 2-4 random target points (not on wall or start)
empty_cells = list(zip(*np.where(maze == 0)))
np.random.seed(142)
num_targets = np.random.randint(2, 5)
possible_targets = [cell for cell in empty_cells if cell != start]
target_indices = np.random.choice(len(possible_targets), num_targets, replace=False)
target_points = [possible_targets[i] for i in target_indices]

# Place targets in the maze (value 9)
maze_targets = maze.copy()
for r, c in target_points:
    maze_targets[r, c] = 9

# --- Add a thick green border around the maze ---
BORDER_THICKNESS = 1
BORDER_VALUE = 2  # New value for border

# Create a new maze with border
maze_with_border = np.full((ROWS + 2 * BORDER_THICKNESS, COLS + 2 * BORDER_THICKNESS), BORDER_VALUE, dtype=int)
maze_with_border[BORDER_THICKNESS:ROWS + BORDER_THICKNESS, BORDER_THICKNESS:COLS + BORDER_THICKNESS] = maze

# Adjust start and target positions for border offset
start_offset = (start[0] + BORDER_THICKNESS, start[1] + BORDER_THICKNESS)
target_points_offset = [(r + BORDER_THICKNESS, c + BORDER_THICKNESS) for r, c in target_points]
maze_targets = maze_with_border.copy()
for r, c in target_points_offset:
    maze_targets[r, c] = 9

# --- Custom color map: 0=free, 1=wall, 2=border, 9=target ---
custom_cmap = ListedColormap([
    '#FFFFFF',  # 0: clear space (white)
    '#00FF00',  # 1: wall (green)
    '#00FF00',  # 2: border (green, same as wall)
    '#F39C12',  # 3: path (orange, for overlay)
    '#F9E79F',  # 4: explored (light yellow, for overlay)
])

fig, ax = plt.subplots()
img = ax.imshow(
    np.where(maze_targets == -1, 0, np.where(maze_targets == 9, 0, maze_targets)),
    cmap=custom_cmap, vmin=0, vmax=4
)

class Node:
    def __init__(self, pos, g, h, parent):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    def __lt__(self, other):
        return self.f < other.f

def a_star(maze, start, end):
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
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] in [0, 9] and (nr, nc) not in visited:
                h = abs(nr - end[0]) + abs(nc - end[1])
                heapq.heappush(heap, Node((nr, nc), current.g + 1, h, current))
    return None, nodes_expanded, set()

def total_path_length(path):
    if not path or len(path) < 2:
        return 0
    return sum(abs(path[i][0] - path[i-1][0]) + abs(path[i][1] - path[i-1][1]) for i in range(1, len(path)))

# Solve the multi-target path (TSP for small n)
print(f"Starting multi-target path search with {len(target_points_offset)} targets...")
all_orders = list(itertools.permutations(target_points_offset))
shortest = None
shortest_path = []
shortest_length = float('inf')
shortest_nodes_expanded = 0
shortest_visited = set()
start_time = time.time()
for idx, order in enumerate(all_orders):
    print(f"Evaluating order {idx+1}/{len(all_orders)}: {order}")
    curr_pos = start_offset
    order_path = []
    order_length = 0
    order_nodes_expanded = 0
    order_visited = set()
    for tgt in order:
        maze_tmp = maze_targets.copy()
        for t in target_points_offset:
            if t != tgt:
                maze_tmp[t] = 9
            else:
                maze_tmp[t] = 0
        path_, nodes_expanded, visited_order = a_star(maze_tmp, curr_pos, tgt)
        if path_ is None:
            break
        order_path.extend(path_[1:] if order_path else path_)
        order_length += total_path_length(path_)
        order_nodes_expanded += nodes_expanded
        order_visited.update(visited_order)
        curr_pos = tgt
    else:
        if order_length < shortest_length:
            print(f"New shortest path found for order {order} with length {order_length}")
            shortest = order
            shortest_path = order_path
            shortest_length = order_length
            shortest_nodes_expanded = order_nodes_expanded
            shortest_visited = order_visited
end_time = time.time()
print("Optimal path search complete. Preparing animation and export...")

# Prepare statistics for display
stats_text = (
    f"Start: {start}\n"
    f"Target order: {shortest}\n"
    f"Total path length: {shortest_length}\n"
    f"Optimal path: {shortest_path}\n"
    f"Nodes expanded: {shortest_nodes_expanded}\n"
    f"Computation time: {end_time - start_time:.4f} s\n"
    f"Number of targets: {len(target_points)}\n"
    f"Target positions: {target_points}"
)

# Legend patches
legend_patches = [
    mpatches.Patch(color='#FFFFFF', label='Clear space'),
    mpatches.Patch(color='#00FF00', label='Obstacle'),
    mpatches.Patch(color='#F39C12', label='Path'),
    mpatches.Patch(color='#F9E79F', label='Explored'),
    mpatches.Patch(color='blue', label='Origin'),
    mpatches.Patch(color='red', label='Goal'),
]

# Animation update function (only animates the path)
def update(frame):
    img.set_data(np.where(maze_targets == -1, 0, np.where(maze_targets == 9, 0, maze_targets)))
    # Draw explored nodes as yellow dots
    if frame > 0 and shortest_visited:
        explored_y, explored_x = zip(*shortest_visited)
        ax.scatter(explored_x, explored_y, c='#F9E79F', s=15, marker='.', zorder=3)
    # Draw path up to current frame
    if frame > 0:
        y, x = zip(*shortest_path[:frame+1])
        ax.plot(x, y, color='#F39C12', linewidth=2.5, zorder=4)
    # Draw start and targets
    ax.scatter(start_offset[1], start_offset[0], c='blue', s=180, marker='*', label='Origin', edgecolors='black', zorder=5)
    for t in target_points_offset:
        ax.scatter(t[1], t[0], c='red', s=180, marker='o', label='Goal', edgecolors='black', zorder=5)
    # Add legend
    if frame == 0:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.9)
    return [img]

anim = animation.FuncAnimation(
    fig, update, frames=len(shortest_path), interval=120, repeat=False)

plt.title("Optimal Path Planning for Robots in Dynamic Environments")
plt.axis('off')
plt.tight_layout()
anim.save('a_star_multi_target.gif', writer='pillow', fps=8)
_ = anim  # Keep reference to prevent garbage collection

# Print statistics/info in the console only
print(stats_text)
print("Exporting GIF and PNG...")

# Save the last frame as PNG (no info box)
for txt in ax.texts:
    txt.remove()
fig.savefig('a_star_multi_target_result.png', bbox_inches='tight')

# Write statistics to a text file
with open('path_planning_output.txt', 'w') as f:
    f.write("Optimal Path Planning for Robots in Dynamic Environments\n")
    f.write("====================================================\n\n")
    f.write(f"Start Position: {tuple(map(int, start))}\n\n")
    f.write(f"Target Order: \n{tuple(tuple(map(int, pos)) for pos in shortest)}\n\n")
    f.write(f"Total Path Length: {shortest_length}\n\n")
    f.write("Optimal Path:\n")
    f.write(f"{[tuple(map(int, pos)) for pos in shortest_path]}\n\n")
    f.write("Performance Metrics:\n")
    f.write("------------------\n")
    f.write(f"Nodes expanded: {shortest_nodes_expanded}\n")
    f.write(f"Computation time: {end_time - start_time:.4f} seconds\n")
    f.write(f"Number of targets: {len(target_points)}\n\n")
    f.write("Target Positions:\n")
    f.write(f"{[tuple(map(int, pos)) for pos in target_points]}\n")
    f.write("\nExplored Points (Visited Nodes):\n")
    f.write(f"{[tuple(map(int, pos)) for pos in shortest_visited]}\n")

