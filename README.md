# CS1001 Advanced Algorithms Project: Optimal Path Planning for Robots in Dynamic Environments

## Description
This project implements an A* based path planner for a robot navigating a 2D maze with multiple targets. The robot must find the shortest path visiting all targets while avoiding obstacles. The solution implements a Traveling Salesman Problem (TSP) approach for optimal target ordering and uses a maze with dynamically placed targets.

## Key Features

### 1. Multiple Dynamic Targets
- Randomly generates 2-4 target points
- Targets are placed in valid positions (not on walls or start point)
- Uses numpy's random seed for reproducibility

### 2. Path Planning
- Implements A* algorithm for optimal pathfinding
- Uses Manhattan distance as heuristic for estimating cost
- Handles obstacle avoidance and navigation in a dynamic environment
- Solves Traveling Salesman Problem (TSP) for optimal target ordering

### 3. Visualization
- Real-time animation of path exploration
- Custom color scheme for clear visualization
- Saves both GIF animation, PNG result, and TXT file for detailed statistics
- Displays explored nodes and optimal path

### 4. Performance Metrics
- Tracks computation time for pathfinding
- Counts nodes expanded during the search
- Measures total path length
- Records the target visit order and explores nodes

### 5. Output Files
- `a_star_target.gif`: Animation of pathfinding
- `a_star_target_result.png`: Final path visualization in PNG format
- `path_planning_output.txt`: Detailed statistics, including optimal path, nodes expanded, and target positions

## Code Structure

### 1. Maze Setup
- Maze definition (12x24 grid)
- Border addition around the maze to ensure proper boundaries
- Random target generation within valid positions
- Start position initialization

### 2. Core Classes
- `Node`: Represents a position in the maze with associated path costs (`g`, `h`, `f`) and parent node
- `PathPlanner`: Handles the A* pathfinding logic, maze setup, and target ordering logic using permutations
- `Visualizer`: Manages the visualization of the pathfinding process, including real-time updates and animation generation

### 3. Key Functions
- `a_star()`: Implements the A* pathfinding algorithm
- `total_path_length()`: Calculates the total distance of the planned path
- `update_animation()`: Animation update function for real-time exploration visualization
- `save_visualization()`: Saves the generated visualization as GIF and PNG files

### 4. Main Algorithm Flow
- Generate random target points in the maze
- Find the optimal path by considering all possible orders of target visits (TSP approach)
- Visualize the exploration process and final path
- Save results and performance metrics to output files

## Requirements
- Python 3.x
- matplotlib
- numpy
- pillow (PIL)

## Installation
```bash
pip install matplotlib numpy pillow
