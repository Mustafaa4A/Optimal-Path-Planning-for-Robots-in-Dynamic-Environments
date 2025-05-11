# CS1001 Advanced Algorithms Project: Optimal Path Planning for Robots in Dynamic Environments

## Description
This project implements an A* based path planner for a robot navigating a 2D maze with multiple targets. The robot must find the shortest path visiting all targets while avoiding obstacles. The solution implements a Traveling Salesman Problem (TSP) approach for optimal target ordering.

## Key Features

### 1. Multiple Dynamic Targets
- Randomly generates 2-4 target points
- Targets are placed in valid positions (not on walls or start point)
- Uses numpy's random seed for reproducibility

### 2. Path Planning
- Implements A* algorithm for optimal path finding
- Uses Manhattan distance as heuristic
- Handles obstacle avoidance
- Solves TSP for optimal target ordering

### 3. Visualization
- Real-time animation of path exploration
- Custom color scheme for clear visualization
- Saves both GIF animation, PNG result and TXT file for detailed statistics
- Displays explored nodes and optimal path

### 4. Performance Metrics
- Tracks computation time
- Counts nodes expanded
- Measures total path length
- Records target visit order

### 5. Output Files
- `a_star_multi_target.gif`: Animation of path finding
- `a_star_multi_target_result.png`: Final path visualization
- `path_planning_output.txt`: Detailed statistics and metrics

## Code Structure

### 1. Maze Setup
- Maze definition (12x24 grid)
- Border addition
- Target generation
- Start position initialization

### 2. Core Classes
- `Node`: Represents a position in the maze with path costs
- Custom color mapping for visualization

### 3. Key Functions
- `a_star()`: Main path finding algorithm
- `total_path_length()`: Calculates path distance
- `update()`: Animation frame update function

### 4. Main Algorithm Flow
- Generate random targets
- Find optimal target order
- Calculate paths between targets
- Visualize and save results

## Requirements
- Python 3.x
- matplotlib
- numpy
- pillow (PIL)

## Installation
```bash
pip install matplotlib numpy pillow
```

## Usage
1. Clone the code
2. Install the required packages
3. Run the script:
```bash
python main.py 
```

4. Output files will be generated in the current directory:
   - `a_star_multi_target.gif`
   - `a_star_multi_target_result.png`
   - `path_planning_output.txt`

## Customization
- Adjust `ROWS` and `COLS` for different maze sizes
- Modify `BORDER_THICKNESS` for border width
- Change color scheme in `custom_cmap`
- Adjust animation speed (interval parameter)
- Modify number of targets (`num_targets`)

## Color Scheme
- Walls: Light green
- Free space: Off-white
- Path: Orange line
- Start: Blue star
- Targets: Red circles
- Explored nodes: Light yellow

## Output Example
The program generates three main output files:
1. An animated GIF showing the path-finding process
2. A PNG image of the final optimal path
3. A text file containing detailed statistics and metrics

## Author
Mustaf Abubakar Abdullahi (MCS240014)

## Version
1.0.0

## License
This project is part of the CS1001 Advanced Algorithms course. 