"""
Debug optimality calculation to find the issue
"""
import sys
sys.path.insert(0, '..')

import pickle
import numpy as np
from optimality_analysis import a_star_pathfinding

# Load dataset
with open('maze_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    test_mazes = data['test_mazes']

print("="*60)
print("DEBUGGING OPTIMALITY CALCULATION")
print("="*60)

# Test first 5 mazes
for i in range(min(5, len(test_mazes))):
    maze = test_mazes[i]
    
    print(f"\n--- Maze {i+1} ---")
    print(f"Grid size: {maze['grid'].shape}")
    print(f"Start: {maze['start_pos']}")
    print(f"Goal: {maze['target_pos']}")
    
    # Run A*
    path, optimal_len = a_star_pathfinding(
        maze['grid'],
        maze['start_pos'],
        maze['target_pos']
    )
    
    print(f"A* optimal path length: {optimal_len} steps")
    
    if path:
        print(f"Path: {path[:5]}... (showing first 5 positions)")
        print(f"Path has {len(path)} positions")
        print(f"Steps = positions - 1 = {len(path) - 1}")
    else:
        print("No path found!")
    
    # Check maze's recorded path length
    if 'path_length' in maze:
        print(f"Maze's stored path_length: {maze['path_length']}")

print("\n" + "="*60)
print("SUMMARY OF FIRST 5 MAZES")
print("="*60)

optimal_lengths = []
for i in range(min(5, len(test_mazes))):
    maze = test_mazes[i]
    _, optimal_len = a_star_pathfinding(maze['grid'], maze['start_pos'], maze['target_pos'])
    optimal_lengths.append(optimal_len)
    print(f"Maze {i+1}: {optimal_len} steps")

print(f"\nAverage optimal path: {np.mean(optimal_lengths):.1f} steps")