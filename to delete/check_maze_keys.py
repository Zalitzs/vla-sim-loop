"""
Check what keys are in the maze templates
"""
import pickle

with open('maze_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    test_mazes = data['test_mazes']

print("="*60)
print("MAZE TEMPLATE STRUCTURE")
print("="*60)

maze = test_mazes[0]
print("\nKeys in maze template:")
for key in maze.keys():
    print(f"  - {key}: {type(maze[key])}")

print("\nFirst maze details:")
print(f"  start_pos: {maze.get('start_pos', 'NOT FOUND')}")
print(f"  target_pos: {maze.get('target_pos', 'NOT FOUND')}")
print(f"  start: {maze.get('start', 'NOT FOUND')}")  
print(f"  target: {maze.get('target', 'NOT FOUND')}")
print(f"  grid shape: {maze['grid'].shape}")