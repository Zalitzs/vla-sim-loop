# FILE: scripts/test_movement_coords.py
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from envs.grid_bullet_world import GridBulletWorld

# Setup visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

print("="*60)
print("Testing Movement and Coordinate System")
print("="*60)

# Create environment
env = GridBulletWorld(gui=True, grid_size=10, dynamic_rules=[])
env.reset()

# Get initial position
initial_pos = env.get_cube_grid_pos()
print(f"\nInitial cube position: {initial_pos}")

# Setup plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

def show_grid(title):
    """Helper to display grid"""
    grid = env.get_grid_state()
    ax.clear()
    ax.imshow(np.flipud(grid), cmap=cmap, norm=norm)
    ax.set_title(title)
    
    # Add coordinate labels
    cube_pos = env.get_cube_grid_pos()
    ax.set_xlabel(f"Cube at: {cube_pos}")
    
    plt.draw()
    plt.pause(1.5)

# Show initial state
show_grid("Initial Position")

# Test each action and see what happens
actions_to_test = ['forward', 'backward', 'left', 'right']

print("\n" + "="*60)
print("Testing each action:")
print("="*60)

for action in actions_to_test:
    # Record position before
    pos_before = env.get_cube_grid_pos()
    
    # Take action
    obs, reward, done, info = env.step(action)
    
    # Record position after
    pos_after = env.get_cube_grid_pos()
    
    # Calculate change
    dx = pos_after[0] - pos_before[0]
    dy = pos_after[1] - pos_before[1]
    
    # Display
    print(f"\nAction: '{action}'")
    print(f"  Before: {pos_before}")
    print(f"  After:  {pos_after}")
    print(f"  Change: dx={dx:+d}, dy={dy:+d}")
    
    if dx > 0:
        print(f"  → Moved RIGHT (x increased)")
    elif dx < 0:
        print(f"  → Moved LEFT (x decreased)")
    
    if dy > 0:
        print(f"  → Moved UP (y increased)")
    elif dy < 0:
        print(f"  → Moved DOWN (y decreased)")
    
    # Show on grid
    show_grid(f"After '{action}': {pos_after}")
    
    if done:
        print("\n  (Reached goal or hit limit)")
        break

print("\n" + "="*60)
print("Summary of Movement Mappings:")
print("="*60)

plt.close()
env.disconnect()