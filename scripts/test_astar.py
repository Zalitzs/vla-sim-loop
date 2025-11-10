# FILE: scripts/test_astar.py
import sys
sys.path.append('..')  # Add parent directory to path
import time  # <-- ADD THIS IMPORT
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from envs.grid_bullet_world import GridBulletWorld
from agents.astar_agent import AStarAgent

# Setup matplotlib visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create environment
env = GridBulletWorld(gui=True, grid_size=20)
env.reset()

# Create agent
agent = AStarAgent()

# Setup plot
plt.ion()  # Interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

# Run episode
print("Testing A* agent...")
for i in range(50):
    # Get action and step
    action = agent.get_action(env)
    obs, reward, done, info = env.step(action)
    
    # Get positions for printing
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    print(f"Step {i}: Action={action}, Cube={cube_pos}, Target={target_pos}")
    
    # Visualize the grid
    grid = env.get_grid_state()
    ax.clear()
    ax.imshow(np.flipud(grid), cmap=cmap, norm=norm)
    ax.set_title(f"Step {i}: Action={action}")
    ax.set_xlabel(f"Cube: {cube_pos} → Target: {target_pos}")
    plt.draw()
    plt.pause(0.5)  # <-- SLEEP HERE! Adjust this value (0.5 seconds)
    
    if done:
        print(f"\n{'✓ SUCCESS!' if info['success'] else '✗ Failed'}")
        print(f"Reached goal in {i+1} steps")
        plt.pause(2)  # Pause longer at the end to see final state
        break

plt.close()
env.disconnect()