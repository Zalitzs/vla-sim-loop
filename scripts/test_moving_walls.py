# FILE: scripts/test_moving_walls.py
import sys
sys.path.append('..')

import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from envs.grid_bullet_world import GridBulletWorld
from envs.env_rules import shift_wall_right_every_5
from agents.astar_agent import AStarAgent

# Setup visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create environment WITH dynamic rule
print("Testing Dynamic Walls with A* Agent")
print("="*60)
env = GridBulletWorld(gui=True, grid_size=20, dynamic_rules=[shift_wall_right_every_5])
env.reset()

agent = AStarAgent()

# Setup plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

print("\nWatch for '[Rule] Walls shifting right' messages every 5 steps")
print("Also watch the visualization - walls should move right!\n")

# Run episode
for i in range(40):  # Run for 40 steps to see multiple shifts
    # Get action from A*
    action = agent.get_action(env)
    
    # Take step
    obs, reward, done, info = env.step(action)
    
    # Get positions
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    
    print(f"Step {env.step_count}: {action:8s} → Cube={cube_pos}, Target={target_pos}")
    
    # Visualize
    grid = env.get_grid_state()
    ax.clear()
    ax.imshow(np.flipud(grid), cmap=cmap, norm=norm)
    ax.set_title(f"Step {env.step_count}: {action} | Cube→Target")
    ax.set_xlabel(f"Cube: {cube_pos} → Target: {target_pos}")
    
    # Highlight when walls should shift
    if env.step_count % 5 == 0:
        ax.text(0.5, 0.98, "⚠️ WALLS SHIFTING NOW!", 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=14, fontweight='bold')
    
    plt.draw()
    plt.pause(0.5)
    
    if done:
        print(f"\n{'✓ SUCCESS!' if info['success'] else '✗ FAILED'}")
        plt.pause(2)
        break

plt.close()
env.disconnect()