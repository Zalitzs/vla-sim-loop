# FILE: scripts/test_llm_simple_maze.py
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from envs.grid_bullet_world import GridBulletWorld
from agents.llm_agent import LLMAgent

# Setup visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

print("="*60)
print("Testing LLM Agent on Simple Maze")
print("="*60)

# Create environment
env = GridBulletWorld(gui=False, grid_size=10, dynamic_rules=[])
env.reset()

# Create LLM agent
agent = LLMAgent(model="gpt-4o-mini", temperature=0.7)

# Setup plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

print("\nLet's see if the LLM makes sensible moves...\n")

# Run for a few steps
for i in range(10):
    # Get current state
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    
    print(f"Step {i}:")
    print(f"  Cube: {cube_pos}, Target: {target_pos}")
    
    # Get action from LLM
    action = agent.get_action(env)
    print(f"  LLM chose: {action}")
    
    # Take action
    obs, reward, done, info = env.step(action)
    
    # Visualize
    grid = env.get_grid_state()
    ax.clear()
    ax.imshow(np.flipud(grid), cmap=cmap, norm=norm)
    ax.set_title(f"Step {i}: LLM chose '{action}'")
    ax.set_xlabel(f"Cube: {cube_pos} → Target: {target_pos}")
    plt.draw()
    plt.pause(1.5)
    
    if done:
        if info['success']:
            print(f"\n✓✓✓ LLM reached the goal in {i+1} steps!")
        else:
            print(f"\n✗ Failed - ran out of steps")
        break

print(f"\nLLM Stats:")
print(f"  API calls made: {agent.num_calls}")
print(f"  Total tokens used: {agent.total_tokens}")

plt.close()
env.disconnect()