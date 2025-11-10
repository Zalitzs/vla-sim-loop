# FILE: scripts/debug_llm_decisions.py
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from envs.grid_bullet_world import GridBulletWorld
from agents.llm_agent import LLMAgent, grid_to_text

# Setup visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

print("="*60)
print("Debugging LLM Decisions")
print("="*60)

# Create simple environment
env = GridBulletWorld(gui=True, grid_size=10, dynamic_rules=[])
env.reset()

# Create LLM agent
agent = LLMAgent(model="gpt-4o-mini", temperature=0.7)

# Setup plot
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

print("\nWatching LLM make decisions...\n")

for i in range(15):
    # Get current state
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    grid = env.get_grid_state()
    grid_flipped = np.flipud(grid)
    
    print(f"\n{'='*60}")
    print(f"STEP {i}")
    print(f"{'='*60}")
    print(f"Cube: {cube_pos}, Target: {target_pos}")
    print(f"Distance: {np.abs(cube_pos[0]-target_pos[0]) + np.abs(cube_pos[1]-target_pos[1])}")
    
    # Show what LLM sees
    print("\nWhat LLM sees:")
    print(grid_to_text(grid_flipped))
    
    # Visualize both grids
    ax1.clear()
    ax1.imshow(grid_flipped, cmap=cmap, norm=norm)
    ax1.set_title("What LLM Sees (Flipped)\nforward=up, backward=down")
    ax1.set_xlabel(f"C at {cube_pos}, T at {target_pos}")
    
    ax2.clear()
    ax2.imshow(grid, cmap=cmap, norm=norm)
    ax2.set_title("Original Grid (Unflipped)\nforward=down, backward=up")
    
    plt.draw()
    plt.pause(0.5)
    
    # Get action
    action = agent.get_action(env, verbose=True)
    print(f"\nChose: {action}")
    
    # Take action
    obs, reward, done, info = env.step(action)
    
    new_pos = env.get_cube_grid_pos()
    dx = new_pos[0] - cube_pos[0]
    dy = new_pos[1] - cube_pos[1]
    
    print(f"Result: {cube_pos} → {new_pos} (dx={dx:+d}, dy={dy:+d})")
    
    # Pause for user
    input("Press Enter to continue...")
    
    if done:
        if info['success']:
            print(f"\n✓✓✓ SUCCESS in {i+1} steps!")
        else:
            print(f"\n✗ Failed")
        break

print(f"\nTotal API calls: {agent.num_calls}")
print(f"Total tokens: {agent.total_tokens}")

plt.close()
env.disconnect()