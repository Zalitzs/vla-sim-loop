# FILE: scripts/test_maze_templates.py
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from envs.grid_bullet_world import GridBulletWorld
from envs.maze_templates import get_maze_template
from agents.astar_agent import AStarAgent

# Setup visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

print("="*60)
print("Testing All Maze Templates with A*")
print("="*60)

# Get all maze names
maze_names = ['corridor', 'u_shape', 'narrow_gap', 'spiral','maze_simple', 'maze_hard']

for maze_name in maze_names:
    print(f"\n{'='*60}")
    print(f"MAZE: {maze_name.upper()}")
    print(f"{'='*60}")
    
    # Load template
    template = get_maze_template(maze_name)
    print(f"Description: {template['description']}")
    print(f"Grid size: {template['grid_size']}")
    print(f"Start: {template['start_pos']}, Target: {template['target_pos']}")
    
    # Create environment with this maze
    env = GridBulletWorld(gui=False, grid_size=template['grid_size'], dynamic_rules=[])
    env.reset(maze_template=template)
    
    # Create A* agent
    agent = AStarAgent()
    
    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{maze_name.upper()}: {template['description']}", fontsize=14, fontweight='bold')
    
    # Show initial state
    grid = env.get_grid_state()
    axes[0].imshow(np.flipud(grid), cmap=cmap, norm=norm)
    axes[0].set_title("Initial State")
    axes[0].set_xlabel(f"Start (blue): {template['start_pos']}, Target (red): {template['target_pos']}")
    
    # Run A* and collect path
    print("\nRunning A*...")
    path_grids = []
    steps = 0
    max_steps = 100
    
    for step in range(max_steps):
        action = agent.get_action(env)
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if done:
            if info['success']:
                print(f"✓ SUCCESS in {steps} steps!")
            else:
                print(f"✗ Failed (timeout)")
            break
    
    # Show final state
    final_grid = env.get_grid_state()
    axes[1].imshow(np.flipud(final_grid), cmap=cmap, norm=norm)
    axes[1].set_title(f"Final State - {steps} steps")
    
    if info.get('success'):
        axes[1].set_xlabel("✓ Goal Reached!", color='green', fontweight='bold')
    else:
        axes[1].set_xlabel("✗ Failed", color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(f'logs/maze_test_{maze_name}.png', dpi=100, bbox_inches='tight')
    print(f"Saved visualization to: logs/maze_test_{maze_name}.png")
    
    # Show figure and wait for user
    plt.show(block=False)
    input(f"\nPress Enter to continue to next maze...")
    plt.close()
    
    # Cleanup
    env.disconnect()
    
    print(f"\nMaze '{maze_name}' complete!")

print("\n" + "="*60)
print("All mazes tested!")
print("="*60)
print("\nCheck the logs/ folder for saved visualizations.")