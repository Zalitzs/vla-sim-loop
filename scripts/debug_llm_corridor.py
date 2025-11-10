# FILE: scripts/debug_llm_corridor.py
import sys
sys.path.append('..')

import numpy as np
from envs.grid_bullet_world import GridBulletWorld
from envs.maze_templates import get_maze_template
from agents.llm_agent import LLMAgent

print("Debugging LLM on Corridor")
print("="*60)

# Load corridor
template = get_maze_template('corridor')

# Create environment with GUI
env = GridBulletWorld(gui=True, grid_size=template['grid_size'])
env.reset(maze_template=template)

# Create LLM agent
agent = LLMAgent()

print("\nWatching LLM navigate corridor...")
print("Press Ctrl+C to stop\n")

for step in range(50):
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    
    print(f"\nStep {step+1}:")
    print(f"  Cube: {cube_pos}, Target: {target_pos}")
    
    # Get action (will print reasoning)
    action = agent.get_action(env, verbose=True)
    print(f"  Chose: {action}")
    
    # Take step
    obs, reward, done, info = env.step(action)
    
    if done:
        status = "SUCCESS" if info['success'] else "FAILED"
        print(f"\n{status} in {step+1} steps!")
        break
    
    import time
    time.sleep(1)  # Pause to watch

env.disconnect()