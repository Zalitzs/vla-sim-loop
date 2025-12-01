"""
Debug: Check what steps are being recorded during agent execution
"""
import sys
sys.path.insert(0, '..')

import pickle
import numpy as np
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym

# Load model and dataset
model = PPO.load('models/ppo_dataset_trained')
with open('maze_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    test_mazes = data['test_mazes']

print("="*60)
print("DEBUGGING STEP COUNTING")
print("="*60)

# Test first maze manually
maze = test_mazes[0]

print(f"\nMaze 1:")
print(f"Start: {maze['start_pos']}")
print(f"Goal: {maze['target_pos']}")
print(f"Optimal: 18 steps")

# Create environment and wrapper
bullet_env = GridBulletWorld(
    gui=False,
    grid_size=10,
    max_steps=60,
    dynamic_rules=[]
)
env = MazeEnvGym(bullet_env)

# Run episode and track carefully
obs, _ = env.reset(options={'maze_template': maze})  # ✅ Fixed!
done = False
steps = 0
positions = [bullet_env.get_cube_grid_pos()]

print(f"\nStarting position: {positions[0]}")

while not done and steps < 60:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
    
    obs, reward, done, truncated, info = env.step(action)
    steps += 1
    
    current_pos = bullet_env.get_cube_grid_pos()
    positions.append(current_pos)
    
    if steps <= 5 or done:
        print(f"Step {steps}: moved to {current_pos}, reward={reward:.2f}, done={done}")
    
    done = done or truncated

env.close()

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"{'='*60}")
print(f"Total steps recorded: {steps}")
print(f"Number of positions: {len(positions)}")
print(f"Reached goal: {info.get('success', False)}")
print(f"Final position: {positions[-1]}")
print(f"Goal position: {maze['target_pos']}")

if steps <= 20:
    print(f"\nFull path:")
    for i, pos in enumerate(positions):
        print(f"  Position {i}: {pos}")