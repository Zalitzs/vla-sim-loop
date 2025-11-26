"""
Debug script to see what the PPO agent is actually doing
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from envs.maze_templates import get_maze_template
from agents.gym_wrapper import MazeEnvGym


def visualize_maze(env):
    """Print the maze grid with agent and target"""
    grid = env.env.grid.copy()
    cx, cy = env.env.get_cube_grid_pos()
    tx, ty = env.env.get_target_grid_pos()
    
    print("\nMaze Layout:")
    print("  ", end="")
    for j in range(grid.shape[1]):
        print(f"{j:2}", end="")
    print()
    
    for i in range(grid.shape[0]):
        print(f"{i:2} ", end="")
        for j in range(grid.shape[1]):
            if (i, j) == (cx, cy) and (i, j) == (tx, ty):
                print("★ ", end="")  # Agent at goal
            elif (i, j) == (cx, cy):
                print("A ", end="")  # Agent
            elif (i, j) == (tx, ty):
                print("G ", end="")  # Goal
            elif grid[i, j] == 1:
                print("█ ", end="")  # Wall
            else:
                print(". ", end="")  # Empty
        print()
    print(f"\nAgent: ({cx}, {cy})  Goal: ({tx}, {ty})")


def debug_episode(model_path, maze_name='maze_simple', max_steps=20):
    """Run one episode with detailed output"""
    
    # Load maze
    template = get_maze_template(maze_name)
    bullet_env = GridBulletWorld(
        gui=False,
        grid_size=template['grid_size'],
        max_steps=50,
        dynamic_rules=[]
    )
    bullet_env.reset(maze_template=template)
    env = MazeEnvGym(bullet_env)
    env._maze_template = template
    
    # Load trained model
    model = PPO.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Reset environment
    obs, _ = env.reset(options={'maze_template': template})
    
    print("\n" + "="*50)
    print("INITIAL STATE")
    print("="*50)
    visualize_maze(env)
    
    action_names = {0: 'forward', 1: 'backward', 2: 'left', 3: 'right'}
    
    # Run episode
    for step in range(max_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        
        print(f"\n--- Step {step + 1} ---")
        print(f"Action: {action} ({action_names[action]})")
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.3f}")
        visualize_maze(env)
        
        if done:
            if info.get('success', False):
                print("\n🎉 SUCCESS! Agent reached the goal!")
            else:
                print("\n❌ Episode ended (timeout or stuck)")
            break
    
    env.close()


def test_random_agent(maze_name='maze_simple', n_episodes=10):
    """Compare: how does a random agent perform?"""
    print("\n" + "="*50)
    print("RANDOM AGENT BASELINE")
    print("="*50)
    
    template = get_maze_template(maze_name)
    bullet_env = GridBulletWorld(
        gui=False,
        grid_size=template['grid_size'],
        max_steps=50,
        dynamic_rules=[]
    )
    bullet_env.reset(maze_template=template)
    env = MazeEnvGym(bullet_env)
    env._maze_template = template
    
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(options={'maze_template': template})
        done = False
        steps = 0
        
        while not done:
            action = np.random.randint(0, 4)  # Random action
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        if info.get('success', False):
            successes += 1
            print(f"  Episode {ep+1}: SUCCESS in {steps} steps")
        else:
            print(f"  Episode {ep+1}: FAILED after {steps} steps")
    
    print(f"\nRandom agent success rate: {successes}/{n_episodes}")
    env.close()


if __name__ == "__main__":
    # Test 1: See what the trained agent does on maze_simple
    print("="*50)
    print("DEBUGGING PPO AGENT ON MAZE_SIMPLE")
    print("="*50)
    
    model_path = "../models/ppo_maze_simple"
    debug_episode(model_path, maze_name='maze_simple', max_steps=20)
    
    # Test 2: Try on corridor maze (generalization test!)
    print("\n" + "="*50)
    print("TESTING GENERALIZATION: CORRIDOR MAZE")
    print("="*50)
    debug_episode(model_path, maze_name='corridor', max_steps=20)
    
    # Test 3: Try on u_shape maze
    print("\n" + "="*50)
    print("TESTING GENERALIZATION: U-SHAPE MAZE")
    print("="*50)
    debug_episode(model_path, maze_name='u_shape', max_steps=20)
    
    # Test 4: Random baseline
    print("\n" + "="*50)
    print("RANDOM AGENT BASELINE")
    print("="*50)
    test_random_agent(maze_name='maze_simple', n_episodes=10)