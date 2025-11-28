"""
Comprehensive Generalization Test
Tests trained PPO agent on ALL maze templates
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


def test_maze(model, maze_name, n_episodes=5, max_steps=50, verbose=False):
    """Test agent on a specific maze"""
    
    # Load maze template
    template = get_maze_template(maze_name)
    
    # Create environment
    bullet_env = GridBulletWorld(
        gui=False,
        grid_size=template['grid_size'],
        max_steps=max_steps,
        dynamic_rules=[]
    )
    bullet_env.reset(maze_template=template)
    env = MazeEnvGym(bullet_env)
    
    # Test statistics
    successes = 0
    total_steps = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(options={'maze_template': template})
        done = False
        steps = 0
        episode_actions = []
        
        if verbose and ep == 0:
            print(f"\n{'='*60}")
            print(f"EPISODE 1 - DETAILED VIEW")
            print(f"{'='*60}")
            visualize_maze(env)
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            
            episode_actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
            
            if verbose and ep == 0 and steps <= 10:
                action_names = {0: 'forward', 1: 'backward', 2: 'left', 3: 'right'}
                print(f"\nStep {steps}: {action_names[action]} (reward: {reward:.2f})")
                visualize_maze(env)
        
        if info.get('success', False):
            successes += 1
        total_steps.append(steps)
    
    env.close()
    
    success_rate = successes / n_episodes * 100
    avg_steps = np.mean(total_steps)
    std_steps = np.std(total_steps)
    
    return {
        'maze_name': maze_name,
        'success_rate': success_rate,
        'successes': successes,
        'total_episodes': n_episodes,
        'avg_steps': avg_steps,
        'std_steps': std_steps,
        'description': template['description']
    }


def run_all_tests(model_path, verbose=False):
    """Run comprehensive tests on all maze templates"""
    
    # Load trained model
    print("="*60)
    print("COMPREHENSIVE GENERALIZATION TEST")
    print("="*60)
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # All maze templates
    maze_names = ['corridor', 'u_shape', 'narrow_gap', 'spiral', 'maze_simple', 'maze_hard', 'maze_hard_2']
    
    results = []
    
    for maze_name in maze_names:
        print(f"\n{'='*60}")
        print(f"Testing: {maze_name.upper()}")
        print(f"{'='*60}")
        
        try:
            result = test_maze(model, maze_name, n_episodes=10, verbose=verbose)
            results.append(result)
            
            # Print result
            status = "✓ PASS" if result['success_rate'] >= 80 else "✗ FAIL"
            print(f"\n{status}")
            print(f"  Description: {result['description']}")
            print(f"  Success Rate: {result['success_rate']:.1f}% ({result['successes']}/{result['total_episodes']})")
            print(f"  Avg Steps: {result['avg_steps']:.1f} ± {result['std_steps']:.1f}")
            
        except Exception as e:
            print(f"✗ ERROR testing {maze_name}: {e}")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: GENERALIZATION RESULTS")
    print("="*60)
    print(f"{'Maze':<15} {'Success Rate':<15} {'Avg Steps':<15} {'Status':<10}")
    print("-"*60)
    
    for result in results:
        status = "✓ PASS" if result['success_rate'] >= 80 else "✗ FAIL"
        print(f"{result['maze_name']:<15} {result['success_rate']:>6.1f}% {' '*8} {result['avg_steps']:>6.1f} {' '*8} {status}")
    
    # Overall stats
    overall_success = np.mean([r['success_rate'] for r in results])
    passing_mazes = sum(1 for r in results if r['success_rate'] >= 80)
    
    print("-"*60)
    print(f"Overall Success Rate: {overall_success:.1f}%")
    print(f"Passing Mazes (≥80%): {passing_mazes}/{len(results)}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PPO agent generalization')
    parser.add_argument('--model', type=str, default='../models/ppo_maze_simple',
                       help='Path to trained model')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed episode walkthrough for first episode')
    parser.add_argument('--maze', type=str, default=None,
                       help='Test specific maze only (e.g., maze_hard)')
    
    args = parser.parse_args()
    
    if args.maze:
        # Test single maze with detailed output
        print(f"Testing single maze: {args.maze}")
        model = PPO.load(args.model)
        result = test_maze(model, args.maze, n_episodes=10, verbose=True)
        print(f"\nFinal Result:")
        print(f"  Success Rate: {result['success_rate']:.1f}%")
        print(f"  Avg Steps: {result['avg_steps']:.1f} ± {result['std_steps']:.1f}")
    else:
        # Test all mazes
        run_all_tests(args.model, verbose=args.verbose)