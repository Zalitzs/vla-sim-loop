"""
Compare LLM-Guided Agent vs Baseline

Tests if LLM guidance improves success rate and efficiency
"""
import sys
sys.path.insert(0, '..')

import pickle
import numpy as np
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym
from llm_guidance import LLMGuidedAgent, simple_llm_query


def get_env_state(env):
    """Extract state info for LLM context"""
    # Access the underlying GridBulletWorld
    bullet_env = env.env
    
    # Get positions using the correct methods
    agent_pos = bullet_env.get_cube_grid_pos()  # Returns (x, y) tuple
    target_pos = bullet_env.get_target_grid_pos()  # Returns (x, y) tuple
    
    # Get walls around agent
    x, y = agent_pos
    grid = bullet_env.grid
    grid_size = grid.shape[0]
    
    # Check walls in 4 directions
    walls = {
        'forward': grid[x+1, y] == 1 if x+1 < grid_size else True,
        'backward': grid[x-1, y] == 1 if x-1 >= 0 else True,
        'right': grid[x, y+1] == 1 if y+1 < grid_size else True,
        'left': grid[x, y-1] == 1 if y-1 >= 0 else True,
    }
    
    return {
        'agent_pos': agent_pos,
        'target_pos': target_pos,
        'walls': walls
    }


def test_agent(model, test_mazes, use_llm=False, uncertainty_threshold=1.0):
    """
    Test agent on mazes with or without LLM guidance
    
    Args:
        model: Trained PPO model
        test_mazes: List of maze templates
        use_llm: Whether to use LLM guidance
        uncertainty_threshold: Entropy threshold for LLM queries
    
    Returns:
        results: Dict with performance metrics
    """
    
    if use_llm:
        agent = LLMGuidedAgent(
            model, 
            uncertainty_threshold=uncertainty_threshold,
            llm_query_fn=simple_llm_query
        )
        print(f"Testing WITH LLM guidance (uncertainty threshold: {uncertainty_threshold})")
    else:
        agent = None
        print("Testing WITHOUT LLM guidance (baseline)")
    
    results = {
        'successes': 0,
        'steps': [],
        'llm_queries': 0,
        'total_steps': 0
    }
    
    for i, maze_template in enumerate(test_mazes):
        # Create environment
        bullet_env = GridBulletWorld(
            gui=False,
            grid_size=maze_template['grid'].shape[0],
            max_steps=60,
            dynamic_rules=[]
        )
        bullet_env.reset(maze_template=maze_template)
        env = MazeEnvGym(bullet_env)
        
        # Run episode
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 60:
            if use_llm:
                # Get environment state for LLM
                env_state = get_env_state(env)
                action, was_uncertain = agent.predict(obs, env_state)
            else:
                # Baseline: just use model
                action, _ = model.predict(obs, deterministic=True)
                action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
        
        # Record results
        success = info.get('success', False)
        if success:
            results['successes'] += 1
        results['steps'].append(steps)
        results['total_steps'] += steps
        
        env.close()
        
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{len(test_mazes)} mazes...")
    
    # Get LLM statistics if used
    if use_llm:
        llm_stats = agent.get_statistics()
        results['llm_queries'] = llm_stats['llm_queries']
        results['query_rate'] = llm_stats['query_rate']
    
    return results


def compare_with_and_without_llm(model_path, dataset_file, uncertainty_threshold=1.0):
    """
    Run full comparison experiment
    """
    print("="*60)
    print("LLM GUIDANCE EXPERIMENT")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_file}")
    print(f"Uncertainty threshold: {uncertainty_threshold}")
    print("="*60)
    
    # Load model and dataset
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    print(f"✓ Loaded {len(test_mazes)} test mazes\n")
    
    # Test baseline (no LLM)
    print("\n" + "─"*60)
    print("BASELINE (No LLM)")
    print("─"*60)
    baseline_results = test_agent(model, test_mazes, use_llm=False)
    
    # Test with LLM
    print("\n" + "─"*60)
    print("WITH LLM GUIDANCE")
    print("─"*60)
    llm_results = test_agent(model, test_mazes, use_llm=True, 
                            uncertainty_threshold=uncertainty_threshold)
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    baseline_success_rate = baseline_results['successes'] / len(test_mazes) * 100
    llm_success_rate = llm_results['successes'] / len(test_mazes) * 100
    
    baseline_avg_steps = np.mean(baseline_results['steps'])
    llm_avg_steps = np.mean(llm_results['steps'])
    
    print(f"\nSuccess Rate:")
    print(f"  Baseline:     {baseline_success_rate:.1f}% ({baseline_results['successes']}/{len(test_mazes)})")
    print(f"  With LLM:     {llm_success_rate:.1f}% ({llm_results['successes']}/{len(test_mazes)})")
    print(f"  Improvement:  {llm_success_rate - baseline_success_rate:+.1f}%")
    
    print(f"\nAverage Steps:")
    print(f"  Baseline:     {baseline_avg_steps:.1f}")
    print(f"  With LLM:     {llm_avg_steps:.1f}")
    print(f"  Difference:   {llm_avg_steps - baseline_avg_steps:+.1f}")
    
    print(f"\nLLM Usage:")
    print(f"  Total queries:  {llm_results['llm_queries']}")
    print(f"  Query rate:     {llm_results['query_rate']*100:.1f}% of steps")
    
    # Determine if LLM helped
    print("\n" + "─"*60)
    if llm_success_rate > baseline_success_rate:
        print("✓ LLM GUIDANCE HELPED! Success rate improved.")
    elif llm_success_rate == baseline_success_rate and llm_avg_steps < baseline_avg_steps:
        print("✓ LLM GUIDANCE HELPED! Same success but fewer steps.")
    else:
        print("✗ LLM guidance did not improve performance.")
        print("  Try adjusting uncertainty threshold or improving LLM hints.")
    print("─"*60)
    
    return baseline_results, llm_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare LLM-guided vs baseline agent')
    parser.add_argument('--model', type=str, default='models/ppo_dataset_trained',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='maze_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Uncertainty threshold for LLM queries')
    
    args = parser.parse_args()
    
    baseline, llm = compare_with_and_without_llm(
        args.model, 
        args.dataset,
        uncertainty_threshold=args.threshold
    )