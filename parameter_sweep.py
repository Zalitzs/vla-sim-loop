"""
COMPREHENSIVE PARAMETER SWEEP
==============================
Tests all combinations of:
- Uncertainty thresholds: [0.6, 0.8, 1.0, 1.2, 1.4]
- Multiplicative boosts: [1.5, 2.0, 3.0, 5.0]
- Additive boosts: [0.1, 0.2, 0.3]
+ 1 baseline run

Total: 36 runs (~3-4 hours overnight)
Outputs: CSV with all results + visualization plots
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym
from llm_guidance import LLMGuidedAgent, real_llm_query
from optimality_analysis import a_star_pathfinding


def get_env_state(env):
    """Extract environment state for LLM"""
    bullet_env = env.env
    agent_pos = bullet_env.get_cube_grid_pos()
    target_pos = bullet_env.get_target_grid_pos()
    
    walls = {
        'forward': bullet_env.is_wall_at_grid(agent_pos[0] + 1, agent_pos[1]),
        'backward': bullet_env.is_wall_at_grid(agent_pos[0] - 1, agent_pos[1]),
        'left': bullet_env.is_wall_at_grid(agent_pos[0], agent_pos[1] - 1),
        'right': bullet_env.is_wall_at_grid(agent_pos[0], agent_pos[1] + 1),
    }
    
    return {'agent_pos': agent_pos, 'target_pos': target_pos, 'walls': walls}


def test_configuration(model, test_mazes, threshold=None, boost=None, boost_type=None, debug=False):
    """
    Test a single configuration
    
    Args:
        threshold: Uncertainty threshold (None for baseline)
        boost: Boost strength
        boost_type: 'multiplicative' or 'additive'
        debug: If True, only test on first 10 mazes
    
    Returns:
        dict with results
    """
    use_llm = threshold is not None
    
    # Debug mode: only use first 10 mazes
    mazes_to_test = test_mazes[:10] if debug else test_mazes
    
    if use_llm:
        agent = LLMGuidedAgent(
            model,
            uncertainty_threshold=threshold,
            llm_query_fn=real_llm_query,
            llm_boost=boost,
            boost_type=boost_type
        )
    
    successes = 0
    steps_list = []
    success_flags = []
    llm_query_count = 0
    
    for i, maze_template in enumerate(mazes_to_test):
        bullet_env = GridBulletWorld(
            gui=False,
            grid_size=maze_template['grid'].shape[0],
            max_steps=60,
            dynamic_rules=[]
        )
        env = MazeEnvGym(bullet_env)
        
        obs, _ = env.reset(options={'maze_template': maze_template})
        done = False
        steps = 0
        
        while not done and steps < 60:
            if use_llm:
                env_state = get_env_state(env)
                action, was_uncertain = agent.predict(obs, env_state)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
        
        success = info.get('success', False)
        if success:
            successes += 1
        steps_list.append(steps)
        success_flags.append(success)
        
        env.close()
    
    # Calculate metrics
    success_rate = successes / len(mazes_to_test) * 100
    avg_steps = np.mean(steps_list)
    
    # Calculate optimality for successful episodes
    successful_steps = [steps_list[i] for i in range(len(steps_list)) if success_flags[i]]
    avg_successful_steps = np.mean(successful_steps) if successful_steps else 0
    
    # Get LLM stats
    if use_llm:
        llm_stats = agent.get_statistics()
        query_rate = llm_stats['query_rate']
        follow_rate = llm_stats['follow_rate']
    else:
        query_rate = 0
        follow_rate = 0
    
    return {
        'threshold': threshold if threshold else 'baseline',
        'boost': boost if boost else 'N/A',
        'boost_type': boost_type if boost_type else 'N/A',
        'success_rate': success_rate,
        'successes': successes,
        'avg_steps_all': avg_steps,
        'avg_steps_successful': avg_successful_steps,
        'query_rate': query_rate * 100 if use_llm else 0,
        'follow_rate': follow_rate * 100 if use_llm else 0
    }


def run_parameter_sweep(model_path, dataset_file, debug=False):
    """Run comprehensive parameter sweep"""
    
    print("="*80)
    print("COMPREHENSIVE PARAMETER SWEEP")
    if debug:
        print("🐛 DEBUG MODE: Testing on 10 mazes only")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_file}")
    print("="*80)
    
    # Load model and dataset
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    
    if debug:
        print(f"✓ Loaded {len(test_mazes)} test mazes (using first 10 for debug)\n")
    else:
        print(f"✓ Loaded {len(test_mazes)} test mazes\n")
    
    # Define parameter grid
    thresholds = [0.6, 0.8, 1.0, 1.2, 1.4]
    multiplicative_boosts = [1.5, 2.0, 3.0, 5.0]
    additive_boosts = [0.1, 0.2, 0.3]
    
    results = []
    
    # Run baseline
    print("\n" + "="*80)
    print(f"Running BASELINE (1/36)")
    print("="*80)
    baseline_result = test_configuration(model, test_mazes, debug=debug)
    results.append(baseline_result)
    print(f"  Success Rate: {baseline_result['success_rate']:.1f}%")
    
    # Run multiplicative experiments
    run_num = 2
    total_runs = 1 + len(thresholds) * len(multiplicative_boosts) + len(thresholds) * len(additive_boosts)
    
    for threshold in thresholds:
        for boost in multiplicative_boosts:
            print("\n" + "="*80)
            print(f"Running MULTIPLICATIVE ({run_num}/{total_runs})")
            print(f"  Threshold: {threshold}, Boost: {boost}x")
            print("="*80)
            
            result = test_configuration(
                model, test_mazes,
                threshold=threshold,
                boost=boost,
                boost_type='multiplicative',
                debug=debug
            )
            results.append(result)
            
            print(f"  Success Rate: {result['success_rate']:.1f}%")
            print(f"  Query Rate: {result['query_rate']:.1f}%")
            print(f"  Follow Rate: {result['follow_rate']:.1f}%")
            
            run_num += 1
    
    # Run additive experiments
    for threshold in thresholds:
        for boost in additive_boosts:
            print("\n" + "="*80)
            print(f"Running ADDITIVE ({run_num}/{total_runs})")
            print(f"  Threshold: {threshold}, Boost: +{boost}")
            print("="*80)
            
            result = test_configuration(
                model, test_mazes,
                threshold=threshold,
                boost=boost,
                boost_type='additive',
                debug=debug
            )
            results.append(result)
            
            print(f"  Success Rate: {result['success_rate']:.1f}%")
            print(f"  Query Rate: {result['query_rate']:.1f}%")
            print(f"  Follow Rate: {result['follow_rate']:.1f}%")
            
            run_num += 1
    
    return results, baseline_result


def save_and_visualize_results(results, baseline_result):
    """Save results to CSV and create visualizations"""
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"parameter_sweep_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Saved results to: {csv_filename}")
    
    # Separate multiplicative and additive
    df_mult = df[df['boost_type'] == 'multiplicative'].copy()
    df_add = df[df['boost_type'] == 'additive'].copy()
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Multiplicative - Threshold vs Success Rate
    ax1 = plt.subplot(2, 3, 1)
    for boost in df_mult['boost'].unique():
        data = df_mult[df_mult['boost'] == boost]
        ax1.plot(data['threshold'], data['success_rate'], 'o-', 
                label=f'{boost}x', linewidth=2, markersize=8)
    ax1.axhline(y=baseline_result['success_rate'], color='red', 
                linestyle='--', label='Baseline', linewidth=2)
    ax1.set_xlabel('Uncertainty Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Multiplicative: Threshold Sensitivity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multiplicative - Boost vs Success Rate
    ax2 = plt.subplot(2, 3, 2)
    for threshold in df_mult['threshold'].unique():
        data = df_mult[df_mult['threshold'] == threshold]
        ax2.plot(data['boost'], data['success_rate'], 's-',
                label=f'Thresh={threshold}', linewidth=2, markersize=8)
    ax2.axhline(y=baseline_result['success_rate'], color='red',
                linestyle='--', label='Baseline', linewidth=2)
    ax2.set_xlabel('Boost Multiplier', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Multiplicative: Boost Strength', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Multiplicative - Heat Map
    ax3 = plt.subplot(2, 3, 3)
    pivot_mult = df_mult.pivot(index='boost', columns='threshold', values='success_rate')
    sns.heatmap(pivot_mult, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax3,
                cbar_kws={'label': 'Success Rate (%)'}, vmin=70, vmax=95)
    ax3.set_xlabel('Uncertainty Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Boost Multiplier', fontsize=12, fontweight='bold')
    ax3.set_title('Multiplicative: Heat Map', fontsize=14, fontweight='bold')
    
    # Plot 4: Additive - Threshold vs Success Rate
    ax4 = plt.subplot(2, 3, 4)
    for boost in df_add['boost'].unique():
        data = df_add[df_add['boost'] == boost]
        ax4.plot(data['threshold'], data['success_rate'], 'o-',
                label=f'+{boost}', linewidth=2, markersize=8)
    ax4.axhline(y=baseline_result['success_rate'], color='red',
                linestyle='--', label='Baseline', linewidth=2)
    ax4.set_xlabel('Uncertainty Threshold', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Additive: Threshold Sensitivity', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Additive - Boost vs Success Rate
    ax5 = plt.subplot(2, 3, 5)
    for threshold in df_add['threshold'].unique():
        data = df_add[df_add['threshold'] == threshold]
        ax5.plot(data['boost'], data['success_rate'], 's-',
                label=f'Thresh={threshold}', linewidth=2, markersize=8)
    ax5.axhline(y=baseline_result['success_rate'], color='red',
                linestyle='--', label='Baseline', linewidth=2)
    ax5.set_xlabel('Additive Boost', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Additive: Boost Strength', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Additive - Heat Map
    ax6 = plt.subplot(2, 3, 6)
    pivot_add = df_add.pivot(index='boost', columns='threshold', values='success_rate')
    sns.heatmap(pivot_add, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax6,
                cbar_kws={'label': 'Success Rate (%)'}, vmin=70, vmax=95)
    ax6.set_xlabel('Uncertainty Threshold', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Additive Boost', fontsize=12, fontweight='bold')
    ax6.set_title('Additive: Heat Map', fontsize=14, fontweight='bold')
    
    plt.suptitle('Comprehensive Parameter Sweep Results', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_filename = f"parameter_sweep_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plots to: {plot_filename}")
    
    # Print best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    
    best_mult = df_mult.loc[df_mult['success_rate'].idxmax()]
    best_add = df_add.loc[df_add['success_rate'].idxmax()]
    
    print(f"\nBest Multiplicative:")
    print(f"  Threshold: {best_mult['threshold']}")
    print(f"  Boost: {best_mult['boost']}x")
    print(f"  Success Rate: {best_mult['success_rate']:.1f}%")
    print(f"  Query Rate: {best_mult['query_rate']:.1f}%")
    
    print(f"\nBest Additive:")
    print(f"  Threshold: {best_add['threshold']}")
    print(f"  Boost: +{best_add['boost']}")
    print(f"  Success Rate: {best_add['success_rate']:.1f}%")
    print(f"  Query Rate: {best_add['query_rate']:.1f}%")
    
    print(f"\nBaseline:")
    print(f"  Success Rate: {baseline_result['success_rate']:.1f}%")
    
    improvement_mult = best_mult['success_rate'] - baseline_result['success_rate']
    improvement_add = best_add['success_rate'] - baseline_result['success_rate']
    
    print(f"\nBest Improvement:")
    if improvement_mult > improvement_add:
        print(f"  Multiplicative: +{improvement_mult:.1f}%")
        print(f"  → Use threshold={best_mult['threshold']}, boost={best_mult['boost']}x")
    else:
        print(f"  Additive: +{improvement_add:.1f}%")
        print(f"  → Use threshold={best_add['threshold']}, boost=+{best_add['boost']}")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive parameter sweep')
    parser.add_argument('--model', type=str, default='models/ppo_dataset_trained',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='maze_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: test on only 10 mazes (~1 hour total)')
    
    args = parser.parse_args()
    
    if args.debug:
        print("\n🐛 DEBUG MODE ENABLED")
        print("   Testing on 10 mazes per configuration")
        print("   Estimated time: ~1 hour for all 36 runs")
        print("   Use this to verify everything works before full run\n")
    else:
        print("\n🚀 Starting comprehensive parameter sweep...")
        print("   Testing on 150 mazes per configuration")
        print("   This will take ~3-4 hours")
        print("   Results will be saved automatically\n")
    
    results, baseline = run_parameter_sweep(args.model, args.dataset, debug=args.debug)
    save_and_visualize_results(results, baseline)
    
    print("\n✅ SWEEP COMPLETE!")
    print("   Check CSV file for raw data")
    print("   Check PNG file for visualizations")
