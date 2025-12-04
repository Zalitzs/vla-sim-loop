"""
COMPREHENSIVE LLM GUIDANCE COMPARISON
======================================
Includes:
- Success rate comparison
- Optimality analysis (successful episodes only)
- Maze difficulty analysis (which mazes each agent solves)
- Detailed statistics and insights
- Flagged mazes for inspection
- JSON export of results
- Statistical significance testing
- Visualization of flagged mazes
"""
import pickle
import numpy as np
import json
from datetime import datetime
from scipy import stats  # For statistical testing
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym
from llm_guidance import LLMGuidedAgent, simple_llm_query, real_llm_query, cached_llm_query
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
    
    return {
        'agent_pos': agent_pos,
        'target_pos': target_pos,
        'walls': walls
    }


def test_agent(model, test_mazes, use_llm=False, uncertainty_threshold=1.0, 
               llm_mode='simple', llm_boost=2.0, boost_type='multiplicative'):
    """
    Test agent on mazes with detailed tracking
    
    Returns:
        results: Dict with performance metrics and per-maze details
    """
    
    if use_llm:
        # Choose LLM query function
        if llm_mode == 'real':
            llm_fn = real_llm_query
            print(f"Testing WITH REAL LLM (GPT-4o-mini, threshold: {uncertainty_threshold})")
        elif llm_mode == 'cached':
            llm_fn = cached_llm_query
            print(f"Testing WITH CACHED LLM (GPT-4o-mini cached, threshold: {uncertainty_threshold})")
        else:
            llm_fn = simple_llm_query
            print(f"Testing WITH LLM guidance (simple heuristic, threshold: {uncertainty_threshold})")
        
        agent = LLMGuidedAgent(
            model, 
            uncertainty_threshold=uncertainty_threshold,
            llm_query_fn=llm_fn,
            llm_boost=llm_boost,
            boost_type=boost_type
        )
    else:
        agent = None
        print("Testing WITHOUT LLM guidance (baseline)")
    
    results = {
        'successes': 0,
        'steps': [],
        'success_flags': [],
        'llm_queries': 0,
        'total_steps': 0,
        'per_maze': []  # NEW: Store per-maze details
    }
    
    for i, maze_template in enumerate(test_mazes):
        # Create environment
        bullet_env = GridBulletWorld(
            gui=False,
            grid_size=maze_template['grid'].shape[0],
            max_steps=60,
            dynamic_rules=[]
        )
        env = MazeEnvGym(bullet_env)
        
        # Run episode
        obs, _ = env.reset(options={'maze_template': maze_template})
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
        results['success_flags'].append(success)
        results['total_steps'] += steps
        
        # Store per-maze details
        results['per_maze'].append({
            'maze_idx': i,
            'success': success,
            'steps': steps,
            'optimal_path': maze_template['path_length'],
            'wall_density': maze_template['wall_density']
        })
        
        env.close()
        
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{len(test_mazes)} mazes...")
    
    # Get LLM statistics if used
    if use_llm:
        llm_stats = agent.get_statistics()
        results['llm_queries'] = llm_stats['llm_queries']
        results['query_rate'] = llm_stats['query_rate']
        results['llm_stats'] = llm_stats
    
    return results


def calculate_statistical_significance(baseline_results, llm_results):
    """
    Perform statistical tests to determine if LLM improvement is significant
    
    Returns:
        dict with test results
    """
    # Chi-square test for success rates
    baseline_successes = baseline_results['successes']
    baseline_failures = len(baseline_results['success_flags']) - baseline_successes
    llm_successes = llm_results['successes']
    llm_failures = len(llm_results['success_flags']) - llm_successes
    
    # Contingency table
    contingency_table = [
        [baseline_successes, baseline_failures],
        [llm_successes, llm_failures]
    ]
    
    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Mann-Whitney U test for step counts (non-parametric)
    # Only compare successful episodes
    baseline_success_steps = [baseline_results['steps'][i] 
                             for i in range(len(baseline_results['steps'])) 
                             if baseline_results['success_flags'][i]]
    llm_success_steps = [llm_results['steps'][i] 
                        for i in range(len(llm_results['steps'])) 
                        if llm_results['success_flags'][i]]
    
    if baseline_success_steps and llm_success_steps:
        u_stat, p_value_mann = stats.mannwhitneyu(baseline_success_steps, llm_success_steps, 
                                                   alternative='two-sided')
    else:
        u_stat, p_value_mann = None, None
    
    return {
        'chi_square': {
            'statistic': chi2,
            'p_value': p_value_chi2,
            'significant': p_value_chi2 < 0.05,
            'interpretation': 'Success rates are significantly different' if p_value_chi2 < 0.05 
                            else 'Success rates are not significantly different'
        },
        'mann_whitney': {
            'statistic': u_stat,
            'p_value': p_value_mann,
            'significant': p_value_mann < 0.05 if p_value_mann else None,
            'interpretation': 'Step counts are significantly different' if (p_value_mann and p_value_mann < 0.05)
                            else 'Step counts are not significantly different' if p_value_mann
                            else 'Not enough data'
        }
    }


def analyze_maze_categories(baseline_results, llm_results, test_mazes):
    """Categorize mazes by which agent(s) solved them"""
    
    both_solved = []
    only_baseline = []
    only_llm = []
    neither = []
    
    for i in range(len(test_mazes)):
        baseline_success = baseline_results['success_flags'][i]
        llm_success = llm_results['success_flags'][i]
        
        if baseline_success and llm_success:
            both_solved.append(i)
        elif baseline_success and not llm_success:
            only_baseline.append(i)
        elif not baseline_success and llm_success:
            only_llm.append(i)
        else:
            neither.append(i)
    
    return {
        'both_solved': both_solved,
        'only_baseline': only_baseline,
        'only_llm': only_llm,
        'neither': neither
    }


def visualize_flagged_mazes(test_mazes, categories, output_dir='flagged_mazes'):
    """Create visualizations of flagged mazes"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize "Only LLM" mazes
    if categories['only_llm']:
        num_to_show = min(9, len(categories['only_llm']))
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, maze_idx in enumerate(categories['only_llm'][:num_to_show]):
            ax = axes[idx]
            maze = test_mazes[maze_idx]
            
            grid = maze['grid']
            start = maze['start_pos']
            goal = maze['target_pos']
            
            # Plot grid
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] == 1:
                        rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                                       facecolor='black', edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                    else:
                        rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                                       facecolor='white', edgecolor='lightgray', linewidth=0.5)
                        ax.add_patch(rect)
            
            # Mark start and goal
            ax.plot(start[0], start[1], 'go', markersize=15, markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgecolor='darkred', markeredgewidth=2)
            
            ax.set_xlim(-0.5, grid.shape[0]-0.5)
            ax.set_ylim(-0.5, grid.shape[1]-0.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Maze #{maze_idx+1} (Only LLM Solved)\n'
                        f'Walls: {maze["wall_density"]:.2f}', fontsize=10, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(num_to_show, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Mazes Only LLM Could Solve (Success Stories)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/only_llm_solved.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir}/only_llm_solved.png")
    
    # Visualize "Neither solved" mazes
    if categories['neither']:
        num_to_show = min(9, len(categories['neither']))
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, maze_idx in enumerate(categories['neither'][:num_to_show]):
            ax = axes[idx]
            maze = test_mazes[maze_idx]
            
            grid = maze['grid']
            start = maze['start_pos']
            goal = maze['target_pos']
            
            # Plot grid
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] == 1:
                        rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                                       facecolor='darkred', edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                    else:
                        rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                                       facecolor='white', edgecolor='lightgray', linewidth=0.5)
                        ax.add_patch(rect)
            
            # Mark start and goal
            ax.plot(start[0], start[1], 'go', markersize=15, markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgecolor='darkred', markeredgewidth=2)
            
            ax.set_xlim(-0.5, grid.shape[0]-0.5)
            ax.set_ylim(-0.5, grid.shape[1]-0.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Maze #{maze_idx+1} (UNSOLVED)\n'
                        f'Walls: {maze["wall_density"]:.2f}', fontsize=10, fontweight='bold', color='darkred')
        
        # Hide unused subplots
        for idx in range(num_to_show, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Hardest Mazes - Neither Agent Could Solve', fontsize=14, fontweight='bold', color='darkred')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neither_solved.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir}/neither_solved.png")


def calculate_optimality_metrics(results, test_mazes):
    """Calculate optimality gap for successful episodes"""
    
    optimal_lengths = []
    agent_lengths = []
    optimality_gaps = []
    optimality_ratios = []
    
    for i, maze in enumerate(test_mazes):
        # Only calculate for successful episodes
        if not results['success_flags'][i]:
            continue
        
        # Get A* optimal path
        _, optimal_len = a_star_pathfinding(
            maze['grid'],
            maze['start_pos'],
            maze['target_pos']
        )
        
        # Get agent's actual path length
        agent_len = results['steps'][i]
        
        if optimal_len != float('inf'):
            optimal_lengths.append(optimal_len)
            agent_lengths.append(agent_len)
            gap = agent_len - optimal_len
            ratio = agent_len / optimal_len if optimal_len > 0 else float('inf')
            
            optimality_gaps.append(gap)
            optimality_ratios.append(ratio)
    
    return {
        'avg_optimal_length': np.mean(optimal_lengths) if optimal_lengths else 0,
        'avg_agent_length': np.mean(agent_lengths) if agent_lengths else 0,
        'avg_optimality_gap': np.mean(optimality_gaps) if optimality_gaps else 0,
        'avg_optimality_ratio': np.mean(optimality_ratios) if optimality_ratios else 0,
        'num_analyzed': len(optimal_lengths),
    }


def print_comprehensive_results(baseline_results, llm_results, test_mazes, categories):
    """Print comprehensive comparison with all analyses"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("="*70)
    
    # 1. Success Rate
    baseline_success_rate = baseline_results['successes'] / len(test_mazes) * 100
    llm_success_rate = llm_results['successes'] / len(test_mazes) * 100
    
    print(f"\n{'='*70}")
    print("1. SUCCESS RATE COMPARISON")
    print(f"{'='*70}")
    print(f"  Baseline:     {baseline_success_rate:.1f}% ({baseline_results['successes']}/{len(test_mazes)})")
    print(f"  With LLM:     {llm_success_rate:.1f}% ({llm_results['successes']}/{len(test_mazes)})")
    print(f"  Improvement:  {llm_success_rate - baseline_success_rate:+.1f}%")
    
    # 2. Efficiency
    baseline_avg_steps = np.mean(baseline_results['steps'])
    llm_avg_steps = np.mean(llm_results['steps'])
    
    print(f"\n{'='*70}")
    print("2. EFFICIENCY (All Episodes)")
    print(f"{'='*70}")
    print(f"  Baseline:     {baseline_avg_steps:.1f} steps")
    print(f"  With LLM:     {llm_avg_steps:.1f} steps")
    print(f"  Difference:   {llm_avg_steps - baseline_avg_steps:+.1f} steps")
    
    # 3. LLM Usage
    if 'llm_stats' in llm_results:
        stats = llm_results['llm_stats']
        print(f"\n{'='*70}")
        print("3. LLM USAGE STATISTICS")
        print(f"{'='*70}")
        print(f"  Total queries:     {stats['llm_queries']}")
        print(f"  Query rate:        {stats['query_rate']*100:.1f}% of steps")
        print(f"  Uncertain steps:   {stats['uncertain_steps']} ({stats['uncertain_rate']*100:.1f}%)")
        print(f"  Times followed:    {stats['llm_followed']} ({stats['follow_rate']*100:.1f}%)")
        print(f"  Times ignored:     {stats['llm_ignored']}")
        print(f"\n  Hint Distribution:")
        for direction, count in stats['hint_distribution'].items():
            if count > 0:
                pct = count / stats['llm_queries'] * 100 if stats['llm_queries'] > 0 else 0
                print(f"    {direction:10s}: {count:3d} ({pct:5.1f}%)")
    
    # 4. Optimality Analysis
    baseline_opt = calculate_optimality_metrics(baseline_results, test_mazes)
    llm_opt = calculate_optimality_metrics(llm_results, test_mazes)
    
    print(f"\n{'='*70}")
    print("4. OPTIMALITY ANALYSIS (Successful Episodes Only)")
    print(f"{'='*70}")
    print(f"\n  Optimal Path (A*):")
    print(f"    Average length: {baseline_opt['avg_optimal_length']:.1f} steps")
    
    print(f"\n  Baseline Agent ({baseline_opt['num_analyzed']} successful):")
    print(f"    Average length:    {baseline_opt['avg_agent_length']:.1f} steps")
    print(f"    Optimality gap:    +{baseline_opt['avg_optimality_gap']:.1f} steps")
    print(f"    Optimality ratio:  {baseline_opt['avg_optimality_ratio']:.2f}x optimal")
    
    print(f"\n  With LLM ({llm_opt['num_analyzed']} successful):")
    print(f"    Average length:    {llm_opt['avg_agent_length']:.1f} steps")
    print(f"    Optimality gap:    +{llm_opt['avg_optimality_gap']:.1f} steps")
    print(f"    Optimality ratio:  {llm_opt['avg_optimality_ratio']:.2f}x optimal")
    
    gap_reduction = baseline_opt['avg_optimality_gap'] - llm_opt['avg_optimality_gap']
    ratio_improvement = baseline_opt['avg_optimality_ratio'] - llm_opt['avg_optimality_ratio']
    
    print(f"\n  LLM Impact:")
    print(f"    Gap reduction:     {gap_reduction:+.1f} steps")
    print(f"    Ratio improvement: {ratio_improvement:+.2f}x")
    
    # 5. Maze Category Analysis
    print(f"\n{'='*70}")
    print("5. MAZE CATEGORY ANALYSIS")
    print(f"{'='*70}")
    print(f"  Both solved:      {len(categories['both_solved'])} mazes")
    print(f"  Only baseline:    {len(categories['only_baseline'])} mazes")
    print(f"  Only LLM:         {len(categories['only_llm'])} mazes")
    print(f"  Neither:          {len(categories['neither'])} mazes")
    
    # Analyze difficulty by category
    if categories['both_solved']:
        both_walls = [test_mazes[i]['wall_density'] for i in categories['both_solved']]
        print(f"\n  Both solved - Wall density: {np.mean(both_walls):.3f} avg")
    
    if categories['only_llm']:
        only_llm_walls = [test_mazes[i]['wall_density'] for i in categories['only_llm']]
        print(f"  Only LLM    - Wall density: {np.mean(only_llm_walls):.3f} avg")
    
    if categories['only_baseline']:
        only_base_walls = [test_mazes[i]['wall_density'] for i in categories['only_baseline']]
        print(f"  Only baseline - Wall density: {np.mean(only_base_walls):.3f} avg")
    
    if categories['neither']:
        neither_walls = [test_mazes[i]['wall_density'] for i in categories['neither']]
        print(f"  Neither     - Wall density: {np.mean(neither_walls):.3f} avg")
    
    # 6. Flagged Mazes for Inspection
    print(f"\n{'='*70}")
    print("6. STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")
    
    sig_tests = calculate_statistical_significance(baseline_results, llm_results)
    
    print(f"\n  Chi-Square Test (Success Rates):")
    print(f"    χ² statistic: {sig_tests['chi_square']['statistic']:.4f}")
    print(f"    p-value:      {sig_tests['chi_square']['p_value']:.4f}")
    print(f"    Result:       {sig_tests['chi_square']['interpretation']}")
    if sig_tests['chi_square']['significant']:
        print(f"    → Success rate difference IS statistically significant (p < 0.05) ✓")
    else:
        print(f"    → Success rate difference is NOT statistically significant (p ≥ 0.05)")
    
    print(f"\n  Mann-Whitney U Test (Step Counts on Successful Episodes):")
    if sig_tests['mann_whitney']['statistic']:
        print(f"    U statistic:  {sig_tests['mann_whitney']['statistic']:.2f}")
        print(f"    p-value:      {sig_tests['mann_whitney']['p_value']:.4f}")
        print(f"    Result:       {sig_tests['mann_whitney']['interpretation']}")
        if sig_tests['mann_whitney']['significant']:
            print(f"    → Step count difference IS statistically significant (p < 0.05) ✓")
        else:
            print(f"    → Step count difference is NOT statistically significant (p ≥ 0.05)")
    else:
        print(f"    Not enough data for comparison")
    
    # 7. Flagged Mazes for Inspection
    print(f"\n{'='*70}")
    print("7. FLAGGED MAZES FOR INSPECTION")
    print(f"{'='*70}")
    
    # Flag: Only LLM solved (LLM success stories)
    if categories['only_llm']:
        print(f"\n  ONLY LLM SOLVED ({len(categories['only_llm'])} mazes):")
        print(f"  These mazes highlight where LLM guidance makes a difference")
        print(f"  {'─'*66}")
        for idx in categories['only_llm'][:10]:  # Show first 10
            maze = test_mazes[idx]
            llm_steps = llm_results['per_maze'][idx]['steps']
            print(f"    Maze #{idx+1:3d}: Path={maze['path_length']:2d} steps, "
                  f"Walls={maze['wall_density']:.2f}, "
                  f"LLM took {llm_steps} steps")
        if len(categories['only_llm']) > 10:
            print(f"    ... and {len(categories['only_llm']) - 10} more")
    
    # Flag: Neither solved (hardest mazes)
    if categories['neither']:
        print(f"\n  NEITHER SOLVED ({len(categories['neither'])} mazes):")
        print(f"  These are the hardest mazes - potential for future improvement")
        print(f"  {'─'*66}")
        for idx in categories['neither'][:10]:  # Show first 10
            maze = test_mazes[idx]
            print(f"    Maze #{idx+1:3d}: Path={maze['path_length']:2d} steps, "
                  f"Walls={maze['wall_density']:.2f} (UNSOLVED)")
        if len(categories['neither']) > 10:
            print(f"    ... and {len(categories['neither']) - 10} more")
    
    # Flag: Only baseline solved (LLM regressions)
    if categories['only_baseline']:
        print(f"\n  ONLY BASELINE SOLVED ({len(categories['only_baseline'])} mazes):")
        print(f"  These are cases where LLM guidance hurt performance")
        print(f"  {'─'*66}")
        for idx in categories['only_baseline']:
            maze = test_mazes[idx]
            baseline_steps = baseline_results['per_maze'][idx]['steps']
            print(f"    Maze #{idx+1:3d}: Path={maze['path_length']:2d} steps, "
                  f"Walls={maze['wall_density']:.2f}, "
                  f"Baseline took {baseline_steps} steps")
    
    # 7. Overall Assessment
    print(f"\n{'='*70}")
    print("8. OVERALL ASSESSMENT")
    print(f"{'='*70}")
    
    if llm_success_rate > baseline_success_rate:
        improvement_pct = ((llm_success_rate - baseline_success_rate) / baseline_success_rate) * 100
        print(f"\n  ✓ LLM GUIDANCE HELPED!")
        print(f"    Success rate improved by {improvement_pct:.1f}%")
        print(f"    ({len(categories['only_llm'])} additional mazes solved)")
        
        if llm_opt['avg_optimality_ratio'] > baseline_opt['avg_optimality_ratio']:
            print(f"\n    Trade-off: Paths are less optimal (+{(llm_opt['avg_optimality_ratio'] - baseline_opt['avg_optimality_ratio']):.2f}x)")
            print(f"    This suggests LLM encourages exploration to solve harder mazes")
        else:
            print(f"\n    Bonus: Paths are MORE optimal!")
            
    elif llm_success_rate == baseline_success_rate and llm_avg_steps < baseline_avg_steps:
        print(f"\n  ✓ LLM GUIDANCE HELPED!")
        print(f"    Same success rate but {baseline_avg_steps - llm_avg_steps:.1f} fewer steps")
    else:
        print(f"\n  ✗ LLM guidance did not improve performance")
        print(f"    Consider adjusting uncertainty threshold or improving LLM hints")
    
    print(f"\n{'='*70}")
    
    return {
        'baseline': baseline_results,
        'llm': llm_results,
        'categories': categories,
        'baseline_opt': baseline_opt,
        'llm_opt': llm_opt,
        'statistical_tests': sig_tests
    }


def compare_with_and_without_llm(model_path, dataset_file, uncertainty_threshold=1.0, 
                                llm_mode='simple', llm_boost=2.0, boost_type='multiplicative'):
    """Main comparison function"""
    
    print("="*70)
    print("COMPREHENSIVE LLM GUIDANCE EXPERIMENT")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_file}")
    print(f"Uncertainty threshold: {uncertainty_threshold}")
    print(f"LLM mode: {llm_mode}")
    print("="*70)
    
    # Load model and dataset
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    print(f"✓ Loaded {len(test_mazes)} test mazes\n")
    
    # Test baseline (no LLM)
    print("\n" + "─"*70)
    print("BASELINE (No LLM)")
    print("─"*70)
    baseline_results = test_agent(model, test_mazes, use_llm=False)
    
    # Test with LLM
    print("\n" + "─"*70)
    print("WITH LLM GUIDANCE")
    print("─"*70)
    llm_results = test_agent(model, test_mazes, use_llm=True, 
                            uncertainty_threshold=uncertainty_threshold,
                            llm_mode=llm_mode,
                            llm_boost=llm_boost,
                            boost_type=boost_type)
    
    # Analyze maze categories
    categories = analyze_maze_categories(baseline_results, llm_results, test_mazes)
    
    # Print comprehensive results
    all_results = print_comprehensive_results(baseline_results, llm_results, test_mazes, categories)
    
    # Save results to JSON
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"results_{llm_mode}_{timestamp}.json"
    
    # Prepare JSON-serializable data
    json_data = {
        'timestamp': timestamp,
        'model': model_path,
        'dataset': dataset_file,
        'llm_mode': llm_mode,
        'uncertainty_threshold': uncertainty_threshold,
        'summary': {
            'baseline_success_rate': baseline_results['successes'] / len(test_mazes),
            'llm_success_rate': llm_results['successes'] / len(test_mazes),
            'improvement': (llm_results['successes'] - baseline_results['successes']) / len(test_mazes),
            'baseline_avg_steps': float(np.mean(baseline_results['steps'])),
            'llm_avg_steps': float(np.mean(llm_results['steps'])),
        },
        'categories': {
            'both_solved': categories['both_solved'],
            'only_baseline': categories['only_baseline'],
            'only_llm': categories['only_llm'],
            'neither': categories['neither']
        },
        'optimality': {
            'baseline': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in all_results['baseline_opt'].items()},
            'llm': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in all_results['llm_opt'].items()}
        },
        'statistical_tests': {
            'chi_square': {
                'statistic': float(all_results['statistical_tests']['chi_square']['statistic']),
                'p_value': float(all_results['statistical_tests']['chi_square']['p_value']),
                'significant': all_results['statistical_tests']['chi_square']['significant'],
                'interpretation': all_results['statistical_tests']['chi_square']['interpretation']
            },
            'mann_whitney': {
                'statistic': float(all_results['statistical_tests']['mann_whitney']['statistic']) 
                           if all_results['statistical_tests']['mann_whitney']['statistic'] else None,
                'p_value': float(all_results['statistical_tests']['mann_whitney']['p_value']) 
                          if all_results['statistical_tests']['mann_whitney']['p_value'] else None,
                'significant': all_results['statistical_tests']['mann_whitney']['significant'],
                'interpretation': all_results['statistical_tests']['mann_whitney']['interpretation']
            }
        },
        'per_maze_baseline': baseline_results['per_maze'],
        'per_maze_llm': llm_results['per_maze']
    }
    
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ Saved results to: {json_filename}")
    
    # Visualize flagged mazes
    print(f"\nGenerating visualizations...")
    visualize_flagged_mazes(test_mazes, categories)
    
    print(f"\n{'='*70}")
    
    return baseline_results, llm_results, categories


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive LLM guidance comparison')
    parser.add_argument('--model', type=str, default='models/ppo_dataset_trained',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='maze_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Uncertainty threshold for LLM queries')
    parser.add_argument('--llm-mode', type=str, default='simple',
                       choices=['simple', 'real', 'cached'],
                       help='LLM mode: simple (heuristic), real (GPT-4o-mini), cached (GPT with cache)')
    parser.add_argument('--boost', type=float, default=2.0,
                       help='Boost strength (e.g., 2.0 for multiplicative, 0.2 for additive)')
    parser.add_argument('--boost-type', type=str, default='multiplicative',
                       choices=['multiplicative', 'additive'],
                       help='Boost type: multiplicative or additive')
    
    args = parser.parse_args()
    
    baseline, llm, categories = compare_with_and_without_llm(
        args.model, 
        args.dataset,
        uncertainty_threshold=args.threshold,
        llm_mode=args.llm_mode,
        llm_boost=args.boost,
        boost_type=args.boost_type
    )