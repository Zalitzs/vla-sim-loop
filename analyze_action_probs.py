"""
Action Probability Analysis for LLM Queries

Captures and visualizes action probability distributions at LLM query points
for mazes where LLM succeeded but baseline failed.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym
from llm_guidance import LLMGuidedAgent, real_llm_query
import torch


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


class ActionProbRecorder:
    """Records action probabilities at LLM query points"""
    
    def __init__(self, model):
        self.model = model
        self.records = []
    
    def predict_with_probs(self, obs):
        """Get action and full probability distribution"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            # Get policy distribution
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            logits = self.model.policy.action_net(latent_pi)
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        # Get deterministic action
        action = np.argmax(probs)
        
        return action, probs, entropy
    
    def record_query(self, obs, probs, entropy, position, llm_hint, action_taken):
        """Record a single LLM query"""
        self.records.append({
            'position': position,
            'probs': probs.copy(),
            'entropy': entropy,
            'llm_hint': llm_hint,
            'action_taken': action_taken
        })


def test_maze_with_recording(model, maze_template, use_llm=False, threshold=1.0, 
                            llm_boost=2.0, boost_type='multiplicative'):
    """
    Test a single maze and record action probabilities at LLM queries
    
    Returns:
        success, steps, records (if use_llm)
    """
    bullet_env = GridBulletWorld(gui=False, grid_size=10, max_steps=60, dynamic_rules=[])
    env = MazeEnvGym(bullet_env)
    
    if use_llm:
        recorder = ActionProbRecorder(model)
        agent = LLMGuidedAgent(model, uncertainty_threshold=threshold, 
                              llm_query_fn=real_llm_query,
                              llm_boost=llm_boost,
                              boost_type=boost_type)
    
    obs, _ = env.reset(options={'maze_template': maze_template})
    done = False
    steps = 0
    
    while not done and steps < 60:
        if use_llm:
            # Get action probabilities
            action_model, probs, entropy = recorder.predict_with_probs(obs)
            
            # Check if LLM should be queried
            env_state = get_env_state(env)
            
            if entropy > threshold:
                # LLM is queried
                llm_hint = agent.llm_query_fn(env_state)
                
                # Get boosted probabilities
                action, _ = agent.predict(obs, env_state)
                
                # Record this query
                position = bullet_env.get_cube_grid_pos()
                recorder.record_query(obs, probs, entropy, position, llm_hint, action)
            else:
                action = action_model
            
            action = int(action)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        done = done or truncated
    
    success = info.get('success', False)
    env.close()
    
    if use_llm:
        return success, steps, recorder.records
    else:
        return success, steps, None


def find_llm_success_mazes(model, test_mazes, threshold=1.0, max_mazes=5):
    """
    Find mazes where LLM succeeded but baseline failed
    
    Returns:
        List of (maze_idx, maze, llm_records)
    """
    print("="*70)
    print("FINDING LLM SUCCESS CASES")
    print("="*70)
    
    llm_success_cases = []
    
    for i, maze in enumerate(test_mazes):
        print(f"\rTesting maze {i+1}/{len(test_mazes)}...", end="")
        
        # Test baseline
        baseline_success, _, _ = test_maze_with_recording(model, maze, use_llm=False)
        
        # Only test with LLM if baseline failed
        if not baseline_success:
            llm_success, steps, records = test_maze_with_recording(
                model, maze, use_llm=True, threshold=threshold
            )
            
            if llm_success:
                llm_success_cases.append((i, maze, records))
                print(f"\n  ✓ Found LLM success case: Maze #{i+1} ({len(records)} LLM queries)")
                
                if len(llm_success_cases) >= max_mazes:
                    break
    
    print(f"\n\n✓ Found {len(llm_success_cases)} mazes where LLM succeeded but baseline failed")
    return llm_success_cases


def visualize_action_probs_for_maze(maze_idx, maze, records, output_file):
    """
    Visualize all LLM queries for a single maze
    """
    action_names = ['Forward', 'Backward', 'Left', 'Right']
    num_queries = len(records)
    
    if num_queries == 0:
        print(f"No LLM queries for maze {maze_idx}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, (num_queries + 1) // 2, figsize=(5 * ((num_queries + 1) // 2), 10))
    if num_queries == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, record in enumerate(records):
        ax = axes[idx]
        
        probs = record['probs']
        entropy = record['entropy']
        position = record['position']
        llm_hint = record['llm_hint']
        action_taken = record['action_taken']
        
        # Bar plot of probabilities
        colors = ['lightblue'] * 4
        colors[action_taken] = 'orange'  # Highlight chosen action
        
        bars = ax.bar(action_names, probs, color=colors, edgecolor='black', linewidth=1.5)
        
        # Mark LLM suggestion
        llm_action_map = {'forward': 0, 'backward': 1, 'left': 2, 'right': 3}
        if llm_hint in llm_action_map:
            llm_idx = llm_action_map[llm_hint]
            bars[llm_idx].set_edgecolor('green')
            bars[llm_idx].set_linewidth(3)
        
        # Labels
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_title(f'Query {idx+1} at {position}\n'
                    f'Entropy: {entropy:.3f}\n'
                    f'LLM suggests: {llm_hint.upper()}',
                    fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for idx in range(num_queries, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Maze #{maze_idx+1}: Action Probabilities at LLM Query Points\n'
                f'Orange = Action Taken, Green Border = LLM Suggestion',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: {output_file}")


def analyze_uncertainty_patterns(llm_success_cases):
    """Analyze patterns in action probabilities"""
    
    print("\n" + "="*70)
    print("UNCERTAINTY PATTERN ANALYSIS")
    print("="*70)
    
    all_entropies = []
    all_max_probs = []
    llm_hints = {'forward': 0, 'backward': 0, 'left': 0, 'right': 0}
    
    for maze_idx, maze, records in llm_success_cases:
        for record in records:
            all_entropies.append(record['entropy'])
            all_max_probs.append(np.max(record['probs']))
            if record['llm_hint'] in llm_hints:
                llm_hints[record['llm_hint']] += 1
    
    print(f"\nTotal LLM queries across {len(llm_success_cases)} mazes: {len(all_entropies)}")
    print(f"\nEntropy at query points:")
    print(f"  Mean:   {np.mean(all_entropies):.3f}")
    print(f"  Median: {np.median(all_entropies):.3f}")
    print(f"  Min:    {np.min(all_entropies):.3f}")
    print(f"  Max:    {np.max(all_entropies):.3f}")
    
    print(f"\nMax probability at query points:")
    print(f"  Mean:   {np.mean(all_max_probs):.3f}")
    print(f"  Median: {np.median(all_max_probs):.3f}")
    
    print(f"\nLLM hint distribution:")
    for direction, count in llm_hints.items():
        pct = count / len(all_entropies) * 100
        print(f"  {direction:10s}: {count:3d} ({pct:5.1f}%)")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze action probabilities for specific maze')
    parser.add_argument('--model', type=str, default='models/ppo_dataset_trained',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='maze_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--maze-idx', type=int, required=True,
                       help='Index of maze to analyze (0-149)')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Uncertainty threshold')
    parser.add_argument('--boost', type=float, default=2.0,
                       help='Boost strength')
    parser.add_argument('--boost-type', type=str, default='multiplicative',
                       choices=['multiplicative', 'additive'],
                       help='Boost type')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Load model and dataset
    print("Loading model and dataset...")
    model = PPO.load(args.model)
    
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    
    if args.maze_idx >= len(test_mazes):
        print(f"❌ Error: Maze index {args.maze_idx} out of range (0-{len(test_mazes)-1})")
        exit(1)
    
    print(f"✓ Model loaded")
    print(f"✓ Dataset loaded ({len(test_mazes)} test mazes)")
    
    maze = test_mazes[args.maze_idx]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING MAZE #{args.maze_idx+1}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Boost:      {args.boost} ({args.boost_type})")
    print(f"  Wall density: {maze['wall_density']:.2f}")
    print(f"  Optimal path: {maze['path_length']} steps")
    
    # Test baseline
    print(f"\nTesting baseline...")
    baseline_success, baseline_steps, _ = test_maze_with_recording(
        model, maze, use_llm=False
    )
    print(f"  Baseline: {'SUCCESS' if baseline_success else 'FAILED'} in {baseline_steps} steps")
    
    # Test with LLM
    print(f"\nTesting with LLM guidance...")
    llm_success, llm_steps, records = test_maze_with_recording(
        model, maze, use_llm=True, 
        threshold=args.threshold,
        llm_boost=args.boost,
        boost_type=args.boost_type
    )
    print(f"  With LLM: {'SUCCESS' if llm_success else 'FAILED'} in {llm_steps} steps")
    print(f"  LLM queries: {len(records)}")
    
    if len(records) == 0:
        print(f"\n⚠️  No LLM queries recorded!")
        print(f"   Threshold {args.threshold} too high, agent never uncertain")
        exit(0)
    
    # Generate output filename
    if args.output is None:
        args.output = f"maze_{args.maze_idx+1}_probs_{args.boost_type}_{args.boost}.png"
    
    # Visualize
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*70}")
    
    visualize_action_probs_for_maze(args.maze_idx, maze, records, args.output)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("QUERY SUMMARY")
    print(f"{'='*70}")
    
    entropies = [r['entropy'] for r in records]
    max_probs = [np.max(r['probs']) for r in records]
    llm_hints = {}
    
    for record in records:
        hint = record['llm_hint']
        llm_hints[hint] = llm_hints.get(hint, 0) + 1
    
    print(f"Total queries: {len(records)}")
    print(f"\nEntropy at query points:")
    print(f"  Mean:   {np.mean(entropies):.3f}")
    print(f"  Range:  {np.min(entropies):.3f} - {np.max(entropies):.3f}")
    
    print(f"\nMax probability at query points:")
    print(f"  Mean:   {np.mean(max_probs):.3f}")
    
    print(f"\nLLM suggestions:")
    for direction, count in sorted(llm_hints.items(), key=lambda x: x[1], reverse=True):
        print(f"  {direction:10s}: {count} times")
    
    print(f"\n✅ COMPLETE!")
    print(f"   Visualization saved to: {args.output}")
    print(f"{'='*70}")