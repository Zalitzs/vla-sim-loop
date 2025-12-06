"""
Table-Based Action Probability Visualization

Shows action probabilities in a clean table format instead of bar charts
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            logits = self.model.policy.action_net(latent_pi)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-8))
        
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
    """Test a single maze and record action probabilities"""
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
            action_model, probs, entropy = recorder.predict_with_probs(obs)
            env_state = get_env_state(env)
            
            if entropy > threshold:
                llm_hint = agent.llm_query_fn(env_state)
                action, _ = agent.predict(obs, env_state)
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


def create_table_visualization(maze_idx, maze, records, output_file):
    """
    Create clean table visualization showing action probabilities
    """
    action_names = ['Forward', 'Backward', 'Left', 'Right']
    num_queries = len(records)
    
    if num_queries == 0:
        print(f"No LLM queries for maze {maze_idx}")
        return
    
    # Create figure
    fig = plt.figure(figsize=(14, 2 + 1.5 * num_queries))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    
    # Header row
    header = ['Query', 'Position', 'Entropy', 'Forward', 'Backward', 'Left', 'Right', 'LLM\nSuggestion', 'Action\nTaken', 'Followed?']
    table_data.append(header)
    
    # Data rows
    for idx, record in enumerate(records):
        probs = record['probs']
        entropy = record['entropy']
        position = f"({record['position'][0]}, {record['position'][1]})"
        llm_hint = record['llm_hint'].capitalize()
        
        action_map = {0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right'}
        action_taken = action_map[record['action_taken']]
        
        followed = 'Y' if action_taken.lower() == record['llm_hint'].lower() else 'N'
        
        row = [
            f"{idx+1}",
            position,
            f"{entropy:.3f}",
            f"{probs[0]:.3f}",
            f"{probs[1]:.3f}",
            f"{probs[2]:.3f}",
            f"{probs[3]:.3f}",
            llm_hint,
            action_taken,
            followed
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.06, 0.1, 0.08, 0.09, 0.09, 0.09, 0.09, 0.11, 0.11, 0.09])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color data cells based on content
    action_map_inv = {'forward': 3, 'backward': 4, 'left': 5, 'right': 6}
    
    for row_idx in range(1, len(table_data)):
        record = records[row_idx - 1]
        probs = record['probs']
        llm_hint = record['llm_hint'].lower()
        action_taken = record['action_taken']
        
        # Highlight action taken (column 8)
        table[(row_idx, 8)].set_facecolor('#FFE082')
        table[(row_idx, 8)].set_text_props(weight='bold')
        
        # Highlight LLM suggestion (column 7)
        table[(row_idx, 7)].set_facecolor('#B3E5FC')
        table[(row_idx, 7)].set_text_props(weight='bold')
        
        # Highlight followed/ignored (column 9)
        followed = table_data[row_idx][9]
        if followed == 'Y':
            table[(row_idx, 9)].set_facecolor('#C8E6C9')
            table[(row_idx, 9)].set_text_props(weight='bold', color='#2E7D32', fontsize=14)
        else:
            table[(row_idx, 9)].set_facecolor('#FFCDD2')
            table[(row_idx, 9)].set_text_props(weight='bold', color='#C62828', fontsize=14)
        
        # Highlight the probability corresponding to LLM suggestion
        if llm_hint in action_map_inv:
            prob_col = action_map_inv[llm_hint]
            table[(row_idx, prob_col)].set_facecolor('#E1F5FE')
            table[(row_idx, prob_col)].set_text_props(weight='bold')
        
        # Color entropy based on value
        entropy_val = record['entropy']
        if entropy_val > 1.3:
            table[(row_idx, 2)].set_facecolor('#FFEBEE')  # High uncertainty
        elif entropy_val > 1.1:
            table[(row_idx, 2)].set_facecolor('#FFF9C4')  # Medium uncertainty
    
    # Add title
    title = (f"Maze #{maze_idx+1}: Action Probabilities at LLM Query Points\n"
            f"Wall Density: {maze['wall_density']:.2f} | Optimal Path: {maze['path_length']} steps | "
            f"Total Queries: {num_queries}")
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_text = (
        "Color Guide:\n"
        "* Light Blue (LLM Suggestion column) = What LLM recommends\n"
        "* Light Orange (Action Taken column) = What agent actually did\n"
        "* Green Y = Agent followed LLM | Red N = Agent ignored LLM\n"
        "* Highlighted probability = LLM's suggested action probability"
    )
    plt.text(0.5, -0.05, legend_text, transform=ax.transAxes, 
            fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved table visualization: {output_file}")
    
    # Also save JSON data
    json_file = output_file.replace('.png', '.json')
    json_data = {
        'maze_idx': int(maze_idx + 1),
        'maze_info': {
            'wall_density': float(maze['wall_density']),
            'optimal_path': int(maze['path_length']),
            'total_queries': int(len(records))
        },
        'queries': []
    }
    
    action_map = {0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right'}
    
    for idx, record in enumerate(records):
        # Convert position to list of ints
        position = record['position']
        if hasattr(position, '__iter__'):
            position_list = [int(p) for p in position]
        else:
            position_list = [int(position)]
        
        query_data = {
            'query_num': int(idx + 1),
            'position': position_list,
            'entropy': float(record['entropy']),
            'probabilities': {
                'forward': float(record['probs'][0]),
                'backward': float(record['probs'][1]),
                'left': float(record['probs'][2]),
                'right': float(record['probs'][3])
            },
            'llm_suggestion': str(record['llm_hint']),
            'action_taken': str(action_map[int(record['action_taken'])]),
            'followed': bool(action_map[int(record['action_taken'])].lower() == str(record['llm_hint']).lower())
        }
        json_data['queries'].append(query_data)
    
    # Calculate summary statistics
    follow_count = sum(1 for q in json_data['queries'] if q['followed'])
    json_data['summary'] = {
        'total_queries': int(len(records)),
        'follow_count': int(follow_count),
        'follow_rate': float(follow_count / len(records) if records else 0),
        'avg_entropy': float(np.mean([r['entropy'] for r in records])),
        'min_entropy': float(np.min([r['entropy'] for r in records])),
        'max_entropy': float(np.max([r['entropy'] for r in records]))
    }
    
    import json
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  [OK] Saved JSON data: {json_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze action probabilities (table format)')
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
                       help='Output filename')
    
    args = parser.parse_args()
    
    # Load model and dataset
    print("Loading model and dataset...")
    model = PPO.load(args.model)
    
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    
    if args.maze_idx >= len(test_mazes):
        print(f"❌ Error: Maze index {args.maze_idx} out of range")
        exit(1)
    
    print(f"[OK] Loaded")
    maze = test_mazes[args.maze_idx]
    
    print(f"\n{'='*70}")
    print(f"MAZE #{args.maze_idx+1} | Threshold: {args.threshold} | "
          f"Boost: {args.boost} ({args.boost_type})")
    print(f"{'='*70}")
    
    # Test baseline
    baseline_success, baseline_steps, _ = test_maze_with_recording(model, maze, use_llm=False)
    baseline_status = "[OK] SUCCESS" if baseline_success else "[X] FAILED"
    print(f"Baseline: {baseline_status} ({baseline_steps} steps)")
    
    # Test with LLM
    llm_success, llm_steps, records = test_maze_with_recording(
        model, maze, use_llm=True, threshold=args.threshold,
        llm_boost=args.boost, boost_type=args.boost_type
    )
    llm_status = "[OK] SUCCESS" if llm_success else "[X] FAILED"
    print(f"With LLM: {llm_status} ({llm_steps} steps, {len(records)} queries)")
    
    if len(records) == 0:
        print(f"\n[!] No LLM queries (threshold {args.threshold} too high)")
        exit(0)
    
    # Generate output
    if args.output is None:
        args.output = f"maze_{args.maze_idx+1}_table.png"
    
    create_table_visualization(args.maze_idx, maze, records, args.output)
    
    print(f"\n[OK] Complete! Table saved to: {args.output}")
    print(f"{'='*70}")