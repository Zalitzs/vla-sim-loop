"""
Test PPO Agent on Held-Out Test Set

Evaluates trained agent on mazes it has NEVER seen before
"""
import sys
sys.path.insert(0, '..')

import pickle
import numpy as np
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym


def test_on_dataset(model_path, dataset_file='maze_dataset.pkl'):
    """Test agent on held-out test mazes"""
    
    # Load model
    print("="*60)
    print("TESTING ON HELD-OUT TEST SET")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_file}\n")
    
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # Load test set
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    
    print(f"✓ Loaded {len(test_mazes)} test mazes (UNSEEN during training)\n")
    
    # Test on each maze
    results = []
    
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
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
        
        success = info.get('success', False)
        results.append({
            'maze_id': i,
            'success': success,
            'steps': steps,
            'path_length': maze_template['path_length']
        })
        
        env.close()
        
        status = "✓" if success else "✗"
        if (i + 1) % 10 == 0:
            print(f"Tested {i+1}/{len(test_mazes)} mazes...")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / len(results) * 100
    
    avg_steps = np.mean([r['steps'] for r in results])
    avg_steps_success = np.mean([r['steps'] for r in results if r['success']]) if successes > 0 else 0
    
    print(f"\nTest Set Performance:")
    print(f"  Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
    print(f"  Avg Steps (all): {avg_steps:.1f}")
    print(f"  Avg Steps (successful): {avg_steps_success:.1f}")
    
    # Difficulty analysis
    easy = [r for r in results if r['path_length'] < 15]
    medium = [r for r in results if 15 <= r['path_length'] < 25]
    hard = [r for r in results if r['path_length'] >= 25]
    
    print(f"\nBy Difficulty:")
    if easy:
        easy_success = sum(1 for r in easy if r['success']) / len(easy) * 100
        print(f"  Easy (path<15):    {easy_success:.1f}% ({len(easy)} mazes)")
    if medium:
        med_success = sum(1 for r in medium if r['success']) / len(medium) * 100
        print(f"  Medium (15-25):    {med_success:.1f}% ({len(medium)} mazes)")
    if hard:
        hard_success = sum(1 for r in hard if r['success']) / len(hard) * 100
        print(f"  Hard (path≥25):    {hard_success:.1f}% ({len(hard)} mazes)")
    
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/ppo_dataset_trained',
                       help='Path to trained model (without .zip extension)')
    parser.add_argument('--dataset', type=str, default='maze_dataset.pkl',
                       help='Path to dataset file')
    
    args = parser.parse_args()
    
    results = test_on_dataset(args.model, args.dataset)