"""
PPO Training for Maze Navigation
Step-by-step:
1. Wrap GridBulletWorld in Gym interface
2. Train PPO agent
3. Save model for later use
"""
import sys
sys.path.insert(0, '..')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import time  # Add timer
import os  # For file checking
import pickle  # For loading dataset

# Import your environment
from envs.grid_bullet_world import GridBulletWorld
from envs.maze_templates import get_maze_template
from agents.gym_wrapper import MazeEnvGym

# Only import if not using dataset
try:
    from random_maze_generator import RandomMazeGenerator
    HAS_RANDOM_GEN = True
except ImportError:
    HAS_RANDOM_GEN = False


class CurriculumEnv(MazeEnvGym):
    """Environment that cycles through different maze templates AND random mazes"""
    def __init__(self, bullet_env, maze_names=['maze_simple', 'corridor', 'u_shape', 'narrow_gap', 'spiral', 'maze_hard'], 
                 use_random=True, random_ratio=0.5, maze_dataset=None):
        super().__init__(bullet_env)
        self.maze_names = maze_names
        self.templates = [get_maze_template(name) for name in maze_names]
        self.use_random = use_random and HAS_RANDOM_GEN  # Only if available
        self.random_ratio = random_ratio
        
        if self.use_random and HAS_RANDOM_GEN:
            self.random_generator = RandomMazeGenerator(grid_size=10, wall_density=0.25)
        else:
            self.random_generator = None
        
        self.current_idx = 0
        
        # NEW: Use pre-generated dataset
        self.maze_dataset = maze_dataset  # List of maze dicts
        self.dataset_idx = 0
    
    def reset(self, seed=None, options=None):
        """Reset with either a fixed template, dataset maze, or random maze"""
        
        if self.maze_dataset is not None:
            # Use maze from dataset (cycles through training set)
            template = self.maze_dataset[self.dataset_idx]
            self.dataset_idx = (self.dataset_idx + 1) % len(self.maze_dataset)
        
        elif self.use_random and self.random_generator and np.random.random() < self.random_ratio:
            # Generate random maze (only if generator available)
            random_maze = self.random_generator.generate_maze(min_path_length=10)
            template = random_maze
        else:
            # Use fixed template (cycle through)
            template = self.templates[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.templates)
        
        # Use parent class reset to properly clear history and reset environment
        return super().reset(seed=seed, options={'maze_template': template})


def make_curriculum_env(use_random=True, dataset_file=None):
    """
    Create environment that trains on multiple mazes + random generation
    
    Args:
        use_random: Use random maze generation
        dataset_file: Path to pre-generated maze dataset (.pkl file)
    """
    bullet_env = GridBulletWorld(
        gui=False,
        grid_size=10,
        max_steps=60,
        dynamic_rules=[]
    )
    bullet_env.reset()
    
    # Load dataset if provided
    maze_dataset = None
    if dataset_file and os.path.exists(dataset_file):
        print(f"Loading maze dataset: {dataset_file}")
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
            maze_dataset = data['train_mazes']  # Use training set
        print(f"✓ Loaded {len(maze_dataset)} training mazes from dataset")
    
    # Create environment with dataset or random generation
    return CurriculumEnv(
        bullet_env, 
        maze_names=['maze_simple', 'corridor', 'u_shape', 'narrow_gap', 'spiral', 'maze_hard'],
        use_random=use_random and (maze_dataset is None),  # Don't use random if we have dataset
        random_ratio=0.5,
        maze_dataset=maze_dataset
    )


def make_env(maze_name='maze_simple'):
    """Create wrapped environment"""
    template = get_maze_template(maze_name)
    
    # Create GridBulletWorld
    bullet_env = GridBulletWorld(
        gui=False,
        grid_size=template['grid_size'],
        max_steps=50,
        dynamic_rules=[]
    )
    bullet_env.reset(maze_template=template)
    
    # Wrap in Gym interface
    gym_env = MazeEnvGym(bullet_env)
    gym_env._maze_template = template  # Store for resets
    
    return gym_env


def train_ppo(maze_name='maze_simple', total_timesteps=200000, use_curriculum=False, 
              use_random_mazes=True, dataset_file=None):
    """
    Train PPO on a specific maze
    
    Args:
        maze_name: Which maze to train on
        total_timesteps: How many environment steps to train
        use_curriculum: Train on multiple mazes for better generalization
        use_random_mazes: Include randomly generated mazes (prevents overfitting!)
        dataset_file: Path to pre-generated dataset (e.g., 'maze_dataset.pkl')
    """
    print(f"\n{'='*50}")
    print(f"Training PPO")
    if use_curriculum:
        if dataset_file:
            print(f"Using DATASET: {dataset_file}")
            print(f"→ 70 training mazes (prevents overfitting!)")
        elif use_random_mazes:
            print(f"Using curriculum: 6 fixed mazes + RANDOM mazes")
            print(f"Random ratio: 50% (prevents overfitting!)")
        else:
            print(f"Using curriculum: 6 fixed mazes only")
    else:
        print(f"Maze: {maze_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"{'='*50}\n")
    
    # Step 1: Create environment
    if use_curriculum:
        env = make_curriculum_env(use_random=use_random_mazes, dataset_file=dataset_file)
    else:
        env = make_env(maze_name)
    
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space}")
    
    # Step 2: Create PPO model
    model = PPO(
        policy="MlpPolicy",           # Neural network policy
        env=env,
        learning_rate=3e-4,           # Standard PPO learning rate
        n_steps=512,                  # Steps before update (smaller for faster feedback)
        batch_size=64,                # Mini-batch size
        n_epochs=10,                  # Epochs per update
        gamma=0.99,                   # Discount factor
        gae_lambda=0.95,              # GAE parameter
        clip_range=0.2,               # PPO clipping
        ent_coef=0.05,                # INCREASED: More exploration (was 0.01)
        verbose=1,                    # Print training info
        tensorboard_log=f"../logs/ppo_curriculum"
    )
    print(f"✓ PPO model created")
    
    # Step 3: Train!
    print(f"\nStarting training...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    print(f"✓ Training complete!")
    print(f"⏱  Training time: {training_time/60:.1f} minutes ({training_time:.0f} seconds)")
    
    # Step 4: Save model
    import os
    os.makedirs("models", exist_ok=True)  # Create models directory if doesn't exist
    
    if dataset_file:
        model_name = "ppo_dataset_trained"
    elif use_curriculum:
        model_name = "ppo_curriculum"
    else:
        model_name = f"ppo_{maze_name}"
    
    model_path = f"models/{model_name}"
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}.zip")
    
    # Step 5: Quick evaluation
    print(f"\nEvaluating trained agent...")
    evaluate_agent(env, model, n_episodes=10)
    
    env.close()
    return model


def evaluate_agent(env, model, n_episodes=10):
    """Test the trained agent"""
    successes = 0
    total_steps = []
    
    for ep in range(n_episodes):
        # Reset environment (curriculum env will cycle through mazes automatically)
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:  # Add max steps safeguard
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
        
        if info.get('success', False):
            successes += 1
        total_steps.append(steps)
    
    print(f"  Success rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"  Avg steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on mazes')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to maze dataset (.pkl file)')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training timesteps')
    
    args = parser.parse_args()
    
    # Train with dataset if provided, otherwise use random generation
    if args.dataset and os.path.exists(args.dataset):
        print(f"Training with pre-generated dataset: {args.dataset}")
        model = train_ppo(
            use_curriculum=True, 
            use_random_mazes=False,  # Dataset replaces random
            dataset_file=args.dataset,
            total_timesteps=args.timesteps
        )
    else:
        print("Training with random maze generation")
        model = train_ppo(
            use_curriculum=True, 
            use_random_mazes=True, 
            total_timesteps=args.timesteps
        )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print("\nTraining Strategy:")
    if args.dataset:
        print("  ✓ Used 70 diverse training mazes from dataset")
        print("  ✓ Can test on 30 held-out test mazes")
    else:
        print("  ✓ Random generation (infinite variety)")
    print("\nNext steps:")
    print("1. Test generalization with test_generalization.py")
    print("2. Evaluate on held-out test set if using dataset")
    print("="*50)