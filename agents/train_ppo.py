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

# Import your environment
from envs.grid_bullet_world import GridBulletWorld
from envs.maze_templates import get_maze_template
from agents.gym_wrapper import MazeEnvGym


class CurriculumEnv(MazeEnvGym):
    """Environment that cycles through different maze templates"""
    def __init__(self, bullet_env, maze_names=['maze_simple', 'corridor', 'u_shape', 'narrow_gap', 'spiral', 'maze_hard']):
        super().__init__(bullet_env)
        self.maze_names = maze_names
        self.templates = [get_maze_template(name) for name in maze_names]
        self.current_idx = 0
    
    def reset(self, seed=None, options=None):
        """Reset with a different maze each time"""
        # Cycle through mazes
        template = self.templates[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.templates)
        
        self.env.reset(maze_template=template)
        obs = self._get_obs()
        return obs, {}


def make_curriculum_env():
    """Create environment that trains on multiple mazes"""
    # Use largest grid size from all mazes
    bullet_env = GridBulletWorld(
        gui=False,
        grid_size=10,
        max_steps=60,  # Increased for harder mazes
        dynamic_rules=[]
    )
    bullet_env.reset()
    
    return CurriculumEnv(bullet_env, maze_names=['maze_simple', 'corridor', 'u_shape', 'narrow_gap', 'spiral', 'maze_hard'])


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


def train_ppo(maze_name='maze_simple', total_timesteps=200000, use_curriculum=False):
    """
    Train PPO on a specific maze
    
    Args:
        maze_name: Which maze to train on
        total_timesteps: How many environment steps to train
        use_curriculum: Train on multiple mazes for better generalization
    """
    print(f"\n{'='*50}")
    print(f"Training PPO")
    if use_curriculum:
        print(f"Using curriculum: maze_simple → corridor → u_shape")
    else:
        print(f"Maze: {maze_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"{'='*50}\n")
    
    # Step 1: Create environment
    if use_curriculum:
        # Train on progressively harder mazes
        env = make_curriculum_env()
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
    model_path = f"../models/ppo_{maze_name}"
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
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
    # Train with curriculum learning (multiple mazes)
    # Increased timesteps for harder mazes like maze_hard
    model = train_ppo(use_curriculum=True, total_timesteps=500000)
    
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Test on all maze types for generalization")
    print("2. If maze_hard still fails, try method 3 or 4")
    print("3. Add LLM guidance for strategic planning")
    print("="*50)