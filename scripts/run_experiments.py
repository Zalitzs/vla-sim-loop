# FILE: scripts/run_experiments.py
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from datetime import datetime
import os

from envs.grid_bullet_world import GridBulletWorld
from agents.astar_agent import AStarAgent, DynamicAStarAgent
from agents.llm_agent import LLMAgent

from envs.maze_templates import get_maze_template
from envs.env_rules import shift_wall_right_every_5
    
class ExperimentRunner:
    """Runs experiments comparing different agents on different mazes"""
    
    def __init__(self, log_dir='../logs'):
        self.log_dir = log_dir
        self.results = []
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        print(f"DEBUG: Logs will be saved to: {os.path.abspath(log_dir)}")
        
    def run_episode(self, env, agent, max_steps=70, verbose=False):
        """Run a single episode with an agent in an environment
        
        Args:
            env: GridBulletWorld environment
            agent: Agent with get_action(env) method
            max_steps: Maximum steps before timeout
            verbose: Print step-by-step info
            
        Returns:
            dict: Episode results with metrics
        """
        obs = env._obs()
        steps = 0
        total_reward = 0
        path = [env.get_cube_grid_pos()]
        actions_taken = []
        
     
        initial_cube_pos = env.get_cube_grid_pos()
        initial_target_pos = env.get_target_grid_pos()
        initial_distance = obs['dist']
        
       
        manhattan_distance = abs(initial_cube_pos[0] - initial_target_pos[0]) + \
                            abs(initial_cube_pos[1] - initial_target_pos[1])
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.get_action(env)
            actions_taken.append(action)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            # After env.step(action) in run_episode
            if env.gui:
                import time
                time.sleep(.5)  # Pause half a second between steps
            # Track metrics
            steps += 1
            total_reward += reward
            path.append(env.get_cube_grid_pos())
            
            if verbose:
                cube_pos = env.get_cube_grid_pos()
                target_pos = env.get_target_grid_pos()
                print(f"  Step {step}: {action:8s} → Cube={cube_pos}, Target={target_pos}, Reward={reward:.3f}")
            
            if done:
                break
        
        # Calculate metrics
        return {
            'success': info.get('success', False),
            'steps': steps,
            'total_reward': total_reward,
            'path_length': len(path),
            'final_distance': obs['dist'],
            'initial_distance': initial_distance,              
            'manhattan_distance': manhattan_distance,          
            'path_efficiency': manhattan_distance / steps if steps > 0 else 0,  
            'initial_cube_pos': initial_cube_pos,          
            'initial_target_pos': initial_target_pos,         
            'path': path,
            'actions': actions_taken
        }
        
    def run_experiment(self, agent_name, agent_class, maze_config, num_trials=10, verbose=False, maze_template=None):
        """Run multiple trials of an agent on a maze configuration
        
        Args:
            agent_name: String name of agent
            agent_class: Agent class to instantiate
            maze_config: Dict of kwargs for GridBulletWorld
            num_trials: Number of episodes to run
            verbose: Print detailed info
            maze_template: Optional dict from get_maze_template() for fixed maze
        """
        print(f"\n{'='*60}")
        print(f"Running: {agent_name} - {num_trials} trials")
        if maze_template:
            print(f"Maze: {maze_template['description']}")
        print(f"{'='*60}")
        
        trial_results = []
        
        for trial in range(num_trials):
            if verbose:
                print(f"\nTrial {trial + 1}/{num_trials}")
            
            # Create environment
            if maze_template:
                env = GridBulletWorld(gui=False, grid_size=maze_template['grid_size'], **maze_config)
                env.reset(maze_template=maze_template)

            else:
                env = GridBulletWorld(gui=False, grid_size=20, **maze_config)
                env.reset()
   
            agent = agent_class()
            
            # Run episode
            result = self.run_episode(env, agent, verbose=verbose)
            
            # Add metadata
            result['agent'] = agent_name
            result['trial'] = trial
            result['timestamp'] = datetime.now().isoformat()
            if maze_template:
                result['maze_name'] = agent_name.split('-')[1]  # Store maze name
            
            trial_results.append(result)
            
            # Clean up
            env.disconnect()
            
            # Print trial summary
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"  Trial {trial + 1}: {status} in {result['steps']} steps (dist={result['final_distance']:.3f})")
        
        # Store results
        self.results.extend(trial_results)
        
        # Print aggregate statistics
        self.print_summary(agent_name, trial_results)
        
        return trial_results
    
    def print_summary(self, agent_name, results):
        """Print aggregate statistics for a set of results"""
        success_rate = np.mean([r['success'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        std_steps = np.std([r['steps'] for r in results])
        avg_reward = np.mean([r['total_reward'] for r in results])
        
        # NEW: Initial condition statistics
        avg_initial_dist = np.mean([r['initial_distance'] for r in results])
        avg_manhattan = np.mean([r['manhattan_distance'] for r in results])
        avg_efficiency = np.mean([r['path_efficiency'] for r in results])
        
        # Only calculate for successful episodes
        successful = [r for r in results if r['success']]
        if successful:
            avg_success_steps = np.mean([r['steps'] for r in successful])
            avg_success_efficiency = np.mean([r['path_efficiency'] for r in successful])
        else:
            avg_success_steps = float('nan')
            avg_success_efficiency = float('nan')
        
        print(f"\n{'─'*60}")
        print(f"Summary for {agent_name}:")
        print(f"{'─'*60}")
        print(f"  Success Rate:          {success_rate*100:.1f}% ({sum([r['success'] for r in results])}/{len(results)})")
        print(f"  Avg Initial Distance:  {avg_initial_dist:.2f} (Manhattan: {avg_manhattan:.1f})")  # NEW
        print(f"  Avg Steps (all):       {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"  Avg Steps (success):   {avg_success_steps:.1f}")
        print(f"  Path Efficiency (all): {avg_efficiency:.2f} (1.0 = optimal)")  # NEW
        print(f"  Path Efficiency (succ):{avg_success_efficiency:.2f}")          # NEW
        print(f"  Avg Total Reward:      {avg_reward:.2f}")
        print(f"{'─'*60}")
        
    def save_results(self, filename=None):
        """Save all results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.csv"
        
        filepath = os.path.join(self.log_dir, filename)
        
        # Convert results to DataFrame (excluding path/actions for CSV)
        df_data = []
        for r in self.results:
            row = {k: v for k, v in r.items() if k not in ['path', 'actions']}
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath
    
    def compare_agents(self):
        """Print comparison table across all agents tested"""
        if not self.results:
            print("No results to compare!")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print("COMPARISON ACROSS ALL AGENTS")
        print(f"{'='*60}\n")
        
        comparison = df.groupby('agent').agg({
            'success': ['mean', 'sum', 'count'],
            'steps': ['mean', 'std'],
            'total_reward': 'mean',
            'final_distance': 'mean',
            'initial_distance': 'mean',           # NEW
            'manhattan_distance': 'mean',         # NEW
            'path_efficiency': 'mean'             # NEW
        }).round(2)
        
        print(comparison)
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Get absolute path to logs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    log_dir = os.path.join(project_dir, 'logs')
    
    print(f"Logs directory: {log_dir}")
    
    # Create experiment runner
    runner = ExperimentRunner(log_dir=log_dir)
    
    # Define maze configurations
    static_config = {'dynamic_rules': []}
    
    # Get all maze templates (including your new maze_hard)
    maze_names = ['corridor', 'u_shape', 'maze_simple', 'narrow_gap', 'spiral', 'maze_hard']
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS: A* vs LLM on All Mazes")
    print("="*70)
    
    # Run experiments on each maze
    for maze_name in maze_names:
        try:
            template = get_maze_template(maze_name)
        except ValueError as e:
            print(f"\nSkipping {maze_name}: {e}")
            continue
        
        print("\n" + "="*70)
        print(f"MAZE: {maze_name.upper()}")
        print(f"Description: {template['description']}")
        print(f"Grid size: {template['grid_size']}")
        print("="*70)
        
        # Experiment 1: A* on this maze
        print(f"\n--- Testing A* on {maze_name} ---")
        runner.run_experiment(
            agent_name=f'A*-{maze_name}',
            agent_class=AStarAgent,
            maze_config=static_config,
            num_trials=10,
            verbose=False,
            maze_template=template
        )
        
        # Experiment 2: LLM on this maze
        print(f"\n--- Testing LLM on {maze_name} ---")
        runner.run_experiment(
            agent_name=f'LLM-{maze_name}',
            agent_class=LLMAgent,
            maze_config=static_config,
            num_trials=5,  # Fewer trials to save money
            verbose=False,
            maze_template=template
        )
    
    # Final comparison across all mazes
    print("\n" + "="*70)
    print("FINAL COMPARISON: A* vs LLM ACROSS ALL MAZES")
    print("="*70)
    runner.compare_agents()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"astar_vs_llm_results_{timestamp}.csv"
    runner.save_results(filename)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {os.path.join(log_dir, filename)}")
    print("\nNext steps:")
    print("1. Check the results CSV")
    print("2. Generate plots for visualization")
    print("3. Implement PPO agent for comparison")