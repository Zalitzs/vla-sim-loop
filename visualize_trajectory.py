"""
Per-Maze Trajectory Visualizer

Visualizes the actual path an agent took through a specific maze
Use this AFTER running comprehensive comparison to inspect interesting mazes
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym
from llm_guidance import LLMGuidedAgent, real_llm_query


def visualize_trajectory(maze_idx, model_path, dataset_file, use_llm=False, llm_mode='real'):
    """
    Visualize agent's trajectory through a specific maze
    
    Args:
        maze_idx: Index of maze to visualize (0-149 for 150 mazes)
        model_path: Path to trained model
        dataset_file: Path to maze dataset
        use_llm: Whether to use LLM guidance
        llm_mode: 'simple', 'real', or 'cached'
    """
    
    # Load maze
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        test_mazes = data['test_mazes']
    
    if maze_idx >= len(test_mazes):
        print(f"Error: Maze index {maze_idx} out of range (0-{len(test_mazes)-1})")
        return
    
    maze = test_mazes[maze_idx]
    
    # Load model
    model = PPO.load(model_path)
    
    # Set up agent
    if use_llm:
        if llm_mode == 'real':
            from llm_guidance import real_llm_query
            llm_fn = real_llm_query
        elif llm_mode == 'cached':
            from llm_guidance import cached_llm_query
            llm_fn = cached_llm_query
        else:
            from llm_guidance import simple_llm_query
            llm_fn = simple_llm_query
        agent = LLMGuidedAgent(model, uncertainty_threshold=1.0, llm_query_fn=llm_fn)
    
    # Run episode and record trajectory
    bullet_env = GridBulletWorld(gui=False, grid_size=10, max_steps=60, dynamic_rules=[])
    env = MazeEnvGym(bullet_env)
    
    obs, _ = env.reset(options={'maze_template': maze})
    done = False
    steps = 0
    
    trajectory = [bullet_env.get_cube_grid_pos()]
    llm_queries_at_step = []
    
    while not done and steps < 60:
        if use_llm:
            def get_env_state(env):
                agent_pos = bullet_env.get_cube_grid_pos()
                target_pos = bullet_env.get_target_grid_pos()
                walls = {
                    'forward': bullet_env.is_wall_at_grid(agent_pos[0] + 1, agent_pos[1]),
                    'backward': bullet_env.is_wall_at_grid(agent_pos[0] - 1, agent_pos[1]),
                    'left': bullet_env.is_wall_at_grid(agent_pos[0], agent_pos[1] - 1),
                    'right': bullet_env.is_wall_at_grid(agent_pos[0], agent_pos[1] + 1),
                }
                return {'agent_pos': agent_pos, 'target_pos': target_pos, 'walls': walls}
            
            env_state = get_env_state(env)
            action, was_uncertain = agent.predict(obs, env_state)
            llm_queries_at_step.append(was_uncertain)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            llm_queries_at_step.append(False)
        
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        trajectory.append(bullet_env.get_cube_grid_pos())
        done = done or truncated
    
    env.close()
    success = info.get('success', False)
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    grid = maze['grid']
    start = maze['start_pos']
    goal = maze['target_pos']
    
    # Draw grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:  # Wall
                rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
            else:  # Empty
                rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                               facecolor='white', edgecolor='lightgray', linewidth=0.5)
                ax.add_patch(rect)
    
    # Draw trajectory
    for i in range(len(trajectory) - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        
        # Color: blue if no LLM query, orange if LLM queried
        color = 'orange' if llm_queries_at_step[i] else 'blue'
        alpha = 0.6
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15,
                               color=color, alpha=alpha, linewidth=2)
        ax.add_patch(arrow)
        
        # Mark step number
        ax.text(x1, y1, str(i), fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=20, 
           markeredgecolor='darkgreen', markeredgewidth=3, label='Start', zorder=10)
    ax.plot(goal[0], goal[1], 'r*', markersize=30, 
           markeredgecolor='darkred', markeredgewidth=3, label='Goal', zorder=10)
    
    # Mark final position
    final_pos = trajectory[-1]
    ax.plot(final_pos[0], final_pos[1], 'mo', markersize=15,
           markeredgecolor='darkmagenta', markeredgewidth=2, label='Final', zorder=10)
    
    ax.set_xlim(-0.5, grid.shape[0]-0.5)
    ax.set_ylim(-0.5, grid.shape[1]-0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title
    agent_type = "LLM-Guided" if use_llm else "Baseline"
    result = "SUCCESS" if success else "FAILED"
    title = f'Maze #{maze_idx+1} - {agent_type} Agent - {result}\n'
    title += f'Steps: {len(trajectory)-1}, Optimal: {maze["path_length"]}, '
    title += f'Wall Density: {maze["wall_density"]:.2f}\n'
    if use_llm:
        llm_count = sum(llm_queries_at_step)
        title += f'LLM Queries: {llm_count} (Orange arrows = LLM guided)'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # Save
    filename = f'trajectory_maze{maze_idx+1}_{agent_type.lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved trajectory visualization to: {filename}")
    print(f"  Result: {result}")
    print(f"  Steps: {len(trajectory)-1} (optimal: {maze['path_length']})")
    if use_llm:
        print(f"  LLM queries: {llm_count} ({llm_count/(len(trajectory)-1)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize agent trajectory through specific maze')
    parser.add_argument('--maze-idx', type=int, required=True,
                       help='Index of maze to visualize (0-149)')
    parser.add_argument('--model', type=str, default='models/ppo_dataset_trained',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='maze_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--llm', action='store_true',
                       help='Use LLM guidance')
    parser.add_argument('--llm-mode', type=str, default='real',
                       choices=['simple', 'real', 'cached'],
                       help='LLM mode')
    
    args = parser.parse_args()
    
    print(f"\nVisualizing maze #{args.maze_idx+1}...")
    visualize_trajectory(args.maze_idx, args.model, args.dataset, 
                        use_llm=args.llm, llm_mode=args.llm_mode)