"""
Analyze Success Patterns: Which mazes does each agent solve?

This helps understand if LLM solves "harder" mazes
"""
import pickle
import numpy as np
from stable_baselines3 import PPO
from envs.grid_bullet_world import GridBulletWorld
from agents.gym_wrapper import MazeEnvGym
from llm_guidance import LLMGuidedAgent, real_llm_query

# Load dataset and model
with open('maze_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    test_mazes = data['test_mazes']

model = PPO.load('models/ppo_dataset_trained')

print("="*60)
print("MAZE DIFFICULTY ANALYSIS")
print("="*60)

# Test both agents and track which mazes succeeded
baseline_successes = []
llm_successes = []

print("\nTesting baseline agent...")
for i, maze in enumerate(test_mazes):
    bullet_env = GridBulletWorld(gui=False, grid_size=10, max_steps=60, dynamic_rules=[])
    env = MazeEnvGym(bullet_env)
    
    obs, _ = env.reset(options={'maze_template': maze})
    done = False
    steps = 0
    
    while not done and steps < 60:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        done = done or truncated
    
    baseline_successes.append(info.get('success', False))
    env.close()
    
    if (i + 1) % 30 == 0:
        print(f"  Tested {i+1}/150 mazes...")

print("\nTesting LLM-guided agent...")
for i, maze in enumerate(test_mazes):
    bullet_env = GridBulletWorld(gui=False, grid_size=10, max_steps=60, dynamic_rules=[])
    env = MazeEnvGym(bullet_env)
    agent = LLMGuidedAgent(model, uncertainty_threshold=1.0, llm_query_fn=real_llm_query)
    
    obs, _ = env.reset(options={'maze_template': maze})
    done = False
    steps = 0
    
    while not done and steps < 60:
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
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        done = done or truncated
    
    llm_successes.append(info.get('success', False))
    env.close()
    
    if (i + 1) % 30 == 0:
        print(f"  Tested {i+1}/150 mazes...")

# Analyze patterns
print("\n" + "="*60)
print("RESULTS")
print("="*60)

both_solved = []
only_baseline = []
only_llm = []
neither = []

for i in range(len(test_mazes)):
    if baseline_successes[i] and llm_successes[i]:
        both_solved.append(i)
    elif baseline_successes[i] and not llm_successes[i]:
        only_baseline.append(i)
    elif not baseline_successes[i] and llm_successes[i]:
        only_llm.append(i)
    else:
        neither.append(i)

print(f"\nMaze Categories:")
print(f"  Both solved: {len(both_solved)} mazes")
print(f"  Only baseline: {len(only_baseline)} mazes")
print(f"  Only LLM: {len(only_llm)} mazes")
print(f"  Neither: {len(neither)} mazes")

# Analyze optimal path lengths
both_optimal = [test_mazes[i]['path_length'] for i in both_solved]
only_llm_optimal = [test_mazes[i]['path_length'] for i in only_llm]
only_baseline_optimal = [test_mazes[i]['path_length'] for i in only_baseline]
neither_optimal = [test_mazes[i]['path_length'] for i in neither]

print(f"\nOptimal Path Lengths (should all be 18 if start/goal are same):")
print(f"  Both solved: {np.mean(both_optimal):.1f} steps (avg)")
print(f"  Only LLM: {np.mean(only_llm_optimal) if only_llm_optimal else 'N/A':.1f} steps (avg)")
print(f"  Only baseline: {np.mean(only_baseline_optimal) if only_baseline_optimal else 'N/A':.1f} steps (avg)")
print(f"  Neither: {np.mean(neither_optimal) if neither_optimal else 'N/A':.1f} steps (avg)")

# Analyze wall density (harder mazes have more walls)
both_density = [test_mazes[i]['wall_density'] for i in both_solved]
only_llm_density = [test_mazes[i]['wall_density'] for i in only_llm]
only_baseline_density = [test_mazes[i]['wall_density'] for i in only_baseline]
neither_density = [test_mazes[i]['wall_density'] for i in neither]

print(f"\nWall Density (higher = more obstacles):")
print(f"  Both solved: {np.mean(both_density):.2f} (avg)")
print(f"  Only LLM: {np.mean(only_llm_density) if only_llm_density else 'N/A':.2f} (avg)")
print(f"  Only baseline: {np.mean(only_baseline_density) if only_baseline_density else 'N/A':.2f} (avg)")
print(f"  Neither: {np.mean(neither_density) if neither_density else 'N/A':.2f} (avg)")

print("\n" + "="*60)
print("CONCLUSION:")
if only_llm_density and np.mean(only_llm_density) > np.mean(both_density):
    print("✓ LLM solves HARDER mazes (more walls) that baseline fails on!")
    print("  This explains why LLM paths are less optimal - it's solving")
    print("  more difficult mazes that require more exploration.")
else:
    print("→ LLM and baseline solve similar difficulty mazes.")
print("="*60)