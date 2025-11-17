import sys
sys.path.append('..')

from envs.grid_bullet_world import GridBulletWorld
from envs.maze_templates import get_maze_template
from agents.llm_agent_path_planner import LLMPathPlannerAgent
import time

print("="*70)
print("TESTING PATH PLANNER LLM AGENT")
print("="*70)

# Choose maze to test
MAZE_NAME = 'u_shape'  # Change to 'u_shape', 'maze_simple', etc.

# Load maze
template = get_maze_template(MAZE_NAME)
print(f"\n📋 Testing on: {MAZE_NAME}")
print(f"   {template['description']}")
print(f"   Start: {template['start_pos']} → Target: {template['target_pos']}")

# Create environment
env = GridBulletWorld(
    gui=True,  # Watch it!
    grid_size=template['grid_size'],
    max_steps=100  # Allow more steps since we might replan
)
env.reset(maze_template=template)

# Create path planner agent
agent = LLMPathPlannerAgent(
    model="gpt-4o-mini",  # Cheapest model
    temperature=0.0,       # Deterministic
    max_replans=3          # Limit replanning to 3 attempts
)

print(f"\n🚀 Starting navigation...")
print("="*70)

success = False
total_steps = 0

for step in range(100):
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    
    print(f"\n{'─'*70}")
    print(f"STEP {step + 1}")
    print(f"  Position: {cube_pos} → Target: {target_pos}")
    
    # Get action (will show which action from the plan)
    action = agent.get_action(env, verbose=False)
    
    print(f"  Action: {action}")
    
    # Execute
    obs, reward, done, info = env.step(action)
    total_steps += 1
    
    new_pos = env.get_cube_grid_pos()
    if new_pos != cube_pos:
        print(f"  ✓ Moved to {new_pos}")
    else:
        print(f"  ⚠️  Blocked at {cube_pos}")
    
    if done:
        if info['success']:
            print(f"\n{'='*70}")
            print(f"🎉 SUCCESS in {total_steps} steps!")
            print(f"{'='*70}")
            success = True
        else:
            print(f"\n{'='*70}")
            print(f"❌ TIMEOUT after {total_steps} steps")
            print(f"{'='*70}")
        break
    
    time.sleep(0.5)  # Pause to watch

# Summary
print(f"\n📊 SUMMARY:")
print(f"  Maze: {MAZE_NAME}")
print(f"  Success: {'YES ✓' if success else 'NO ✗'}")
print(f"  Total steps: {total_steps}")
print(f"  API calls (plans): {agent.num_calls}")
print(f"  Replans used: {agent.replan_count}/{agent.max_replans}")
print(f"  Total tokens: {agent.total_tokens}")

env.disconnect()
print(f"\n✓ Done!")