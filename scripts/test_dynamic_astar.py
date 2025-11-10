# FILE: scripts/test_dynamic_astar.py
import sys
sys.path.append('..')

from envs.grid_bullet_world import GridBulletWorld
from envs.env_rules import shift_wall_right_every_5
from agents.astar_agent import AStarAgent, DynamicAStarAgent

print("="*60)
print("Comparing A* vs Dynamic A* on Dynamic Maze")
print("="*60)

# Test regular A*
print("\n1. Testing Regular A* (should NOT see replanning):")
print("-"*60)
env = GridBulletWorld(gui=False, grid_size=20, dynamic_rules=[shift_wall_right_every_5])
env.reset()
agent = AStarAgent()

for i in range(20):
    action = agent.get_action(env)
    obs, reward, done, info = env.step(action)
    if done:
        break
env.disconnect()

# Test Dynamic A*
print("\n2. Testing Dynamic A* (SHOULD see replanning every 3 steps):")
print("-"*60)
env = GridBulletWorld(gui=False, grid_size=20, dynamic_rules=[shift_wall_right_every_5])
env.reset()
agent = DynamicAStarAgent(replan_frequency=3)

for i in range(20):
    action = agent.get_action(env)
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"\nDynamic A* replanned {agent.replan_count} times")
env.disconnect()