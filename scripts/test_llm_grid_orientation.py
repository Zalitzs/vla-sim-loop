# FILE: scripts/test_llm_grid_orientation.py
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from envs.grid_bullet_world import GridBulletWorld
from agents.llm_agent import LLMAgent, grid_to_text

# Setup visualization
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

print("="*60)
print("Testing Grid Orientation for LLM (Using get_action)")
print("="*60)

# Create environment
env = GridBulletWorld(gui=False, grid_size=10, dynamic_rules=[])
env.reset()

# Create LLM agent
print("\nCreating LLM agent...")
agent = LLMAgent(model="gpt-4o-mini", temperature=0.7)

# Get the grid states
grid = env.get_grid_state()
cube_pos = env.get_cube_grid_pos()
target_pos = env.get_target_grid_pos()

print(f"\nCube position: {cube_pos}")
print(f"Target position: {target_pos}")

# Show matplotlib version (what we see)
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(np.flipud(grid), cmap=cmap, norm=norm)
plt.title("Matplotlib View\n(What we see - forward = up)")
plt.xlabel(f"Cube: {cube_pos}, Target: {target_pos}")

plt.subplot(1, 3, 2)
# Show unflipped text (wrong)
unflipped_text = grid_to_text(grid)
plt.text(0.05, 0.5, unflipped_text, 
         fontfamily='monospace', fontsize=9,
         verticalalignment='center')
plt.title("Without Flip (WRONG)\n(Original grid_to_text)")
plt.axis('off')

plt.subplot(1, 3, 3)
# Show what get_action actually sends (should be flipped)
plt.text(0.05, 0.5, "Calling get_action...\n(check console for debug output)", 
         fontfamily='monospace', fontsize=9,
         verticalalignment='center')
plt.title("What LLM Actually Sees\n(via get_action - see console)")
plt.axis('off')

plt.tight_layout()
plt.savefig('logs/grid_orientation_test.png')

print("\n" + "="*60)
print("Calling agent.get_action() - watch console for debug output:")
print("="*60)

# This will trigger all the debug prints
action = agent.get_action(env)

print("="*60)
print(f"LLM chose action: {action}")
print("="*60)

print("\n" + "="*60)
print("VERIFICATION:")
print("="*60)
print("Look at the 'Grid text being sent to LLM' above.")
print("Compare it to the Matplotlib view in the image.")
print("Question: Is the C (cube) in the same visual position in both?")
print("="*60)

plt.show()
env.disconnect()