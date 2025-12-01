"""
Visualize Sample Mazes from Dataset

Shows 50 mazes in a grid layout with difficulty info
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load dataset
with open('maze_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    test_mazes = data['test_mazes']

print(f"Total test mazes: {len(test_mazes)}")
print("Creating visualization of 50 sample mazes...")

# Select 50 mazes (evenly distributed)
num_to_show = min(50, len(test_mazes))
indices = np.linspace(0, len(test_mazes)-1, num_to_show, dtype=int)
sample_mazes = [test_mazes[i] for i in indices]

# Create figure with subplots (10 rows x 5 columns)
fig, axes = plt.subplots(10, 5, figsize=(20, 40))
axes = axes.flatten()

for idx, maze in enumerate(sample_mazes):
    ax = axes[idx]
    
    grid = maze['grid']
    start = maze['start_pos']
    goal = maze['target_pos']
    path_len = maze['path_length']
    wall_density = maze['wall_density']
    
    # Plot maze
    ax.imshow(grid.T, cmap='binary', origin='lower')
    
    # Mark start (green) and goal (red)
    ax.plot(start[0], start[1], 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(goal[0], goal[1], 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=2)
    
    # Title with info
    ax.set_title(f'Maze {indices[idx]+1}\nPath={path_len}, Walls={wall_density:.2f}', 
                 fontsize=8)
    ax.axis('off')

# Hide unused subplots
for idx in range(num_to_show, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Sample Mazes from Test Set\n(Green=Start, Red=Goal, Black=Walls)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_mazes.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to: sample_mazes.png")
plt.close()

# Also create a detailed view of 6 interesting mazes
print("\nCreating detailed view of 6 example mazes...")

# Pick 6 interesting ones: low, medium, high wall density
sorted_by_walls = sorted(enumerate(test_mazes), key=lambda x: x[1]['wall_density'])
interesting_indices = [
    sorted_by_walls[5][0],      # Very low walls
    sorted_by_walls[25][0],     # Low-medium walls
    sorted_by_walls[50][0],     # Medium walls
    sorted_by_walls[75][0],     # Medium-high walls
    sorted_by_walls[100][0],    # High walls
    sorted_by_walls[125][0],    # Very high walls
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, maze_idx in enumerate(interesting_indices):
    ax = axes[idx]
    maze = test_mazes[maze_idx]
    
    grid = maze['grid']
    start = maze['start_pos']
    goal = maze['target_pos']
    path_len = maze['path_length']
    wall_density = maze['wall_density']
    
    # Create larger, clearer visualization
    # Plot grid
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
    
    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=20, 
           markeredgecolor='darkgreen', markeredgewidth=3, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=25, 
           markeredgecolor='darkred', markeredgewidth=3, label='Goal')
    
    # Grid
    ax.set_xlim(-0.5, grid.shape[0]-0.5)
    ax.set_ylim(-0.5, grid.shape[1]-0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title
    difficulty = "Easy" if wall_density < 0.22 else "Medium" if wall_density < 0.26 else "Hard"
    ax.set_title(f'Maze {maze_idx+1} ({difficulty})\n'
                f'Optimal Path: {path_len} steps\n'
                f'Wall Density: {wall_density:.1%}', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

plt.suptitle('Detailed View: Example Mazes by Difficulty\n'
            '(Green Circle = Start (0,0), Red Star = Goal (9,9), Black = Walls)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('detailed_mazes.png', dpi=150, bbox_inches='tight')
print("✓ Saved detailed view to: detailed_mazes.png")
plt.close()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print("Created two images:")
print("  1. sample_mazes.png - Grid of 50 sample mazes")
print("  2. detailed_mazes.png - Detailed view of 6 mazes by difficulty")
print("\nOpen these images to see the maze layouts!")
print("="*60)