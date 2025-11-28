"""
Maze Dataset Generator with Train/Test Split

Uses procedural algorithms (random density) to generate diverse mazes
Supports 70/30 train/test split to evaluate generalization
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import pickle
from collections import deque


class MazeDataset:
    """Generate and manage large maze datasets using procedural generation"""
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.train_mazes = []
        self.test_mazes = []
    
    def generate_dataset(self, total_mazes=100, test_split=0.3):
        """
        Generate full dataset using random density algorithm
        
        Args:
            total_mazes: Total number of mazes to generate
            test_split: Fraction for test set (e.g., 0.3 = 30%)
        """
        print("="*60)
        print("MAZE DATASET GENERATOR")
        print("="*60)
        print(f"Generating {total_mazes} mazes ({self.grid_size}×{self.grid_size})")
        print(f"Train/Test split: {int((1-test_split)*100)}/{int(test_split*100)}")
        print(f"Algorithm: Random density (15-35% walls)")
        print("="*60)
        
        all_mazes = []
        failed_count = 0
        
        while len(all_mazes) < total_mazes:
            # Vary wall density for diversity (15% to 35%)
            density = random.uniform(0.15, 0.35)
            
            maze = self._generate_random_density_maze(density)
            
            if maze['path_length'] >= 5:  # At least 5 steps
                all_mazes.append(maze)
                
                if len(all_mazes) % 10 == 0:
                    print(f"Generated: {len(all_mazes)}/{total_mazes}")
            else:
                failed_count += 1
                if failed_count > total_mazes * 3:
                    print("Warning: Many short mazes, continuing anyway...")
                    failed_count = 0
        
        # Shuffle and split
        random.shuffle(all_mazes)
        split_idx = int(total_mazes * (1 - test_split))
        
        self.train_mazes = all_mazes[:split_idx]
        self.test_mazes = all_mazes[split_idx:]
        
        print(f"\n✓ Generated {total_mazes} mazes")
        print(f"  Train set: {len(self.train_mazes)} mazes")
        print(f"  Test set:  {len(self.test_mazes)} mazes")
        
        # Statistics
        self._print_statistics()
        
        return self.train_mazes, self.test_mazes
    
    def _generate_random_density_maze(self, density):
        """
        Generate maze with random wall density
        Guaranteed solvable
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Random wall placement
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < density:
                    grid[i, j] = 1
        
        # Start and goal at opposite corners
        start_pos = (0, 0)
        target_pos = (self.grid_size - 1, self.grid_size - 1)
        
        # Ensure start/goal are free
        grid[start_pos] = 0
        grid[target_pos] = 0
        
        # Clear 2x2 area around start and goal to ensure connectivity
        for i in range(min(2, self.grid_size)):
            for j in range(min(2, self.grid_size)):
                grid[i, j] = 0
                grid[self.grid_size - 1 - i, self.grid_size - 1 - j] = 0
        
        # Find path
        path = self._find_path(grid, start_pos, target_pos)
        
        # If no path exists, carve one
        if path is None:
            path = self._carve_simple_path(grid, start_pos, target_pos)
        
        path_length = len(path) - 1 if path else -1
        
        # Get wall positions
        walls = [(i, j) for i in range(self.grid_size) 
                for j in range(self.grid_size) if grid[i, j] == 1]
        
        return {
            'grid': grid,
            'start_pos': start_pos,
            'target_pos': target_pos,
            'walls': walls,
            'path_length': path_length,
            'wall_density': len(walls) / (self.grid_size ** 2),
            'grid_size': self.grid_size
        }
    
    def _carve_simple_path(self, grid, start, goal):
        """Carve a simple path if maze is unsolvable"""
        x, y = start
        gx, gy = goal
        path = [(x, y)]
        
        while (x, y) != goal:
            # Move toward goal
            if x < gx:
                x += 1
            elif x > gx:
                x -= 1
            elif y < gy:
                y += 1
            elif y > gy:
                y -= 1
            
            grid[x, y] = 0
            path.append((x, y))
        
        return path
    
    def _find_path(self, grid, start, target):
        """BFS pathfinding"""
        if grid[start] == 1 or grid[target] == 1:
            return None
        
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == target:
                return path
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    grid[nx, ny] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return None
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "─"*60)
        print("DATASET STATISTICS")
        print("─"*60)
        
        for name, mazes in [("TRAIN", self.train_mazes), ("TEST", self.test_mazes)]:
            if not mazes:
                continue
            
            path_lengths = [m['path_length'] for m in mazes]
            densities = [m['wall_density'] for m in mazes]
            
            print(f"\n{name} SET ({len(mazes)} mazes):")
            print(f"  Path Length:  min={min(path_lengths):2d}, "
                  f"max={max(path_lengths):2d}, "
                  f"avg={np.mean(path_lengths):.1f}")
            print(f"  Wall Density: min={min(densities):.2f}, "
                  f"max={max(densities):.2f}, "
                  f"avg={np.mean(densities):.2f}")
        
        print("─"*60)
    
    def save_dataset(self, filename='maze_dataset.pkl'):
        """Save dataset to file"""
        data = {
            'grid_size': self.grid_size,
            'train_mazes': self.train_mazes,
            'test_mazes': self.test_mazes
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Saved dataset to: {filename}")
        print(f"  File size: {os.path.getsize(filename) / 1024:.1f} KB")
    
    def load_dataset(self, filename='maze_dataset.pkl'):
        """Load dataset from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.grid_size = data['grid_size']
        self.train_mazes = data['train_mazes']
        self.test_mazes = data['test_mazes']
        
        print(f"✓ Loaded dataset from: {filename}")
        print(f"  Train: {len(self.train_mazes)} mazes")
        print(f"  Test:  {len(self.test_mazes)} mazes")
        
        return self.train_mazes, self.test_mazes
    
    def visualize_sample(self, num_samples=5, from_test=False):
        """Show sample mazes"""
        mazes = self.test_mazes if from_test else self.train_mazes
        dataset_name = "TEST" if from_test else "TRAIN"
        
        print(f"\n{'='*60}")
        print(f"SAMPLE MAZES FROM {dataset_name} SET")
        print(f"{'='*60}")
        
        samples = random.sample(mazes, min(num_samples, len(mazes)))
        
        for i, maze in enumerate(samples):
            print(f"\nMaze {i+1}: Path={maze['path_length']} steps, "
                  f"Density={maze['wall_density']:.2f}")
            self._visualize_maze(maze)
            input("Press Enter for next...")
    
    def _visualize_maze(self, maze):
        """Print single maze"""
        grid = maze['grid']
        start = maze['start_pos']
        target = maze['target_pos']
        
        print("  ", end="")
        for j in range(self.grid_size):
            print(f"{j:2}", end="")
        print()
        
        for i in range(self.grid_size):
            print(f"{i:2} ", end="")
            for j in range(self.grid_size):
                if (i, j) == start:
                    print("S ", end="")
                elif (i, j) == target:
                    print("G ", end="")
                elif grid[i, j] == 1:
                    print("█ ", end="")
                else:
                    print(". ", end="")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate maze dataset')
    parser.add_argument('--num', type=int, default=100, 
                       help='Total number of mazes (default: 100)')
    parser.add_argument('--size', type=int, default=10,
                       help='Grid size (default: 10)')
    parser.add_argument('--test-split', type=float, default=0.3,
                       help='Test set fraction (default: 0.3)')
    parser.add_argument('--output', type=str, default='maze_dataset.pkl',
                       help='Output filename')
    parser.add_argument('--preview', action='store_true',
                       help='Show sample mazes')
    parser.add_argument('--preview-test', action='store_true',
                       help='Show samples from test set')
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = MazeDataset(grid_size=args.size)
    train, test = dataset.generate_dataset(
        total_mazes=args.num,
        test_split=args.test_split
    )
    
    # Save
    dataset.save_dataset(args.output)
    
    # Preview
    if args.preview:
        dataset.visualize_sample(num_samples=5, from_test=False)
    
    if args.preview_test:
        dataset.visualize_sample(num_samples=5, from_test=True)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"1. Dataset saved to: {args.output}")
    print(f"2. Train on {len(train)} training mazes")
    print(f"3. Test on {len(test)} unseen test mazes")
    print(f"4. Measure generalization!")
    print("="*60)
