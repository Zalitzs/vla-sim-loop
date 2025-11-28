"""
Procedural Maze Generator using Classic Algorithms

Uses Prim's Algorithm and DFS to generate proper mazes
Plus random wall density for variety
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import deque
import random


class ProceduralMazeGenerator:
    """Generate mazes using proper algorithms"""
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
    
    def generate_all_mazes(self, num_prims=3, num_dfs=3, num_random=4, output_file='generated_mazes.txt'):
        """Generate diverse set of mazes"""
        
        print("="*60)
        print("PROCEDURAL MAZE GENERATOR")
        print("="*60)
        print(f"\nGenerating {self.grid_size}×{self.grid_size} mazes...")
        print(f"  - {num_prims} Prim's algorithm mazes")
        print(f"  - {num_dfs} DFS algorithm mazes")
        print(f"  - {num_random} Random density mazes")
        print()
        
        mazes = []
        
        # Prim's algorithm mazes (good branching)
        for i in range(num_prims):
            maze = self.generate_prims_maze(f'prims_maze_{i+1}')
            if maze['path_length'] > 0:  # Verify solvable
                mazes.append(maze)
                print(f"✓ {maze['name']:20s} - Path: {maze['path_length']:2d} steps - Prim's")
            else:
                print(f"✗ {maze['name']:20s} - UNSOLVABLE (regenerating...)")
                # Try again with random instead
                maze = self.generate_random_density(0.2, f'prims_fallback_{i+1}')
                if maze['path_length'] > 0:
                    mazes.append(maze)
                    print(f"  ↳ {maze['name']:20s} - Path: {maze['path_length']:2d} steps - Random")
        
        # DFS mazes (long corridors)
        for i in range(num_dfs):
            maze = self.generate_dfs_maze(f'dfs_maze_{i+1}')
            if maze['path_length'] > 0:  # Verify solvable
                mazes.append(maze)
                print(f"✓ {maze['name']:20s} - Path: {maze['path_length']:2d} steps - DFS")
            else:
                print(f"✗ {maze['name']:20s} - UNSOLVABLE (using random instead)")
                maze = self.generate_random_density(0.25, f'dfs_fallback_{i+1}')
                if maze['path_length'] > 0:
                    mazes.append(maze)
                    print(f"  ↳ {maze['name']:20s} - Path: {maze['path_length']:2d} steps - Random")
        
        # Random density mazes (varied difficulty)
        densities = [0.15, 0.20, 0.25, 0.30]
        for i, density in enumerate(densities[:num_random]):
            maze = self.generate_random_density(density, f'random_{int(density*100)}')
            if maze['path_length'] > 0:  # Only if solvable
                mazes.append(maze)
                print(f"✓ {maze['name']:20s} - Path: {maze['path_length']:2d} steps - Random {int(density*100)}%")
        
        print(f"\n✓ Generated {len(mazes)} valid mazes")
        
        # Export
        self._export_to_file(mazes, output_file)
        
        return mazes
    
    def generate_prims_maze(self, name='prims_maze'):
        """
        Prim's algorithm - creates maze with good branching
        
        Simplified version that ensures connectivity
        """
        # Start with all walls
        grid = np.ones((self.grid_size, self.grid_size), dtype=int)
        
        # Pick random starting cell (make it odd to work with wall-carving)
        if self.grid_size < 5:
            # For small grids, use simpler approach
            return self.generate_random_density(0.25, name)
        
        # Start in center-ish area
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        grid[start_x, start_y] = 0
        
        # Walls adjacent to passages
        walls = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                walls.append((nx, ny, start_x, start_y))
        
        while walls:
            # Pick random wall
            wx, wy, px, py = random.choice(walls)
            walls.remove((wx, wy, px, py))
            
            if grid[wx, wy] == 1:  # Still a wall
                # Check the cell on the other side
                dx, dy = wx - px, wy - py
                ox, oy = wx + dx, wy + dy
                
                if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                    if grid[ox, oy] == 1:  # Other side is unvisited
                        # Carve passage
                        grid[wx, wy] = 0
                        grid[ox, oy] = 0
                        
                        # Add neighboring walls
                        for ddx, ddy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = ox + ddx, oy + ddy
                            if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                                grid[nx, ny] == 1 and (nx, ny, ox, oy) not in walls):
                                walls.append((nx, ny, ox, oy))
        
        # Ensure start and goal are accessible
        start_pos = (0, 0)
        target_pos = (self.grid_size - 1, self.grid_size - 1)
        grid[start_pos] = 0
        grid[target_pos] = 0
        
        # Clear path near start and goal to ensure connectivity
        for i in range(min(2, self.grid_size)):
            for j in range(min(2, self.grid_size)):
                grid[i, j] = 0
                grid[self.grid_size - 1 - i, self.grid_size - 1 - j] = 0
        
        return self._make_template(name, grid, 'Prim\'s algorithm - branching paths')
    
    def generate_dfs_maze(self, name='dfs_maze'):
        """
        DFS - creates maze with long corridors
        
        Simplified to ensure solvability
        """
        # Start with empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Add random walls (but not too many)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < 0.3:  # 30% walls
                    grid[i, j] = 1
        
        # Ensure start and goal are clear
        start_pos = (0, 0)
        target_pos = (self.grid_size - 1, self.grid_size - 1)
        grid[start_pos] = 0
        grid[target_pos] = 0
        
        # Clear small area around start and goal
        for i in range(min(2, self.grid_size)):
            for j in range(min(2, self.grid_size)):
                grid[i, j] = 0
                grid[self.grid_size - 1 - i, self.grid_size - 1 - j] = 0
        
        # Verify solvable, if not, carve a path
        if not self._find_path(grid, start_pos, target_pos):
            # Force carve a path using DFS
            visited = set()
            path = []
            self._dfs_carve_path(grid, start_pos[0], start_pos[1], target_pos, visited, path)
        
        return self._make_template(name, grid, 'DFS-style - winding corridors')
    
    def generate_random_density(self, density=0.2, name=None):
        """Random wall placement with given density"""
        if name is None:
            name = f'random_{int(density*100)}'
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Random walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < density:
                    grid[i, j] = 1
        
        # Ensure solvability by clearing a path if needed
        start_pos = (0, 0)
        target_pos = (self.grid_size - 1, self.grid_size - 1)
        grid[start_pos] = 0
        grid[target_pos] = 0
        
        # Check if solvable, if not, reduce some walls
        max_attempts = 5
        for attempt in range(max_attempts):
            if self._find_path(grid, start_pos, target_pos):
                break
            
            # Remove random walls to make solvable
            for _ in range(self.grid_size):
                rx = random.randint(0, self.grid_size - 1)
                ry = random.randint(0, self.grid_size - 1)
                grid[rx, ry] = 0
        
        desc = f'Random walls - {int(density*100)}% density'
        return self._make_template(name, grid, desc)
    
    def _dfs_carve_path(self, grid, x, y, target, visited, path):
        """DFS to carve a path if maze is unsolvable"""
        if (x, y) == target:
            # Reached goal, carve the path
            for px, py in path:
                grid[px, py] = 0
            return True
        
        if (x, y) in visited:
            return False
        
        visited.add((x, y))
        path.append((x, y))
        
        # Try all directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self._dfs_carve_path(grid, nx, ny, target, visited, path):
                    return True
        
        path.pop()
        return False
    
    def _add_frontier(self, grid, x, y, frontier):
        """Add frontier cells (2 steps away in cardinal directions)"""
        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if grid[nx, ny] == 1 and (nx, ny) not in frontier:
                    frontier.append((nx, ny))
    
    def _make_template(self, name, grid, description):
        """Create template dict from grid"""
        start_pos = (0, 0)
        target_pos = (self.grid_size - 1, self.grid_size - 1)
        
        # Ensure start and target are free
        grid[start_pos] = 0
        grid[target_pos] = 0
        
        # Calculate path
        path = self._find_path(grid, start_pos, target_pos)
        path_length = len(path) - 1 if path else -1
        
        # Get walls
        walls = [(i, j) for i in range(self.grid_size) 
                for j in range(self.grid_size) if grid[i, j] == 1]
        
        return {
            'name': name,
            'description': description,
            'grid_size': self.grid_size,
            'grid': grid,
            'start_pos': start_pos,
            'target_pos': target_pos,
            'walls': walls,
            'path_length': path_length
        }
    
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
    
    def _export_to_file(self, mazes, filename):
        """Export as Python code"""
        with open(filename, 'w') as f:
            f.write("# AUTO-GENERATED MAZES - Procedural Generation\n")
            f.write("# Generated using Prim's, DFS, and Random algorithms\n")
            f.write("# Copy these into your maze_templates.py\n\n")
            
            for maze in mazes:
                f.write(f"    '{maze['name']}': {{\n")
                f.write(f"        'description': '{maze['description']}',\n")
                f.write(f"        'grid_size': {maze['grid_size']},\n")
                f.write(f"        'walls': {maze['walls']},\n")
                f.write(f"        'start': {maze['start_pos']},\n")
                f.write(f"        'target': {maze['target_pos']}\n")
                f.write(f"    }},\n\n")
        
        print(f"\n✓ Exported to: {filename}")
        print(f"  → Copy contents into maze_templates.py")
    
    def visualize_maze(self, maze_data):
        """Print maze"""
        grid = maze_data['grid']
        start = maze_data['start_pos']
        target = maze_data['target_pos']
        
        print(f"\n{maze_data['name']}: {maze_data['description']}")
        print(f"Path length: {maze_data['path_length']} steps\n")
        
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
    
    parser = argparse.ArgumentParser(description='Procedural Maze Generator')
    parser.add_argument('--size', type=int, default=10, help='Grid size')
    parser.add_argument('--prims', type=int, default=3, help='Number of Prim\'s mazes')
    parser.add_argument('--dfs', type=int, default=3, help='Number of DFS mazes')
    parser.add_argument('--random', type=int, default=4, help='Number of random mazes')
    parser.add_argument('--output', type=str, default='generated_mazes.txt')
    parser.add_argument('--preview', action='store_true', help='Preview all mazes')
    
    args = parser.parse_args()
    
    generator = ProceduralMazeGenerator(grid_size=args.size)
    mazes = generator.generate_all_mazes(
        num_prims=args.prims,
        num_dfs=args.dfs,
        num_random=args.random,
        output_file=args.output
    )
    
    if args.preview:
        print("\n" + "="*60)
        print("MAZE PREVIEWS")
        print("="*60)
        for maze in mazes:
            generator.visualize_maze(maze)
            input("\nPress Enter for next maze...")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"✓ Generated {len(mazes)} mazes")
    print(f"✓ Saved to: {args.output}")
    print("\nAlgorithms used:")
    print("  - Prim's: Good branching, multiple paths")
    print("  - DFS: Long winding corridors")
    print("  - Random: Varied difficulty levels")
    print("="*60)