"""
Interactive Maze Builder & Debug Tool - V2

NEW FEATURES:
- Bulk wall commands (line, rect, hline, vline)
- Better help system
- Input validation

Usage: python -m scripts.maze_builder_v2 --build
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from envs.maze_templates import get_maze_template
from collections import deque


class MazeBuilder:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.start_pos = (0, 0)
        self.target_pos = (grid_size-1, grid_size-1)
    
    def visualize(self, show_path=False):
        """Print the maze with nice formatting"""
        print("\n" + "="*60)
        print(f"MAZE (Size: {self.grid_size}×{self.grid_size})")
        print("="*60)
        
        # Header
        print("    ", end="")
        for j in range(self.grid_size):
            print(f"{j:2}", end=" ")
        print()
        print("   " + "─"*(self.grid_size*3 + 1))
        
        # Calculate path if requested
        path_positions = set()
        if show_path:
            path = self.find_path()
            if path:
                path_positions = set(path)
        
        # Grid content
        for i in range(self.grid_size):
            print(f"{i:2} │", end=" ")
            for j in range(self.grid_size):
                pos = (i, j)
                
                if pos == self.start_pos and pos == self.target_pos:
                    char = "⊛"
                elif pos == self.start_pos:
                    char = "S"
                elif pos == self.target_pos:
                    char = "G"
                elif self.grid[i, j] == 1:
                    char = "█"
                elif pos in path_positions:
                    char = "·"
                else:
                    char = " "
                
                print(f"{char:2}", end=" ")
            print("│")
        
        print("   " + "─"*(self.grid_size*3 + 1))
        print("Legend: S=Start, G=Goal, █=Wall, ·=Path")
    
    def get_statistics(self):
        """Calculate maze statistics"""
        total_cells = self.grid_size * self.grid_size
        wall_cells = np.sum(self.grid == 1)
        
        manhattan = abs(self.target_pos[0] - self.start_pos[0]) + \
                    abs(self.target_pos[1] - self.start_pos[1])
        
        path = self.find_path()
        path_length = len(path) - 1 if path else -1
        
        print("\n" + "─"*60)
        print("STATISTICS")
        print("─"*60)
        print(f"Start:          {self.start_pos}")
        print(f"Goal:           {self.target_pos}")
        print(f"Manhattan Dist: {manhattan} steps")
        print(f"Optimal Path:   {path_length} steps" + (" ✗ NO PATH!" if path_length == -1 else " ✓"))
        print(f"Wall Density:   {wall_cells}/{total_cells} ({wall_cells/total_cells*100:.1f}%)")
        
        if path_length > 0:
            complexity = path_length / manhattan
            print(f"Complexity:     {complexity:.2f}× (path/manhattan)")
        print("─"*60)
        
        return {'solvable': path_length > 0, 'path_length': path_length}
    
    def find_path(self):
        """BFS pathfinding"""
        if self.grid[self.start_pos] == 1 or self.grid[self.target_pos] == 1:
            return None
        
        visited = set()
        queue = deque([(self.start_pos, [self.start_pos])])
        visited.add(self.start_pos)
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == self.target_pos:
                return path
            
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    self.grid[nx, ny] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return None
    
    def export_template(self, name, description):
        """Export as Python code"""
        walls = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i, j] == 1]
        
        print("\n" + "="*60)
        print("PYTHON CODE - Copy to maze_templates.py")
        print("="*60)
        code = f"""
'{name}': {{
    'description': '{description}',
    'grid_size': {self.grid_size},
    'walls': {walls},
    'start': {self.start_pos},
    'target': {self.target_pos}
}},"""
        print(code)
        print("="*60)
    
    def add_wall(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[x, y] = 1
    
    def remove_wall(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[x, y] = 0
    
    def set_start(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.start_pos = (x, y)
    
    def set_target(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.target_pos = (x, y)
    
    def clear(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)


def interactive_builder():
    """Interactive maze building"""
    print("="*60)
    print("MAZE BUILDER - Interactive Mode")
    print("="*60)
    print("Type 'help' for commands\n")
    
    builder = MazeBuilder(grid_size=10)
    builder.visualize()
    
    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue
            
            action = cmd[0].lower()
            
            if action in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            elif action == 'help':
                print("\n" + "="*60)
                print("COMMANDS")
                print("="*60)
                print("\n📦 Single Walls:")
                print("  wall <x> <y>              - Add one wall")
                print("  remove <x> <y>            - Remove wall")
                print("\n📏 Bulk Walls:")
                print("  hline <row> <col1> <col2> - Horizontal line")
                print("  vline <row1> <row2> <col> - Vertical line")
                print("  line <x1> <y1> <x2> <y2>  - Any line")
                print("  rect <x1> <y1> <x2> <y2>  - Filled rectangle")
                print("\n🎯 Positions:")
                print("  start <x> <y>             - Set start (S)")
                print("  target <x> <y>            - Set goal (G)")
                print("\n🔍 View:")
                print("  show                      - Refresh display")
                print("  path                      - Show with path overlay")
                print("  stats                     - Show statistics")
                print("\n💾 Other:")
                print("  export <name>             - Generate Python code")
                print("  clear                     - Remove all walls")
                print("  quit                      - Exit")
                print("\nExamples:")
                print("  hline 5 0 9     → Horizontal wall across row 5")
                print("  vline 0 5 3     → Vertical wall from rows 0-5 at col 3")
                print("  rect 2 2 4 4    → 3×3 block of walls")
                print("="*60)
            
            elif action == 'wall' and len(cmd) == 3:
                x, y = int(cmd[1]), int(cmd[2])
                builder.add_wall(x, y)
                print(f"✓ Wall at ({x},{y})")
                builder.visualize()
            
            elif action == 'hline' and len(cmd) == 4:
                x, y1, y2 = int(cmd[1]), int(cmd[2]), int(cmd[3])
                for y in range(min(y1,y2), max(y1,y2)+1):
                    builder.add_wall(x, y)
                print(f"✓ Horizontal line: row {x}")
                builder.visualize()
            
            elif action == 'vline' and len(cmd) == 4:
                x1, x2, y = int(cmd[1]), int(cmd[2]), int(cmd[3])
                for x in range(min(x1,x2), max(x1,x2)+1):
                    builder.add_wall(x, y)
                print(f"✓ Vertical line: column {y}")
                builder.visualize()
            
            elif action == 'line' and len(cmd) == 5:
                x1, y1, x2, y2 = [int(c) for c in cmd[1:5]]
                # Bresenham's algorithm
                dx = abs(x2-x1)
                dy = abs(y2-y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                x, y = x1, y1
                while True:
                    builder.add_wall(x, y)
                    if x == x2 and y == y2:
                        break
                    e2 = 2*err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy
                print(f"✓ Line from ({x1},{y1}) to ({x2},{y2})")
                builder.visualize()
            
            elif action == 'rect' and len(cmd) == 5:
                x1, y1, x2, y2 = [int(c) for c in cmd[1:5]]
                for x in range(min(x1,x2), max(x1,x2)+1):
                    for y in range(min(y1,y2), max(y1,y2)+1):
                        builder.add_wall(x, y)
                print(f"✓ Rectangle added")
                builder.visualize()
            
            elif action == 'remove' and len(cmd) == 3:
                x, y = int(cmd[1]), int(cmd[2])
                builder.remove_wall(x, y)
                print(f"✓ Removed wall at ({x},{y})")
                builder.visualize()
            
            elif action == 'start' and len(cmd) == 3:
                x, y = int(cmd[1]), int(cmd[2])
                builder.set_start(x, y)
                print(f"✓ Start → ({x},{y})")
                builder.visualize()
            
            elif action == 'target' and len(cmd) == 3:
                x, y = int(cmd[1]), int(cmd[2])
                builder.set_target(x, y)
                print(f"✓ Goal → ({x},{y})")
                builder.visualize()
            
            elif action == 'show':
                builder.visualize()
            
            elif action == 'path':
                builder.visualize(show_path=True)
            
            elif action == 'stats':
                builder.visualize()
                builder.get_statistics()
            
            elif action == 'export' and len(cmd) >= 2:
                name = cmd[1]
                desc = input("Description: ")
                builder.export_template(name, desc)
            
            elif action == 'clear':
                builder.clear()
                builder.visualize()
                print("✓ Cleared")
            
            else:
                print("❌ Unknown command. Type 'help'")
        
        except (ValueError, IndexError) as e:
            print(f"❌ Invalid input. Type 'help' for usage")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--view', type=str)
    args = parser.parse_args()
    
    if args.build:
        interactive_builder()
    elif args.view:
        builder = MazeBuilder()
        builder.load_from_template(args.view)
        builder.visualize(show_path=True)
        builder.get_statistics()
    else:
        print("Usage: python -m scripts.maze_builder_v2 --build")