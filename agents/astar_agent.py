# FILE: agents/astar_agent.py
import heapq
import numpy as np

class AStarAgent:
    def __init__(self):
        """Initialize the agent"""
        self.path = []
        self.path_index = 0
    
    def get_action(self, env):
        """Get next action using A* pathfinding"""
        cube_pos = env.get_cube_grid_pos()
        target_pos = env.get_target_grid_pos()
        grid = env.get_grid_state()
        
        # If we're at the goal, just return any action (episode will end)
        if cube_pos == target_pos:
            return 'forward'
        
        # Replan if we don't have a path or reached end of path
        if not self.path or self.path_index >= len(self.path) - 1:
            self.path = self.astar(grid, cube_pos, target_pos)
            self.path_index = 0
            
            # If still no path after replanning, random action
            if not self.path or len(self.path) <= 1:
                return np.random.choice(['forward', 'backward', 'left', 'right'])
        
        # Get next position in path
        current = cube_pos
        next_pos = self.path[self.path_index + 1]
        self.path_index += 1
        
        # Convert position difference to action
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx > 0:
            return 'forward'
        elif dx < 0:
            return 'backward'
        elif dy > 0:
            return 'left'
        elif dy < 0:
            return 'right'
        else:
            return 'forward'
    
    def astar(self, grid, start, goal):
        """A* pathfinding algorithm"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid.shape[0] and 
                    0 <= ny < grid.shape[1] and 
                    grid[nx, ny] != 1):
                    neighbors.append((nx, ny))
            return neighbors
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []


class DynamicAStarAgent(AStarAgent):
    """A* that replans every N steps to handle dynamic environments"""
    def __init__(self, replan_frequency=3):
        super().__init__()
        self.replan_frequency = replan_frequency
        self.steps_since_replan = 0
        self.replan_count = 0  # NEW: Track how many times we replan
    
    def get_action(self, env):
        self.steps_since_replan += 1
        
        # Force replan every N steps
        if self.steps_since_replan >= self.replan_frequency:
            self.path = []
            self.steps_since_replan = 0
            self.replan_count += 1  # NEW
            print(f"    [Dynamic A*] Replanning #{self.replan_count} at step {env.step_count}")  # NEW
        
        return super().get_action(env)