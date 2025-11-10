import numpy as np
import pybullet as p
from envs.bullet_world import BulletWorld
from envs.env_rules import shift_wall_right_every_5

class GridBulletWorld(BulletWorld):
    def __init__(self, grid_size=20, dynamic_rules=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)  # 0=empty, 1=wall
        self.dynamic_rules = dynamic_rules or []
        self.step_count = 0
    
    def get_cube_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self._cube)
        return pos[0], pos[1]

    def get_target_pos(self):
        return self._target_xy
    
    def get_cube_grid_pos(self):
        return self.discretize(self.get_cube_pos())

    def get_target_grid_pos(self):
        return self.discretize(self.get_target_pos())
    
    def reset(self, maze_template=None, seed=None):
        """Reset environment with optional maze template"""
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        
        if maze_template is not None:
            # Connect to PyBullet if needed
            if self._client is None:
                self.connect()
            
            # DON'T call super().reset() - set positions directly
            self._step_count = 0
            
            # Use template
            self.grid = maze_template['grid'].copy()
            
            # Set positions from template
            self._set_cube_to_grid_pos(maze_template['start_pos'])
            self._set_target_to_grid_pos(maze_template['target_pos'])
            
            # Spawn walls
            self._spawn_walls_from_grid()
            
            # Get observation
            obs = self._obs()
            
        else:
            # Original random generation
            obs = super().reset()
            self.grid[:] = 0
            
            num_walls = np.random.randint(5, 10)
            for _ in range(num_walls):
                x, y = np.random.randint(0, self.grid_size, size=2)
                self.grid[x, y] = 1
            
            cx, cy = self.discretize(self.get_cube_pos())
            tx, ty = self.discretize(self.get_target_pos())
            self.grid[cx, cy] = 0
            self.grid[tx, ty] = 0
            
            self._spawn_walls_from_grid()
        
        return obs
    
    def _set_cube_to_grid_pos(self, grid_pos):
        """Move cube to specific grid position
        
        Args:
            grid_pos: Tuple (x, y) in grid coordinates
        """
        grid_cell_size = (2 * self.bounds) / self.grid_size
        bullet_x = (grid_pos[0] * grid_cell_size) - self.bounds + (grid_cell_size / 2)
        bullet_y = (grid_pos[1] * grid_cell_size) - self.bounds + (grid_cell_size / 2)

        p.resetBasePositionAndOrientation(
            self._cube, 
            [bullet_x, bullet_y, 0.025], 
            [0, 0, 0, 1]
        )

    def _set_target_to_grid_pos(self, grid_pos):
        """Move target to specific grid position
        
        Args:
            grid_pos: Tuple (x, y) in grid coordinates
        """
        grid_cell_size = (2 * self.bounds) / self.grid_size
        bullet_x = (grid_pos[0] * grid_cell_size) - self.bounds + (grid_cell_size / 2)
        bullet_y = (grid_pos[1] * grid_cell_size) - self.bounds + (grid_cell_size / 2)
        
        self._target_xy = (bullet_x, bullet_y)
        
        import pybullet as p
        p.resetBasePositionAndOrientation(
            self._target_body, 
            [bullet_x, bullet_y, 0.03], 
            [0, 0, 0, 1]
        )

    def _spawn_walls_from_grid(self):
        """Spawn PyBullet walls based on current grid state"""
        import pybullet as p
        
        # Remove old walls if they exist
        if not hasattr(self, '_walls'):
            self._walls = []
        
        for wall_id in self._walls:
            try:
                p.removeBody(wall_id)
            except:
                pass
        self._walls = []
        
        # Find all wall positions in grid
        wall_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == 1:  # Wall
                    wall_positions.append((x, y))
        
        # Spawn walls in PyBullet
        grid_cell_size = (2 * self.bounds) / self.grid_size
        wall_height = 0.5
        half_extents = [grid_cell_size / 2, grid_cell_size / 2, wall_height / 2]
        
        for gx, gy in wall_positions:
            # Convert to bullet coordinates
            bullet_x = (gx * grid_cell_size) - self.bounds + (grid_cell_size / 2)
            bullet_y = (gy * grid_cell_size) - self.bounds + (grid_cell_size / 2)
            
            # Create wall
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_shape = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=half_extents,
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[bullet_x, bullet_y, wall_height / 2]
            )
            
            self._walls.append(wall_id)
        
    def is_wall_at_grid(self, gx, gy):
        """Check if there's a wall at the given grid position"""
        if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
            return True  # Out of bounds = wall
        return self.grid[gx, gy] == 1

    def step(self, action):
        # Get current grid position
        cx, cy = self.get_cube_grid_pos()
        
        # Calculate next grid position
        next_gx, next_gy = cx, cy
        if action == "forward":
            next_gx += 1
        elif action == "backward":
            next_gx -= 1
        elif action == "left":
            next_gy += 1
        elif action == "right":
            next_gy -= 1
        
        # Check bounds and walls
        if (next_gx < 0 or next_gx >= self.grid_size or 
            next_gy < 0 or next_gy >= self.grid_size or 
            self.is_wall_at_grid(next_gx, next_gy)):
            # Don't move
            self.step_count += 1
            self.apply_dynamic_rules()
            obs = self._obs()
            done = (obs["dist"] <= self.success_thresh) or (self.step_count >= self.max_steps)
            reward = -obs["dist"]
            info = {"success": obs["dist"] <= self.success_thresh}
            return obs, reward, done, info
        
        # Convert grid position to bullet coordinates
        grid_cell_size = (2 * self.bounds) / self.grid_size
        bullet_x = (next_gx * grid_cell_size) - self.bounds + (grid_cell_size / 2)
        bullet_y = (next_gy * grid_cell_size) - self.bounds + (grid_cell_size / 2)
        
        # Move cube directly to exact grid position
        import pybullet as p
        p.resetBasePositionAndOrientation(self._cube, [bullet_x, bullet_y, 0.025], [0,0,0,1])
        
        self.step_count += 1
        self.apply_dynamic_rules()
        obs = self._obs()
        reward = -obs["dist"]
        
        # GRID-BASED SUCCESS CHECK (override parent's distance check)
        cube_grid = self.get_cube_grid_pos()
        target_grid = self.get_target_grid_pos()
        
        if cube_grid == target_grid:
            done = True
            info = {"success": True}
        else:
            done = self.step_count >= self.max_steps
            info = {"success": False}

        return obs, reward, done, info

    def apply_dynamic_rules(self):
        for rule in self.dynamic_rules:
            rule(self)

    def get_grid_state(self):
        """Return full grid with cube and target overlayed"""
        g = self.grid.copy()
        cx, cy = self.discretize(self.get_cube_pos())
        tx, ty = self.discretize(self.get_target_pos())
        g[cx, cy] = 2  # Cube
        g[tx, ty] = 3  # Target
        return g

    def discretize(self, pos):
        """Map bullet (x, y) to grid coordinates using floor division for consistent boundaries"""
        grid_cell_size = (2 * self.bounds) / self.grid_size
        # Use floor division instead of round to get consistent cell boundaries
        x = int((pos[0] + self.bounds) / grid_cell_size)
        y = int((pos[1] + self.bounds) / grid_cell_size)
        return np.clip(x, 0, self.grid_size - 1), np.clip(y, 0, self.grid_size - 1)

