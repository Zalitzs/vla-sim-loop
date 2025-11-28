import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MazeEnvGym(gym.Env):
    """Gym wrapper for GridBulletWorld"""
    
    def __init__(self, grid_bullet_world, history_length=5):  # Increased from 3 to 5
        super().__init__()
        self.env = grid_bullet_world
        self.history_length = history_length
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        self.action_map = {0: 'forward', 1: 'backward', 2: 'left', 3: 'right'}
        
        # OBSERVATION with position history
        # [dx, dy, manhattan, wall_forward, wall_back, wall_left, wall_right, 
        #  prev_dx1, prev_dy1, prev_dx2, prev_dy2, ...]
        obs_size = 7 + (history_length * 2)  # 7 base + 2 coords per history step
        self.observation_space = spaces.Box(
            low=-grid_bullet_world.grid_size, 
            high=grid_bullet_world.grid_size, 
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Track position history
        self.position_history = []
        self.visited_positions = set()  # Track all visited positions
    
    def _get_obs(self):
        """Convert environment state to observation vector with position history"""
        cx, cy = self.env.get_cube_grid_pos()
        tx, ty = self.env.get_target_grid_pos()
        
        # Relative position to goal
        dx = tx - cx
        dy = ty - cy
        manhattan = abs(dx) + abs(dy)
        
        # Check if each direction is blocked (1=blocked, 0=free)
        wall_forward = 1.0 if self.env.is_wall_at_grid(cx + 1, cy) else 0.0
        wall_backward = 1.0 if self.env.is_wall_at_grid(cx - 1, cy) else 0.0
        wall_left = 1.0 if self.env.is_wall_at_grid(cx, cy + 1) else 0.0
        wall_right = 1.0 if self.env.is_wall_at_grid(cx, cy - 1) else 0.0
        
        # Base observation
        base_obs = [dx, dy, manhattan, wall_forward, wall_backward, wall_left, wall_right]
        
        # Add position history (helps agent remember where it's been)
        history_obs = []
        for i in range(self.history_length):
            if i < len(self.position_history):
                prev_dx, prev_dy = self.position_history[-(i+1)]
                history_obs.extend([prev_dx, prev_dy])
            else:
                history_obs.extend([0.0, 0.0])  # Pad with zeros
        
        obs = np.array(base_obs + history_obs, dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Clear position history and visited set
        self.position_history = []
        self.visited_positions = set()
        
        # Reset underlying environment
        if options and 'maze_template' in options:
            self.env.reset(maze_template=options['maze_template'])
        else:
            self.env.reset()
        
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        """Take action in environment"""
        action_str = self.action_map[action]
        
        # Store previous distance AND position
        prev_dist = self.env._obs()['dist']
        prev_pos = self.env.get_cube_grid_pos()
        
        # Get target for history
        tx, ty = self.env.get_target_grid_pos()
        
        obs_dict, reward, done, info = self.env.step(action_str)
        
        # Update position history
        cx, cy = self.env.get_cube_grid_pos()
        dx = tx - cx
        dy = ty - cy
        self.position_history.append((dx, dy))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)  # Keep only recent history
        
        # Check if agent moved
        new_pos = self.env.get_cube_grid_pos()
        moved = (prev_pos != new_pos)
        
        # Check if revisiting a position
        revisited = new_pos in self.visited_positions
        self.visited_positions.add(new_pos)
        
        # SHAPED REWARD: reward progress, not just final distance
        new_dist = obs_dict['dist']
        
        # Reward for getting closer (positive) or further (negative)
        distance_delta = prev_dist - new_dist
        shaped_reward = distance_delta * 10  # Scale up the signal
        
        # PENALTY for hitting walls (not moving when trying to)
        if not moved:
            shaped_reward -= 1.0  # Strong penalty for invalid moves
        
        # PENALTY for revisiting same cell (helps avoid loops)
        if revisited and moved:
            shaped_reward -= 0.5  # Mild penalty for backtracking
        
        # Big bonus for reaching goal
        if info.get('success', False):
            shaped_reward += 100
        
        # Small penalty for each step (encourages efficiency)
        shaped_reward -= 0.1
        
        obs = self._get_obs()
        truncated = False  # Gymnasium requires this
        
        return obs, shaped_reward, done, truncated, info
    
    def render(self):
        """Optional: render environment"""
        pass
    
    def close(self):
        """Clean up"""
        self.env.disconnect()