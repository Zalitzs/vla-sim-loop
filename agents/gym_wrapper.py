import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MazeEnvGym(gym.Env):
    """Gym wrapper for GridBulletWorld"""
    
    def __init__(self, grid_bullet_world):
        super().__init__()
        self.env = grid_bullet_world
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        self.action_map = {0: 'forward', 1: 'backward', 2: 'left', 3: 'right'}
        
        # SIMPLIFIED observation: distance + blocked directions + relative goal position
        # [dx_to_goal, dy_to_goal, manhattan_dist, wall_forward, wall_back, wall_left, wall_right]
        self.observation_space = spaces.Box(
            low=-grid_bullet_world.grid_size, 
            high=grid_bullet_world.grid_size, 
            shape=(7,),  # Much smaller!
            dtype=np.float32
        )
    
    def _get_obs(self):
        """Convert environment state to observation vector - SIMPLIFIED"""
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
        
        obs = np.array([
            dx, dy, manhattan,
            wall_forward, wall_backward, wall_left, wall_right
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
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
        
        obs_dict, reward, done, info = self.env.step(action_str)
        
        # Check if agent moved
        new_pos = self.env.get_cube_grid_pos()
        moved = (prev_pos != new_pos)
        
        # SHAPED REWARD: reward progress, not just final distance
        new_dist = obs_dict['dist']
        
        # Reward for getting closer (positive) or further (negative)
        distance_delta = prev_dist - new_dist
        shaped_reward = distance_delta * 10  # Scale up the signal
        
        # PENALTY for hitting walls (not moving when trying to)
        if not moved:
            shaped_reward -= 1.0  # Strong penalty for invalid moves
        
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