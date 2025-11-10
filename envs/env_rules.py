# FILE: envs/env_rules.py
import pybullet as p

def shift_wall_right_every_5(env):
    """Shift all walls one cell to the right every 5 steps
    
    This creates a dynamic environment where static planning fails.
    """
    # Only trigger every 5 steps
    if env.step_count % 5 != 0:
        return
    
    print(f"  [Rule] Walls shifting right at step {env.step_count}")
    
    # Find all wall positions
    wall_positions = []
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            if env.grid[x, y] == 1:  # If it's a wall
                wall_positions.append((x, y))
    
    if not wall_positions:
        return  # No walls to shift
    
    # Clear all walls
    for x, y in wall_positions:
        env.grid[x, y] = 0
    
    # Shift walls right (increase y by 1)
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    
    new_wall_positions = []
    for x, y in wall_positions:
        new_y = (y + 1) % env.grid_size  # Wrap around if at edge
        
        # Don't place wall on cube or target
        if (x, new_y) != cube_pos and (x, new_y) != target_pos:
            env.grid[x, new_y] = 1
            new_wall_positions.append((x, new_y))
    
    # Update PyBullet walls if they exist
    if hasattr(env, '_walls') and env._walls:
        # Remove old walls
        for wall_id in env._walls:
            try:
                p.removeBody(wall_id)
            except:
                pass  # Wall might already be removed
        env._walls = []
        
        # Create new walls at shifted positions
        grid_cell_size = (2 * env.bounds) / env.grid_size
        wall_height = 0.5
        half_extents = [grid_cell_size / 2, grid_cell_size / 2, wall_height / 2]
        
        for gx, gy in new_wall_positions:
            # Convert to bullet coordinates
            bullet_x = (gx * grid_cell_size) - env.bounds + (grid_cell_size / 2)
            bullet_y = (gy * grid_cell_size) - env.bounds + (grid_cell_size / 2)
            
            # Create wall
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                            rgbaColor=[0.5, 0.5, 0.5, 1])
            
            wall_id = p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=col_shape,
                                         baseVisualShapeIndex=vis_shape,
                                         basePosition=[bullet_x, bullet_y, wall_height / 2])
            
            env._walls.append(wall_id)