import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from envs.grid_bullet_world import GridBulletWorld
from envs.env_rules import shift_wall_right_every_5

# Setup color map: 0 = empty, 1 = wall, 2 = cube, 3 = target
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

env = GridBulletWorld(gui=True,dynamic_rules=[shift_wall_right_every_5])
obs = env.reset()

print("\nControls: forward, backward, left, right, or q to quit\n")

while True:
    grid = env.get_grid_state()
    cube_pos = env.get_cube_grid_pos()
    target_pos = env.get_target_grid_pos()
    
    from pybullet import getBasePositionAndOrientation
    raw_pos, _ = getBasePositionAndOrientation(env._cube)
    print(f"Cube raw (continuous): {raw_pos[:2]}")
    print(f"Cube grid (discrete): {cube_pos}")

    print(f"\nStep: {env.step_count}")
    print(f"Cube:  {cube_pos}")
    print(f"Target: {target_pos}")
    print(grid)

    # Visual update
    plt.imshow(np.flipud(np.fliplr(grid)), cmap=cmap, norm=norm)
    plt.title(f"Step {env.step_count}")
    plt.pause(0.1)
    plt.clf()

    # Manual input
    action = input("Enter action: ").strip().lower()
    if action == "q":
        break

    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.3f}  Success: {info.get('success')}")

    if done:
        print("Episode finished.")
        break

plt.close()
