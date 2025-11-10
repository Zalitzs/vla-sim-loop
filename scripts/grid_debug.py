import matplotlib.pyplot as plt
from envs.grid_bullet_world import GridBulletWorld
from envs.env_rules import shift_wall_right_every_5
import matplotlib.colors as mcolors

# 0 = empty, 1 = wall, 2 = cube, 3 = target
cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Initialize the environment with dynamic rule(s)
env = GridBulletWorld(dynamic_rules=[shift_wall_right_every_5])
env.reset()

# Run and visualize for 20 steps
for step in range(20):
    grid = env.get_grid_state()
    print(f"Step {step}")
    print(grid)

    plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
    plt.title(f"Step {step}")
    plt.pause(0.3)

    obs, reward, done, info = env.step("forward")

plt.show()
