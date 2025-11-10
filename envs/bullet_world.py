import pybullet as p
import pybullet_data
import numpy as np
import math
import random

class BulletWorld:
    def __init__(self, gui=False, step_size=0.10, success_thresh=0.05, max_steps=40, bounds=0.8, min_start_dist=0.30, action_jitter=0.0):
        self.gui = gui
        self.step_size = float(step_size)
        self.success_thresh = float(success_thresh)
        self.max_steps = int(max_steps)
        self.bounds = float(bounds)
        self.min_start_dist = float(min_start_dist)
        self.action_jitter = float(action_jitter)
        self._client = None
        self._cube = None
        self._step_count = 0
        self._target_xy = (0.0, 0.0)

    def connect(self):
        if self._client is not None: 
            return
        self._client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        # simple visual marker for the target: small sphere
        self._target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1,0,0,1])
        self._target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=self._target_vis, basePosition=[0,0,0.03])
        # the cube
        self._cube = p.loadURDF("cube_small.urdf", basePosition=[0,0,0.025])

    def disconnect(self):
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    def _sample_xy(self, min_r=0.1, max_r=None):
        if max_r is None:
            max_r = self.bounds
        grid = 0.1  # 10 cm grid
        for _ in range(1000):
            x = round(random.uniform(-self.bounds, self.bounds) / grid) * grid
            y = round(random.uniform(-self.bounds, self.bounds) / grid) * grid
            r = math.hypot(x, y)
            if r >= min_r and r <= max_r:
                return (x, y)
        return (0.0, 0.0)

    def reset(self, difficulty="medium"):
        if self._client is None:
            self.connect()
        self._step_count = 0
        # place target
        self._target_xy = self._sample_xy(min_r=0.2, max_r=self.bounds*0.9)
        p.resetBasePositionAndOrientation(self._target_body, [self._target_xy[0], self._target_xy[1], 0.03], [0,0,0,1])
        # place cube far enough from target
        while True:
            cube_xy = self._sample_xy(min_r=0.05, max_r=self.bounds*0.9)
            if self._dist_xy(cube_xy, self._target_xy) >= (0.6 if difficulty=="medium" else 0.3):
                break
        p.resetBasePositionAndOrientation(self._cube, [cube_xy[0], cube_xy[1], 0.025], [0,0,0,1])
        return self._obs()

    def _dist_xy(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _obs(self):
        cube_pos, _ = p.getBasePositionAndOrientation(self._cube)
        cx, cy = cube_pos[0], cube_pos[1]
        tx, ty = self._target_xy
        dist = self._dist_xy((cx, cy), (tx, ty))
        return {
            "cube": (cx, cy),
            "target": (tx, ty),
            "dist": dist,
            "step": self._step_count
        }

    def step(self, action):
        self._step_count += 1
        dx, dy = {
            "forward":  ( self.step_size, 0.0),
            "backward": (-self.step_size, 0.0),
            "left":     (0.0,  self.step_size),
            "right":    (0.0, -self.step_size),
        }.get(action, (0.0,0.0))
        # jitter (relative)
        if self.action_jitter > 0.0:
            jitter = 1.0 + np.random.uniform(-self.action_jitter, self.action_jitter)
            dx *= jitter
            dy *= jitter
        cube_pos, _ = p.getBasePositionAndOrientation(self._cube)
        nx = np.clip(cube_pos[0] + dx, -self.bounds, self.bounds)
        ny = np.clip(cube_pos[1] + dy, -self.bounds, self.bounds)
        grid = 0.1
        nx = round(nx / grid) * grid
        ny = round(ny / grid) * grid
        p.resetBasePositionAndOrientation(self._cube, [nx, ny, 0.025], [0,0,0,1])
        obs = self._obs()
        done = (obs["dist"] <= self.success_thresh) or (self._step_count >= self.max_steps)
        reward = -obs["dist"]
        info = {"success": obs["dist"] <= self.success_thresh}
        return obs, reward, done, info

    def screenshot(self, path):
        # Optional: save a top-down image
        cube_pos, _ = p.getBasePositionAndOrientation(self._cube)
        cam_target = [cube_pos[0], cube_pos[1], 0.0]
        view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=cam_target, distance=1.2, yaw=0, pitch=-89.9, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
        w,h,_,rgba, _ = p.getCameraImage(512,512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        import imageio
        imageio.imwrite(path, rgba[:,:,:3])
