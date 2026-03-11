import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class BallJointPendulumEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.client = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.dt = 1.0 / 240.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.steps_survived = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.dt)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.loadURDF("plane.urdf")

        point_radius = 0.001
        pole_radius = 0.015
        pole_height = 0.2

        cart_col = p.createCollisionShape(p.GEOM_SPHERE, radius=point_radius)
        cart_vis = p.createVisualShape(p.GEOM_SPHERE, radius=point_radius, rgbaColor=[0, 1, 0, 1])
        pole_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=pole_radius, height=pole_height,
                                          collisionFramePosition=[0, 0, pole_height / 2])
        pole_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=pole_radius, length=pole_height,
                                       rgbaColor=[0.8, 0, 0, 1], visualFramePosition=[0, 0, pole_height / 2])

        self.robot = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cart_col,
            baseVisualShapeIndex=cart_vis,
            basePosition=[0, 0, 0.1],
            linkMasses=[0.5],
            linkCollisionShapeIndices=[pole_col],
            linkVisualShapeIndices=[pole_vis],
            linkPositions=[[0, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, pole_height / 2]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_SPHERICAL],
            linkJointAxis=[[0, 0, 1]]
        )

        p.changeDynamics(self.robot, 0, linearDamping=0, angularDamping=0, jointDamping=0, maxJointVelocity=100)
        p.setJointMotorControlMultiDof(self.robot, 0, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1], force=[0, 0, 0])

        # 全向随机初始化 ---
        # 随机生成 Roll 和 Pitch，确保杆子向 360 度任意方向随机倾斜
        init_roll = self.np_random.uniform(-0.15, 0.15)
        init_pitch = self.np_random.uniform(-0.15, 0.15)
        init_yaw = self.np_random.uniform(-np.pi, np.pi)  # 增加随机偏航

        init_ori = p.getQuaternionFromEuler([init_roll, init_pitch, init_yaw])
        p.resetJointStateMultiDof(self.robot, 0, targetValue=init_ori)

        self.steps_survived = 0

        return self._get_obs(), {}

    def _get_obs(self):
        c_pos, _ = p.getBasePositionAndOrientation(self.robot)
        j_state = p.getJointStateMultiDof(self.robot, 0)
        euler = p.getEulerFromQuaternion(j_state[0])
        # 观测值：底座位置(2), 速度占位(2), 角度sin/cos(4), 角速度(2)
        return np.array([c_pos[0], c_pos[1], 0, 0,
                         np.sin(euler[0]), np.cos(euler[0]), np.sin(euler[1]), np.cos(euler[1]),
                         j_state[1][0], j_state[1][1]], dtype=np.float32)

    def step(self, action):
        curr_pos, _ = p.getBasePositionAndOrientation(self.robot)

        force_scale = 0.05
        new_pos = [curr_pos[0] + action[0] * force_scale,
                   curr_pos[1] + action[1] * force_scale,
                   0.1]

        p.resetBasePositionAndOrientation(self.robot, new_pos, [0, 0, 0, 1])
        p.stepSimulation()

        obs = self._get_obs()

        self.steps_survived += 1

        # --- 强化奖励机制 ---
        # 1. 姿态奖励：使用 cos 值的指数，倾斜一点点奖励就大幅下降
        upright = (obs[5] + obs[7]) / 2.0
        reward_upright = np.power(max(0, upright), 10)

        # 2. 距离惩罚：不能跑离中心太远
        dist_sq = obs[0] ** 2 + obs[1] ** 2
        reward_dist = -0.03 * dist_sq

        # 3. 速度惩罚：防止底座疯狂抖动
        reward_vel = -0.005 * (obs[8] ** 2 + obs[9] ** 2)

        reward = reward_upright + reward_dist + reward_vel

        reward_time = 0.02  # 您可以调整这个数值

        # 修改这一行：综合奖励（加上 reward_time）
        reward = reward_upright + reward_dist + reward_vel + reward_time * reward_time * 50

        # 结束条件：角度超过约 45 度 (cos 0.7) 或跑得太远
        done = bool(upright < 0.7 or dist_sq > 4.0)

        return obs, reward, done, False, {}