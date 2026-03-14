# inverted_pendulum_env.py
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config


class InvertedPendulum3D(gym.Env):
    def __init__(self, render=True):
        super().__init__()
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)

        # 动作空间：底座在 X, Y 轴的位移增量 (速度)
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # 记录底座当前位置
        self.base_pos = np.array([0.0, 0.0, 0.0])
        #上一步的速度
        self.last_action = np.zeros(2, dtype=np.float32)

        self.steps = 0  # 记录当前episode的步数
        self.max_steps = 1000  # 最大步数，防止无限运行

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(config.TIME_STEP)

        # 1. 创建底座 (质量为0，不受外界力影响)
        b_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01], rgbaColor=[0.2, 0.2, 0.2, 1])
        self.base_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                         baseVisualShapeIndex=b_v, basePosition=[0, 0, 0])

        # 2. 创建杆子
        pv_id = p.createVisualShape(p.GEOM_CAPSULE, radius=config.POLE_RADIUS, length=config.POLE_LENGTH,
                                    rgbaColor=[1, 0, 0, 1])
        pc_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=config.POLE_RADIUS, height=config.POLE_LENGTH)

        # 初始位置随机偏移 (模拟杆子初始就不稳)
        rand_x = np.random.uniform(-0.05, 0.05)
        rand_y = np.random.uniform(-0.05, 0.05)
        self.pole_id = p.createMultiBody(baseMass=config.POLE_MASS, baseCollisionShapeIndex=pc_id,
                                         baseVisualShapeIndex=pv_id,
                                         basePosition=[rand_x, rand_y, config.POLE_LENGTH / 2])

        # 3. 约束：点对点约束 (球向铰链)
        # 将杆子的底部 ([0,0,-L/2]) 连到底座中心
        self.constraint_id = p.createConstraint(self.base_id, -1, self.pole_id, -1,
                                                p.JOINT_POINT2POINT, [0, 0, 1], [0, 0, 0],
                                                [0, 0, -config.POLE_LENGTH / 2])

        # 彻底关闭杆子的一切阻尼和摩擦
        p.changeDynamics(self.pole_id, -1, linearDamping=0, angularDamping=0, jointLowerLimit=0, jointUpperLimit=0)

        self.base_pos = np.array([0.0, 0.0, 0.0])
        self.last_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 获取底座位置
        b_pos, _ = p.getBasePositionAndOrientation(self.base_id)
        b_vel, _ = p.getBaseVelocity(self.base_id)
        # 获取杆子姿态
        _, p_orn = p.getBasePositionAndOrientation(self.pole_id)
        _, p_w = p.getBaseVelocity(self.pole_id)
        euler = p.getEulerFromQuaternion(p_orn)
        return np.array([b_pos[0], b_pos[1], euler[0], euler[1], b_vel[0], b_vel[1], p_w[0], p_w[1]], dtype=np.float32)

    def step(self, action):
        # 核心逻辑：底座不受力，直接改变位置 (Kinematic Control)
        # 这样底座移动时，杆子会因为惯性向反方向倒
        self.base_pos[0] += action[0]
        self.base_pos[1] += action[1]

        # 瞬间移动底座，不通过物理引擎算力，从而实现“质量无穷大”的效果
        p.resetBasePositionAndOrientation(self.base_id, [self.base_pos[0], self.base_pos[1], 0], [0, 0, 0, 1])

        p.stepSimulation()

        obs = self._get_obs()

        self.steps += 1
        step_bonus = 0.03 * self.steps

        distance = np.linalg.norm([obs[0] , obs[1]])
        # 奖励：角度偏差平方越小奖励越高
        angle_diff = np.sum(np.square(obs[2:4]))

        velocity_magnitude = np.linalg.norm(action)

        acceleration = (action - self.last_action) / config.TIME_STEP
        acc_magnitude = np.linalg.norm(acceleration)  # 加速度的大小（模长）

        reward = 2.0 - 8.0 * angle_diff - 1.8 * velocity_magnitude - 0.05 * acc_magnitude + 3.0 * step_bonus - 0.1 *distance

        self.last_action = action.copy()

        # 终止条件
        terminated = bool(angle_diff > 1.0 or abs(obs[0]) > 5 or abs(obs[1]) > 5)
        return obs, reward, terminated, False, {}