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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # 记录底座当前位置
        self.base_pos = np.array([0.0, 0.0, 0.0])
        #上一步的速度
        self.last_action = np.zeros(2, dtype=np.float32)

        self.steps = 0  # 记录当前episode的步数
        self.max_steps = 1000  # 最大步数，防止无限运行

        self.traj_points = []
        self.max_traj_len = 50

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
        init_orn = p.getQuaternionFromEuler([np.pi + rand_x, rand_y, 0])

        # 向下初始化时，杆子的几何中心应该在底座下方 (0.05 - 半个杆长)
        self.pole_id = p.createMultiBody(baseMass=config.POLE_MASS, baseCollisionShapeIndex=pc_id,
                                         baseVisualShapeIndex=pv_id,
                                         basePosition=[0, 0, 0.05 - config.POLE_LENGTH / 2],
                                         baseOrientation=init_orn)

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
        b_pos, _ = p.getBasePositionAndOrientation(self.base_id)
        b_vel, _ = p.getBaseVelocity(self.base_id)

        # 重心的世界坐标 Z
        p_pos, p_orn = p.getBasePositionAndOrientation(self.pole_id)
        _, p_w = p.getBaseVelocity(self.pole_id)

        # 欧拉角
        euler = p.getEulerFromQuaternion(p_orn)

        return np.array([
            b_pos[0], b_pos[1],
            p_pos[2],  # 新加的分量：重心高度 Z
            euler[0], euler[1],
            b_vel[0], b_vel[1],
            p_w[0], p_w[1]
        ], dtype=np.float32)

    def step(self, action):
        action = np.array(action, dtype=np.float32)

        # 核心逻辑：底座不受力，直接改变位置 (Kinematic Control)
        # 这样底座移动时，杆子会因为惯性向反方向倒
        self.base_pos[0] += action[0]
        self.base_pos[1] += action[1]

        # 瞬间移动底座，不通过物理引擎算力，从而实现“质量无穷大”的效果
        p.resetBasePositionAndOrientation(self.base_id, [self.base_pos[0], self.base_pos[1], 0], [0, 0, 0, 1])

        p.stepSimulation()

        #轨迹
        p_pos, p_orn = p.getBasePositionAndOrientation(self.pole_id)

        top_pos, _ = p.multiplyTransforms(p_pos, p_orn, [0, 0, config.POLE_LENGTH / 2], [0, 0, 0, 1])

        self.traj_points.append(top_pos)
        if len(self.traj_points) > 1:
            p.addUserDebugLine(self.traj_points[-2], self.traj_points[-1],
                               lineColorRGB=[1, 0, 1],
                               lineWidth=2,
                               lifeTime=20.0)

        if len(self.traj_points) > self.max_traj_len:
            self.traj_points.pop(0)

        obs = self._get_obs()

        self.steps += 1
        step_bonus = 0.03 * self.steps

        distance = np.linalg.norm([obs[0] , obs[1]])
        # 奖励：角度偏差平方越小奖励越高
        current_pole_z = obs[2]

        velocity_magnitude = np.linalg.norm(action)

        acceleration = (action - self.last_action) / config.TIME_STEP
        acc_magnitude = np.linalg.norm(acceleration)  # 加速度的大小（模长）

        high = current_pole_z if current_pole_z > 0.2 else -1

        reward = 1.0 * current_pole_z #- 0.02 * distance - 0.0001 * velocity_magnitude - 0.005 * acc_magnitude + 0.01 * step_bonus

        self.last_action = action.copy()

        # 终止条件
        terminated = bool(abs(obs[0]) > 5 or abs(obs[1]) > 5)
        return obs, reward, terminated, False, {}