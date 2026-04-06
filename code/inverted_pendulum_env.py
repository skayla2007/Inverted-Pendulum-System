# inverted_pendulum_env.py
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
import pybullet_data


class InvertedPendulum3D(gym.Env):
    def __init__(self, render=True, use_mesh=False):
        super().__init__()
        self.render_mode = render
        self.use_mesh = use_mesh # 记录是否使用模型
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)

        # 动作空间：底座在 X, Y 轴的位移增量 (速度)
        self.action_space = spaces.Box(low=-config.MAX_SPEED, high=config.MAX_SPEED, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # 记录底座当前位置
        self.base_pos = np.array([0.0, 0.0, 0.0])
        #上一步的速度
        self.last_action = np.zeros(2, dtype=np.float32)

        self.steps = 0  # 记录当前episode的步数
        self.max_steps = 1000  # 最大步数，防止无限运行

        self.traj_points = []
        self.max_traj_len = 50

        self.smoothed_action = np.zeros(2, dtype=np.float32)
        self.alpha = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(config.TIME_STEP)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #创建地面
        self.plane_id = p.loadURDF("plane.urdf")
        tex_id = p.loadTexture("image/grass_240.png")
        p.changeVisualShape(self.plane_id, -1, textureUniqueId=tex_id)

        # 1. 创建底座 (质量为0，不受外界力影响)
        b_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01], rgbaColor=[0.2, 0.2, 0.2, 1])
        self.base_id = p.createMultiBody(baseMass=config.BASE_MASS, baseCollisionShapeIndex=-1,
                                         baseVisualShapeIndex=b_v, basePosition=[0, 0, 0])

        # 2. 创建杆子
        if self.use_mesh:
            correction_euler = [1.57, 0, 0]
            correction_orn = p.getQuaternionFromEuler(correction_euler)
            pv_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName="obj/Saturn V.obj",
                meshScale=[0.5, 0.5, 0.5],
                visualFramePosition=[0, 0.57, 1.2],
                visualFrameOrientation=correction_orn
            )
        else:
            pv_id = p.createVisualShape(p.GEOM_CAPSULE, radius=config.POLE_RADIUS, length=config.POLE_LENGTH,
                                        rgbaColor=[0, 0.5, 1, 1])

        pc_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=config.POLE_RADIUS, height=config.POLE_LENGTH)

        # 初始位置随机偏移 (模拟杆子初始就不稳)
        rand_x = np.random.uniform(-config.RAND_ANGLE, config.RAND_ANGLE)
        rand_y = np.random.uniform(-config.RAND_ANGLE, config.RAND_ANGLE)
        init_orn = p.getQuaternionFromEuler([config.INIT_ANGLE + rand_x, rand_y, 0])

        # 向下初始化时，杆子的几何中心应该在底座下方 (0.05 - 半个杆长)
        self.pole_id = p.createMultiBody(baseMass=config.POLE_MASS, baseCollisionShapeIndex=pc_id,
                                         baseVisualShapeIndex=pv_id,
                                         basePosition=[0, 0, 0 + config.POLE_LENGTH / 2],
                                         baseOrientation=init_orn)

        # 3. 约束：点对点约束 (球向铰链)
        # 将杆子的底部 ([0,0,-L/2]) 连到底座中心
        self.constraint_id = p.createConstraint(self.base_id, -1, self.pole_id, -1,
                                                p.JOINT_POINT2POINT, [0, 0, 1], [0, 0, 0],
                                                [0, 0, -config.POLE_LENGTH / 2])

        # 杆子底座阻力，杆子地面碰撞，底座地面碰撞
        p.changeDynamics(self.pole_id, -1, linearDamping=0, angularDamping=0.2, jointLowerLimit=0, jointUpperLimit=0)
        p.setCollisionFilterPair(self.pole_id, self.plane_id, -1, -1, enableCollision=0)
        p.setCollisionFilterPair(self.base_id, self.plane_id, -1, -1, enableCollision=0)

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
            p_pos[2],
            euler[0], euler[1],
            b_vel[0], b_vel[1],
            p_w[0], p_w[1]
        ], dtype=np.float32)

    def step(self, action):
        action = np.array(action, dtype=np.float32)

        self.smoothed_action = self.alpha * action + (1 - self.alpha) * self.smoothed_action

        self.base_pos[0] += self.smoothed_action[0]
        self.base_pos[1] += self.smoothed_action[1]

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

        v_cm, omega = p.getBaseVelocity(self.pole_id)
        p_pos, _ = p.getBasePositionAndOrientation(self.pole_id)
        r = np.array(top_pos) - np.array(p_pos)
        v_top = np.array(v_cm) + np.cross(np.array(omega), r)

        top_vx = v_top[0]  # 末端x速度
        top_vy = v_top[1]  #末端y速度
        top_vz = v_top[2]  # 末端z速度
        top_v = np.linalg.norm(v_top)
        self.steps += 1
        step_bonus = 0.03 * self.steps  #步数
        distance = np.linalg.norm([obs[0] , obs[1]])  #底座位移大小
        current_pole_z = obs[2]  #重心高度
        velocity_magnitude = np.linalg.norm(action)  #底座速度

        acceleration = (action - self.last_action) / config.TIME_STEP
        acc_magnitude = np.linalg.norm(acceleration)  # 加速度的大小
        high = current_pole_z if current_pole_z > 0.2 else -1  # 正高度
        pole_angle = np.arccos(2 * current_pole_z / config.POLE_LENGTH - 0.01)

        reward =  (
                #20.0 * (- current_pole_z * current_pole_z + current_pole_z)
                +np.pi/2 - pole_angle
                +1.0 * (current_pole_z)
                #+ 8.0 * np.power(current_pole_z,3)
                - 1 * np.power(distance,1)
                - 0.03 * np.power(distance, 3)
                #-(0.2 * top_v + 0.5)
                #- 0.005 * acc_magnitude
                #- 0.001 * velocity_magnitude
                + 0.08 * step_bonus
                )
        self.last_action = action.copy()

        # 终止条件
        terminated = bool(abs(obs[0]) > 10 or abs(obs[1]) > 10 or pole_angle > 0.2 or self.steps >= self.max_steps)
        return obs, reward, terminated, False, {}