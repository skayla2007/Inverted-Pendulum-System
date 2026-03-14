# config.py
import numpy as np

# 物理引擎参数
GRAVITY = -9.81
TIME_STEP = 1/120.0

# 杆件参数
BASE_MASS = 0        # 0 代表质量无穷大（静止或运动学控制）
POLE_MASS = 1.0
POLE_LENGTH = 1.0
POLE_RADIUS = 0.02

# 训练超参数
LEARNING_RATE = 0.0003
GAMMA = 0.99
TOTAL_TIMESTEPS = 500000  # 3D平衡较难，建议步数多一点