import time
import numpy as np
import pybullet as p
import csv
from datetime import datetime
from stable_baselines3 import PPO
from inverted_pendulum_env import InvertedPendulum3D

# ===================== 新增：统计函数 =====================
def init_episode_stats():
    """初始化单轮实验统计变量"""
    return {
        "episode_num": 0,
        "start_time": time.time(),
        "max_pole_angle": 0.0,
        "force_apply_count": 0
    }

def calculate_pole_angle(env):
    """计算摆的倾斜角度（弧度→角度，越大越不稳）"""
    # 获取摆的姿态四元数 → 转换为欧拉角
    pos, orn = p.getBasePositionAndOrientation(env.pole_id)
    euler = p.getEulerFromQuaternion(orn)
    # 计算x/y轴倾斜角度的模长（总倾斜角）
    angle_x = abs(euler[0])
    angle_y = abs(euler[1])
    total_angle = np.sqrt(angle_x**2 + angle_y**2)
    return np.degrees(total_angle)  # 转换为角度，更直观

def save_episode_stats(stats, csv_path="run_stats.csv"):
    """保存单轮统计结果到CSV，实时打印"""
    duration = time.time() - stats["start_time"]
    # 打印控制台
    print(f"\n=== 第{stats['episode_num']}轮结束 ===")
    print(f"平衡时长：{duration:.2f}s")
    print(f"最大倾斜角：{stats['max_pole_angle']:.2f}°")
    print(f"手动外力次数：{stats['force_apply_count']}")
    print("==========================\n")

    # 保存到CSV文件（追加模式，不覆盖历史数据）
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "时间", "轮次", "平衡时长(s)", "最大倾斜角(°)", "外力次数"
        ])
        # 文件为空时写入表头
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "轮次": stats["episode_num"],
            "平衡时长(s)": round(duration, 2),
            "最大倾斜角(°)": round(stats["max_pole_angle"], 2),
            "外力次数": stats["force_apply_count"]
        })

# ===================== 主运行函数 =====================
def run():
    env = InvertedPendulum3D(render=True)
    # ✅ 修复：指定CPU运行，消除GPU警告
    model = PPO.load("kinematic_balance_model", device="cpu")
    # 初始化统计
    stats = init_episode_stats()

    # 首次重置环境
    obs, _ = env.reset()

    while True:
        # 1. 键盘外力控制
        keys = p.getKeyboardEvents()
        force_x, force_y = 0, 0
        push_magnitude = 0.01

        # ✅ 修复：所有方向键判断逻辑统一，无KeyError
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            force_x = -push_magnitude
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            force_x = push_magnitude
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            force_y = push_magnitude
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            force_y = -push_magnitude

        # 施加外力 + 计数
        if force_x != 0 or force_y != 0:
            p.applyExternalForce(env.pole_id, -1,
                                 forceObj=[force_x, force_y, 0],
                                 posObj=[0, 0, 0],
                                 flags=p.WORLD_FRAME)
            stats["force_apply_count"] += 1

        # 2. 模型推理
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        # 3. 更新统计：实时计算摆角，更新最大值
        current_angle = calculate_pole_angle(env)
        if current_angle > stats["max_pole_angle"]:
            stats["max_pole_angle"] = current_angle

        time.sleep(1/120.0)

        # 4. 轮次结束：保存统计 + 重置
        if terminated or truncated:
            stats["episode_num"] += 1
            save_episode_stats(stats)  # 调用统计函数
            stats = init_episode_stats()  # 重置统计变量
            obs, _ = env.reset()

if __name__ == "__main__":
    run()