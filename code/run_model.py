import time
import numpy as np
from stable_baselines3 import PPO
from BallJointEnv import BallJointPendulumEnv


def run_expert():
    # 开启可视化渲染
    env = BallJointPendulumEnv(render_mode='human')

    model_path = "ppo_ball_joint_3d_heavy_gravity.zip"

    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print("成功加载已训练的模型。")
    else:
        print("未发现模型文件，请先运行 train.py")
        return

    obs, _ = env.reset()

    print("AI 演示中 (Ctrl+C 退出)...")
    try:
        while True:
            # deterministic=True 确保动作稳定
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)

            # 物理步长调节，确保视觉流畅
            time.sleep(1.0 / 60.0)

            if done:
                print("杆子失衡，正在重置...")
                time.sleep(0.5)
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n演示结束")
    finally:
        env.close()


if __name__ == "__main__":
    import os

    run_expert()