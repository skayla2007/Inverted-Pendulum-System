import time
import numpy as np
from BallJointEnv import BallJointPendulumEnv


def test_full_fall():
    env = BallJointPendulumEnv(render_mode='human')
    obs, _ = env.reset()

    print(">>> 质点底座测试：观察是否能突破 64 度并完全倒下")

    try:
        for i in range(1200):
            obs, _, done, _, _ = env.step(np.array([0.0, 0.0]))

            # 计算总倾斜角
            angle_r = np.arccos(np.clip(obs[5], -1, 1))
            angle_p = np.arccos(np.clip(obs[7], -1, 1))
            total_angle = np.degrees(np.sqrt(angle_r ** 2 + angle_p ** 2))

            if i % 5 == 0:
                print(f"步数: {i:4d} | 角度: {total_angle:6.2f}°")

            time.sleep(1. / 120.)  # 加快一点显示速度

            if done:
                print(f">>> 成功倒下！最终角度: {total_angle:.2f}°")
                time.sleep(0.1)
                obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    test_full_fall()