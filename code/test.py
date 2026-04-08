# physics_test.py
import pybullet as p
import time
import numpy as np
from inverted_pendulum_env import InvertedPendulum3D


def test_manual():
    env = InvertedPendulum3D(render=True, use_mesh=False)
    obs, _ = env.reset()
    print(">>> 物理验证：底座不受杆子影响，按方向键移动底座，观察杆子惯性。")

    while True:
        keys = p.getKeyboardEvents()
        dx, dy = 0, 0
        speed = 0.01
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            dx = -speed
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            dx = speed
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            dy = speed
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            dy = -speed

        obs, _, terminated, _, _ = env.step([dx, dy])

        if terminated:
            print("杆子完全倒下，重置...")
            obs, _ = env.reset()
        time.sleep(1 / 120.0)


if __name__ == "__main__":
    test_manual()
