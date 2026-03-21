# run.py
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from inverted_pendulum_env import InvertedPendulum3D

def run():
    env = InvertedPendulum3D(render=True)
    model = PPO.load("kinematic_balance_model")
    obs, _ = env.reset()
    while True:
        keys = p.getKeyboardEvents()
        force_x, force_y = 0, 0
        push_magnitude = 0.01

        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: force_x = -push_magnitude
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: force_x = push_magnitude
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: force_y = push_magnitude
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: force_y = -push_magnitude

        if force_x != 0 or force_y != 0:
            p.applyExternalForce(env.pole_id, -1,
                                 forceObj=[force_x, force_y, 0],
                                 posObj=[0, 0, 0],
                                 flags=p.WORLD_FRAME)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        time.sleep(1/120.0)
        if terminated or truncated: obs, _ = env.reset()

if __name__ == "__main__":
    run()