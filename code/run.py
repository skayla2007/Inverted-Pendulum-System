# run.py
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from inverted_pendulum_env import InvertedPendulum3D

def run():
    env = InvertedPendulum3D(render=True, use_mesh=True)
    model = PPO.load("kinematic_balance_model")
    obs, _ = env.reset()
    while True:
        keys = p.getKeyboardEvents()
        impulse = 0.5
        ang_vel = [0, 0, 0]

        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: ang_vel = [0, -impulse, 0]
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: ang_vel = [0, impulse, 0]
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: ang_vel = [-impulse, 0, 0]
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: ang_vel = [impulse, 0, 0]

        if any(v != 0 for v in ang_vel):
            curr_v, curr_w = p.getBaseVelocity(env.pole_id)
            p.resetBaseVelocity(env.pole_id, curr_v, [curr_w[0] + ang_vel[0], curr_w[1] + ang_vel[1], 0])

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        time.sleep(1/120.0)
        if terminated or truncated: obs, _ = env.reset()

if __name__ == "__main__":
    run()