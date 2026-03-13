# run.py
import time
from stable_baselines3 import PPO
from inverted_pendulum_env import InvertedPendulum3D

def run():
    env = InvertedPendulum3D(render=True)
    model = PPO.load("kinematic_balance_model")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)
        time.sleep(1/120.0)
        if terminated: obs, _ = env.reset()

if __name__ == "__main__":
    run()