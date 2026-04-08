# train.py
from stable_baselines3 import PPO
from inverted_pendulum_env import InvertedPendulum3D
import config
import torch


def train():
    env = InvertedPendulum3D(render=False, use_mesh=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    print(f"使用 {device} 训练中...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
    model.save("kinematic_balance_model")


if __name__ == "__main__":
    train()
