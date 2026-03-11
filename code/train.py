from stable_baselines3 import PPO
from BallJointEnv import BallJointPendulumEnv


def train():
    env = BallJointPendulumEnv(render_mode=None)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # 调低学习率，因为重力大，环境极度敏感
        n_steps=2048,
        batch_size=128,
        gamma=0.95,  # 降低折现率，让 AI 更关注眼前的存活
        ent_coef=0.01,  # 增加探索，防止过早收敛到错误姿态
    )

    print("开始强化训练...")
    # 建议至少训练 1,000,000 步。由于是 DIRECT 模式，速度会很快
    model.learn(total_timesteps=10000)
    model.save("ppo_ball_joint_3d_heavy_gravity")
    print("模型已保存。")
    env.close()


if __name__ == "__main__":
    train()