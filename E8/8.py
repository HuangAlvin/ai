import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")  # 若改用這個，會畫圖
# env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
total_timesteps = 0
episode_timesteps = []

for _ in range(1000):
    env.render()
    
    # 獲取竿子的角度
    theta = observation[2]
    
    # 根據竿子的角度決定動作
    action = 1 if theta > 0 else 0
    
    observation, reward, terminated, truncated, info = env.step(action)
    total_timesteps += 1
    
    print('observation=', observation)
    
    if terminated or truncated:
        print(f'Episode finished after {total_timesteps} timesteps')
        episode_timesteps.append(total_timesteps)
        total_timesteps = 0
        observation, info = env.reset()
        print('done')

env.close()

# 輸出每次撐住的時間步數
print("Episode durations:", episode_timesteps)
