import gym

env = gym.make('procgen:procgen-coinrun-v0', start_level=10, num_levels=0, render_mode='human')
print("env.action_space: ", env.action_space)

obs = env.reset()

step = 0
while True:
    print("step: ", step)

    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        break

    step += 1