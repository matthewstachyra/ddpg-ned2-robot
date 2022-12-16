import torch
import numpy as np

from agent import Agent
from env import Environment as Env

if __name__ == "__main__":
    TARGET = (0.5, -0.5, -0.5, 1, 0, 0)
    ALPHA = 0.000025
    BETA = 0.00025
    TAU = 0.001
    BATCH_SIZE = 64
    SEED = 1337
    EPISODES_N = 5000
    RUN_DATA = []

    # seed to ensure results are consistent
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env   = Env(...)
    agent = Agent(alpha=ALPHA, beta=BETA, sdim=[6], adim=[6], tau=TAU, env=env, \
                  batch_size=BATCH_SIZE)
            
    for episode in range(EPISODES_N):
        steps, tr = 0, 0
        done = False
        state = env.start()
        while not done:
            action = agent.act(state)
            sprime, reward, done = env.step(action)
            agent.remember(state, action, reward, sprime, int(done))
            agent.learn()
            state = sprime

            steps += 1
            tr += reward

        RUN_DATA.append(tr)
        print(f"{episode} had {steps} steps with total reward {tr}.")

        if episode%25==0: agent.save_models()
