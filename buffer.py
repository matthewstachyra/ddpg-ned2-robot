import numpy as np

# memory of events that happened to sample in learning process
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.count = 0

        self.state_memory = np.zeroes((self.mem_size, 6))
        self.new_state_memory = np.zeroes((self.mem_size, 6))
        self.action_memory = np.zeros((self.mem_size, 6))
        self.reward_memory = np.zeroes(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store(self, state, action, reward, state_, done):
        index = self.count % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done # used in bellman equation to multiply by 
                                               # whether we are done
    
    def sample(self, batch_size):
        max_mem = min(self.count, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal