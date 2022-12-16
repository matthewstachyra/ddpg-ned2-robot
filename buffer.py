import numpy as np

# class ReplayBuffer:
    # def __init__(self, capacity=1e6):
        # self.buffer = []
        # self.capacity = capacity
        # self.remove_index = 0


    # def append(self, data):
        # if len(self.buffer)==self.capacity:
            # # remove element that has been in the buffer the longest
            # self.buffer[self.remove_index] = data
            # self.remove_index = (self.remove_index + 1) % self.capacity
            # return
        # self.buffer.append(data)
    

    # def getTrainingBatch(self, batch_size):
        # indices = np.random.randint(0, len(self.buffer), size=batch_size)
        # s, a, r, sprime, d = [], [], [], [], []
        # for i in indices:
            # states, actions, rewards, stateprimes, done = self.buffer[i]
            # s.append(np.array(states, copy=False))
            # a.append(np.array(actions, copy=False))
            # r.append(np.array(rewards, copy=False))
            # sprime.append(np.array(stateprimes, copy=False))
            # d.append(np.array(done, copy=False))
        # return np.array(s), np.array(a), np.array(r).reshape(-1, 1), np.array(sprime), np.array(done)


# memory of events that happened to sample in learning process
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.count = 0

        self.state_memory = np.zeroes((self.mem_size, *input_shape))
        self.new_state_memory = np.zeroes((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
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