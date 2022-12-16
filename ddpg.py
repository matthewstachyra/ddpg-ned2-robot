import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Actor, Critic
from buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG:
    def __init__(self, sdim, adim, amax):
        # actor network and actor-target network
        self.actor = Actor(sdim, adim, amax).to(device)
        self.actor_target = Actor(sdim, adim, amax).to(device)
        self.actor_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # critic network and critic-target network
        self.critic = Critic(sdim, adim).to(device)
        self.critic_target = Critic(sdim, adim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
    
    def tensorify(self, state, action, reward, sprime, done, weights):
        # helper method to convert all objects received from buffer to tensors
        return torch.FloatTensor(state).to(device), torch.FloatTensor(action).to(device), \
               torch.FloatTensor(reward).to(device), torch.FloatTensor(sprime).to(device), \
               torch.FloatTensor(1 - done).to(device), torch.FloatTensor(weights).to(device)
    
    def save_checkpoint(self):
        print("...saving checkpoint")
        torch.save(self.state_dict(), self.checkpoint)

    
    def load_checkpoint(self):
        print("...loading checkpoint")
        self.load_state_dict(torch.load(self.checkpoint))


    def get_action(self, state): 
        # pass the state through the network
        # this calls the __call__ method which via pytorch's
        #   nn.Module is changed to call a bunch of hooks
        return self.actor(torch.FloatTensor(state.reshape(1, -1)).to(device)).cpu().data.numpy().flatten()
    

    def train(self, buffer, prioritized, beta, epsilon, T, batch_size=64, gamma=0.99, tau=0.005):
        for i in range(T):
            # get state, action, reward, next state from buffer to update networks
            s, a, r, sprime, done = buffer.sample(batch_size)
            weights, batch_indices = np.ones_like(r), None
            s, a, r, sprime, done, weights = self.tensorify(s, a, r, sprime, done, weights)
            
            # compute q, y, and td errors
            q_target = self.critic_target(sprime, self.actor_target(sprime)) 
            y = r + (done * gamma * q_target).detach()
            q = self.critic(s, a)
            td_error = y - q
            w_td_error = torch.mul(td_error, np.sqrt(weights))
            zero_tensor = torch.zeros(w_td_error.shape)
            critic_loss = F.mse_loss(w_td_error, zero_tensor)

            # update critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # compute actor loss
            actor_loss = -self.critic(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # update target models
            fo
            


