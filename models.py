import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    # Outputs q-value for state, action pair - "criticizing" the action output by the Actor.
    #
    # sdim <- dimension of the state which is (6,) because state is 
    #   represented as a 6-tuple of joint values
    # adim <- dimension of the action which is (6,) because action is
    #   also represented as a 6-tuple of joint  values
    def __init__(self, beta, sdim, adim, hidden1=400, hidden2=300, name, checkpoint_dir='tmp'):
        super().__init__()

        # layer 1
        self.fc1 = nn.Linear(sdim, hidden1)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0]) # initialize weights to avoid vanishing or exploding gradient
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(hidden1)

        # layer 2
        self.fc2 = nn.Linear(hidden1 + adim, hidden2)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(hidden2)
        
        # layer 3
        self.action_value = nn.Linear(adim, hidden2)
        f3 = 0.003
        self.fc3 = nn.Linear(hidden2, 1)
        torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

        # Adam optimizer
        self.opt = nn.optim.Adam(self.parameters(), lr=beta) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg')


    def forward(self, state, action):
        # passes the input through the network to get an output; in
        #   this case for the critic network it is a q value which
        #   should have dimension (1,)
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.fc3(state_action_value)
        return state_action_value
    

    def save(self):
        print("Saving.")
        torch.save(self.state_dict(), self.checkpoint_file)
    

    def load(self):
        print("Loading.")
        self.load_state_dict(torch.load(self.checkpoint_file))



class Actor(nn.Module):
    # Outputs action for the input state.
    def __init__(self, alpha, sdim, adim, hidden1=400, hidden2=300, name, checkpoint_dir='tmp'):
        super().__init__()

        # layer 1
        self.fc1 = nn.Linear(sdim, hidden1)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0]) 
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(hidden1)

        # layer 2
        self.fc2 = nn.Linear(hidden1, hidden2)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0]) 
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(hidden2)

        # layer 3 - mu
        self.action_value = nn.Linear(adim, hidden2)
        f3 = 0.003
        self.fc3 = nn.Linear(hidden2, adim)
        torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        
        # Adam optimizer
        self.opt = nn.optim.Adam(self.parameters(), lr=alpha) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg')


    def forward(self, state):
        # passes the input through the network to get an output; in
        #   this case for the actor network it is an action which
        #   should have dimension (6,)
        action = self.fc1(state)
        action = self.bn1(action)
        action = F.relu(action)
        action = self.fc2(action)
        action = self.bn2(action)
        action = F.relu(action)
        action = torch.tanh(self.fc3(action))
        return action


    def save(self):
        print("Saving.")
        torch.save(self.state_dict(), self.checkpoint_file)
    

    def load(self):
        print("Loading.")
        self.load_state_dict(torch.load(self.checkpoint_file))