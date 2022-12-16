import random
import numpy as np
import torch
import rospy
from niryo_robot_python_ros_wrapper import *

from buffer import ReplayBuffer
from models import Actor, Critic
from noise import OUACtionNoise as Noise

class Agent:
    def __init__(self, alpha, beta, sdim=6, adim=6, tau, env, gamma=0.99,
                max_size=1000000, hidden1=400, hidden2=300, batch_size=64):

        # joint ranges are constrained to a cone shape in front of the robot arm
        self.joint_ranges = {1 : (-1.25, 1.25),
                             2 : (-1.25, 0.25),
                             3 : (-1, 1),
                             4 : (-1.75, 1.75),
                             5 : (-0.25, .25),
                             6 : (-0.25, 0.25)}
        self.n_joints = len(self.joint_ranges)
        
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(max_size, sdim, adim)
        self.batch_size = batch_size

        # noise to encourage action exploration
        self.noise = Noise(np.zeros(adim))
        
        # actor networks
        self.actor = Actor(alpha, sdim, adim, hidden1, hidden2, "Actor")
        # NOTE  this target is there to assist convergence / stability. it makes this off policy
        self.actor_target = Actor(alpha, sdim, adim, hidden1, hidden2, "Actor Target")

        # critic networks
        self.critic = Critic(beta, sdim, adim, hidden1, hidden2, "Critic")
        self.critic_target = Critic(beta, sdim, adim, hidden1, hidden2, "Critic Target")

        # NOTE  Solves moving target problem. If you use one network to get action and
        #       value of action then you chase moving target. Solution is to use a 
        #       target network to learn values of state, action pairs, and the other 
        #       network learns the policy. We slowly update the target to match the
        #       evaluation policy.

        # Ros specific stuff
        self.update_params(tau=1)
        # initialize node
        rospy.init_node('ned2_ddpg_reaching')
        # store wrapper
        self.ned = NiryoRosWrapper()


    def tensorify(self, state, action, reward, sprime, done):
            # helper method to convert all objects received from buffer to tensors
            return torch.FloatTensor(state, dtype=torch.float).to(self.critic.device), \
                   torch.FloatTensor(action, dtype=torch.float).to(self.critic.device), \
                   torch.FloatTensor(reward, dtype=torch.float).to(self.critic.device), \
                   torch.FloatTensor(sprime, dtype=torch.float).to(self.critic.device), \
                   torch.FloatTensor(1 - done, dtype=torch.float).to(self.critic.device)


    def act(self, obv):
        # NOTE  We need to call the built-in eval method of nn.Module because
        #       we use the batch norm layer 
        self.actor.eval() 
        obv = torch.tensor(obv, dtype=torch.float).to(self.actor.device)
        # get action from network
        mu = self.actor(obv).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
        self.actor.train()
        # idiom within pytorch to get numpy rather than tensor
        return mu_prime.cpu().detach.numpy()

    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    
    def learn(self):
        # we do not want to do anything if we do not have at least one batch's worth
        #   of data
        if self.buffer.count < self.batch_size:
            return
        
        # provided we have enough data, we sample from our buffer and convert
        #   all values to tensors with our helper method
        # each of these will be an array of 64 values
        s, a, r, sprime, done = self.buffer.sample(self.batch_size)
        s, a, r, sprime, done = self.tensorify(s, a, r, sprime, done)

        # NOTE  We need to call the built-in eval method of nn.Module because
        #       we use the batch norm layer 
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic.eval()

        target_actions = self.actor_target.forward(sprime)
        critic_value_ = self.critic_target.forward(sprime, target_actions)
        critic_value = self.critic.forward(s, a)

        # get target
        target = []
        for i in range(self.batch_size):
            target.append(r[i] + self.gamma*critic_value_[i]*done[i])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        # train critic
        self.critic.train()
        self.critic.opt.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.opt.step()
        self.critic.eval()

        # now the actor
        self.actor.opt.zero_grad()
        mu = self.actor.forward(s)
        self.actor.train()
        actor_loss = -self.critic.forward(s, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.opt.step()

        self.update_params()


    def update_params(self, tau=None):
        # NOTE  Tau is a hyperparameter that allows the target network
        #       values to gradually approach that of the evaluation network.
        #       This ensures slow convergence so we don't take large steps -
        #       therefore tau has a very low value.
        if tau is None: tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_target_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        critic_target_state_dict = dict(critic_target_params)
        actor_target_state_dict = dict(actor_target_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1 - tau)*critic_target_state_dict[name].clone()
        self.critic_target.load_state_dict(critic_state_dict)
            
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1 - tau)*actor_target_state_dict[name].clone()
        self.actor_target.load_state_dict(actor_state_dict)
    

    def save_models(self):
        self.actor.save()
        self.critic.save()
        self.actor_target.save()
        self.critic_target.save()
    

    def load_models(self):
        self.actor.load()
        self.critic.load()
        self.actor_target.load()
        self.critic_target.load()











































    def reset_agent(self):
        self.s = (0, 0, 0, 0, 0, 0)


    def get_start(self):
        # reset ned to starting position 
        self.reset_agent()

        # move ned to starting position
        self.ned.move_joints(*self.s)

        return self.s
        

    def set_discrete_action_space(self, delta):
        # 12 actions
        #   +/- delta to each of 6 joint values
        actions = []
        for i in range(6):
            a = [0]*6
            a2 = list(a) # create a local copy
            a[i] = delta
            a2[i] = -delta
            actions.append(tuple(a))
            actions.append(tuple(a2))
        return actions

    
    def get_actions(self):
        return self.a

    
    def get_valid_actions(self, state):
        # valid is defined as not exceeding
        #   the possible range of values for
        #   any of the 6 joints in Niryo Ned2.
        valid_actions = []
        # 12 actions
        for action in self.a:
            new_state = tuple(np.asarray(state) + np.asarray(action))
            # check each component of new state (i.e., each joint value)
            #   is within its perscribed range
            update = True
            for i in range(len(new_state)):
                valid_range = self.joint_ranges[i+1]
                joint_value = new_state[i]
                # append the joint adjustment (i.e., action) if within both - and + ranges
                if (joint_value < valid_range[0]) or (joint_value > valid_range[1]):
                   update = False 
            # only update if new state is o.k. for every joint in 6-tuple
            if update:
                valid_actions.append(action)
        return valid_actions
    

    def execute_trajectory(self, episode, load=False, trajectory=None):
        # episode has form episode[t] = ((s, a), r)
        #   train using qlearn()
        #   then run an episode with qlearn.run()
        #   then transform that to extract the states (to get the trajectory)

        # load (and do not regenerate trajectory) if 'load' is set to True with
        #   a provided trajectory name to load
        if load:
            trajectory = self.get_saved_trajectory(trajectory)
            self.ned.execute_trajectory_from_poses_and_joints(trajectory)
            return 
        
        # a trajectory will contain the sequence of states (i.e., joint values) that
        #   we progressively move through
        trajectory = [tup[0][0] for tup in episode.values()]
        
        print("Running optimal episode with trajectory that has {0} steps; \n trajectory: {1}".format(len(trajectory), trajectory))
        
        # execute tracjectory using niryo's provided wrapper
        self.ned.execute_trajectory_from_poses_and_joints(trajectory, list_type=['joint'])


