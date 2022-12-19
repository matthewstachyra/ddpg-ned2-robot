import numpy as np
import rospy
from niryo_robot_python_ros_wrapper import *

class Environment:
    # This class contains Ned2 specific behavior, including what are the
    #   valid joint ranges and checking whether an action is valid.
    def __init__(self, target):
        self.state = self.reset() # (0, 0, 0, 0, 0, 0)
        self.target = target      # (j1, j2, j3, j4, j5, j6) 

        # joint ranges are constrained to a cone shape in 
        #   front of the robot arm
        self.joint_ranges = {1 : (-2.5, 2.5),
                             2 : (-1.5, 0.5),
                             3 : (-1, 1.5),
                             4 : (-1.75, 1.75),
                             5 : (-1.5, .75),
                             6 : (-1, 1)}
        self.n_joints = len(self.joint_ranges)

        # initialize node and wrapper
        rospy.init_node('ned2_ddpg_reaching')
        self.ned = NiryoRosWrapper() 
    

    def reset(self):
        return (0, 0, 0, 0, 0, 0)
    

    def step(self, action):
        assert self.is_valid_action(action)
        self.state = tuple(map(sum, zip(self.state, tuple(action.tolist()))))
        reward = self.get_reward()
        done = self.check_done()
        return self.state, reward, done


    def is_valid_action(self, action):
        valid = True
        candidate_state = tuple(map(sum, zip(self.state, tuple(action.tolist()))))
        for i, range in enumerate(self.joint_ranges.values()):
            if candidate_state[i] < range[0] or \
                candidate_state[i] > range[1]:
                valid = False
        return valid
            

    def get_reward(self):
        return np.lingalg.norm(self.state - self.target)

    
    def check_done(self, thresh=0.15):
        distance = np.lingalg.norm(self.state - self.target)
        return distance < thresh


    def move(self, state):
        self.ned.move_joints(*state)


    def execute_trajectory(self, trajectory=None):
        print("Running optimal episode with trajectory that has {0} steps; \n trajectory: {1}".format(len(trajectory), trajectory))
        self.ned.execute_trajectory_from_poses_and_joints(trajectory, list_type=['joint'])
