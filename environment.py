import math
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
        self.joint_ranges = {1 : (-1.25, 1.25),
                             2 : (-1.25, 0.25),
                             3 : (-1, 1),
                             4 : (-1.75, 1.75),
                             5 : (-0.25, .25),
                             6 : (-0.25, 0.25)}
        self.n_joints = len(self.joint_ranges)

        # initialize node and wrapper
        rospy.init_node('ned2_ddpg_reaching')
        self.ned = NiryoRosWrapper() 
    

    def reset(self):
        return (0, 0, 0, 0, 0, 0)
    

    def step(self, action):
        assert self.is_valid_action(action)
        self.state = self.state + action
        reward = self.get_reward()
        done = self.check_done()
        return self.state, reward, done


    def is_valid_action(self, action):
        valid = True
        candidate_state = self.state + action
        for i, range in enumerate(self.joint_ranges.values()):
            if candidate_state[i] < range[0] or \
                candidate_state[i] > range[1]:
                valid = False
        return valid
            

    def get_reward(self):
        # x, x2 = self.state[0], self.target[0]
        # y, y2 = self.state[1], self.target[1]
        # z, z2 = self.state[2], self.target[2]
        # return math.sqrt((z2 - z)**2 + (y2 - y)**2 + (x2 - x)**2)
        return -1

    
    def check_done(self):
        return self.state == self.target


    def move(self, state):
        self.ned.move_joints(*state)


    def execute_trajectory(self, trajectory=None):
        print("Running optimal episode with trajectory that has {0} steps; \n trajectory: {1}".format(len(trajectory), trajectory))
        self.ned.execute_trajectory_from_poses_and_joints(trajectory, list_type=['joint'])