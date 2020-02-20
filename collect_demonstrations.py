from real_robot import Rozum,VideoCapture
import cv2
import gym
from threading import Thread
import numpy as np
import time

class rozum_real:

    def __init__(self):
        self.robot = Rozum()
        self.DoF = 6
        # self.action_bound = [[-15,15],[-10,110],[-30,30],[-120,120],[-180,180],[-180,180]]
        self.action_bound = [[-240, -180], [-180, 180], [-180, 180], [-220, -100], [-180, 180], [-180, 180]]
        self.action_range = [-5, 5]
        self.cam = VideoCapture(2)
        self.w = self.cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.action_space = gym.spaces.Box(shape=(self.DoF,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(5 + self.DoF * 2,), low=-1, high=1)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]
        #
        self.currents_thread = Thread(target=self.current_reader)
        self.currents_thread.daemon = True
        self.currents_thread.start()

        self.angles_thread = Thread(target=self.angle_reader)
        self.angles_thread.daemon = True
        self.angles_thread.start()

        # self.robot.open_gripper()
        self.init_pose, self.init_orientation = self.robot.get_position()
        # self.init_angles = [-200,-90,-90,-90,90,0]
        self.init_angles = [-210.0, -100.0, -110.0, -60.0, 90.0, -35.0]
        self.robot.update_joint_angles(self.init_angles)
        self.robot.send_joint_angles()
        # self.robot.relax()
        self.path="/home/ali/VAE/demo1/"
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.count=0
        self.num_demonstrations=3

    def current_reader(self):
        while True:
            self.currents = self.robot.get_joints_current()

    def angle_reader(self):
        while True:
            self.angles = self.robot.get_joint_angles()

    def collect_demo(self):
        for i in range(self.num_demonstrations):
            self.robot.update_joint_angles(self.init_angles)
            self.robot.send_joint_angles()
            x=input("press enter when ready")
            out = cv2.VideoWriter(self.path + 'output%d.mp4'%i, self.fourcc, 15.0, (int(self.w), int(self.h)))
            angles = []
            self.robot.relax()
            while True:
                self.robot.freeze()
                time.sleep(1)
                img=self.cam.read()
                cv2.imshow("1",img)
                c = cv2.waitKey(0)
                out.write(img)
                angles.append(self.angles)
                if c==27:
                    x=input("positive(1) or negative(0)?")
                    out.release()
                    if x==1:
                        np.savetxt("%d_pos"%i,X=angles, delimiter=",")
                    else:
                        np.savetxt("%d_neg"%i,X=angles,delimiter=",")
                else:
                    self.robot.relax()
                    time.sleep(2)

env=rozum_real()
env.robot.relax()

