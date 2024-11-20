import mujoco 
import mujoco.msh2obj_test
import mujoco.viewer

import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from icecream import ic

class Controller:
    def __init__(self,model,data,viewer):
        self.model = model
        self.data = data
        self.viewer = viewer

    def get_params(self):
        self.actuators = np.array([self.model.actuator(name).id for name in range(self.model.nu)])
        robot_joints = [self.model.joint(i).name for i in range(self.model.njnt) if "panda" in self.model.joint(i).name]
        self.joints = np.array([self.model.joint(name).id for name in robot_joints])
        self.njoints = len(self.joints)
        self.extern_nv = self.model.nv - self.njoints


    def __call__(self, tau):
        self.data.ctrl[self.actuators] = tau
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        return 1

class Impedance(Controller):
    def __init__(self, model, data, viewer):
        super().__init__(model, data, viewer)

    def set_params(self,K,K_null):
        self.K = K
        self.K_null = K_null
        self.D = 2*np.sqrt(self.K)
        self.D_null = 2*np.sqrt(self.K_null)
        self.eef = "end_effector"
        self.key_name = "home"
        self.key_id = self.model.key(self.key_name).id
        self.q0 = self.model.key(self.key_name).qpos[self.joints]
        self.qd0 = self.data.qvel[self.joints]
        self.jac = np.zeros((6, self.model.nv))
        self.jacDot = np.zeros((6, self.model.nv))
        self.M_all = np.zeros((self.model.nv, self.model.nv))
        self.Lmd = np.zeros((6, 6))
        self.eye = np.eye(7)
        self.twist = np.zeros(6)
        self.quat = np.zeros(4)
        self.quat_conj = np.zeros(4)
        self.quat_error = np.zeros(4)
        self.pose_error = np.zeros(6) #? respresenting cartesian pose error as dx dy dz dwx dwy dwz
        self.patience = 600


    def resetViewer(self,flag,key = None):
        if key is not None:
            self.key_name = key
            self.key_id = self.model.key(self.key_name).id

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)
        if(flag):
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def step(self,target):
        self.X = self.data.body(self.eef).xpos
        self.dX = target[:3] - self.X

        self.twist[:3] = self.dX/self.model.opt.timestep
        self.quat = self.data.body(self.eef).xquat
        mujoco.mju_negQuat(self.quat_conj, self.quat)
        mujoco.mju_mulQuat(self.quat_error, target[3:], self.quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.quat_error, 1.0)

        rotnErr = R.from_quat(np.array([self.quat_error[1],self.quat_error[2],self.quat_error[3],self.quat_error[0]]))
        self.ro = rotnErr.as_euler("xyz",degrees=False)

        self.pose_error = np.concatenate((-self.dX,self.ro))

        mujoco.mj_jacBody(self.model, self.data, self.jac[:3],self.jac[3:], self.model.body(self.eef).id)
        mujoco.mj_jacDot(self.model, self.data, self.jacDot[:3],self.jacDot[3:],np.array([0,0,0]),self.model.body(self.eef).id)
    

        self.M = np.zeros((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model,self.M,self.data.qM)
        self.M = self.M[:self.njoints,:self.njoints]

        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        self.M_inv=self.M_all[:self.njoints,:self.njoints];
        self.Lmd_inv = self.jac[:,self.joints] @ self.M_inv @ self.jac[:,self.joints].T

        if abs(np.linalg.det(self.Lmd_inv)) >= 1e-5:
            self.Lmd = np.linalg.inv(self.Lmd_inv)
        else:
            self.Lmd = np.linalg.pinv(self.Lmd_inv, rcond=1e-5)

        self.h = self.data.qfrc_bias[self.joints]
        self.mu = self.Lmd @ (self.jac[:,self.joints] @ self.M_inv @ self.h + self.jacDot[:,self.joints] @ self.data.qvel[self.joints])

        self.tau = self.jac[:,self.joints].T @ (self.K@self.pose_error + self.D@self.twist + self.mu)

        self.Jbar = self.M_inv @ self.jac[:,self.joints].T @ self.Lmd
        self.ddq = self.K_null @ (self.data.qpos[self.joints]) - self.D_null @ (self.data.qvel[self.joints])
        self.tau1 = self.M @ self.ddq + self.h
        self.tau += (np.eye(self.model.nv-self.extern_nv) - self.jac[:,self.joints].T @ self.Jbar.T) @ self.tau1

        return self.tau

    def __call__(self, target,catch=True):
        assert target.shape == (7,)
        tau = self.step(target)
        if np.linalg.norm(target[:3]-[0,0,0.33]) > 1.1:
            target = np.array([-0.3,0,0.3,0,1,0,0])
            tau = self.step(target)
        grip = False
        if catch:
            if np.linalg.norm(self.data.body("target").xpos - self.X) < 0.03:
                self.patience -= 1
            if self.patience < 20:
                grip = True
            if grip:
                self.eef = "end_effector_grasp"
                if self.patience < 0:
                    tau[7:9] = 0
                if self.patience < -40 or np.linalg.norm(self.data.body("target").xpos - self.X) > 2:
                    time.sleep(1)
                    self.patience = 300
                    self.eef = "end_effector"
                    return 0 #!status code RESET   
            else:
                tau[7:9] = 0.04
                if np.linalg.norm(self.data.body("target").xpos - self.X) > 2:
                    self.patience = 300
                    self.eef = "end_effector"
                    return 0
        else:
            tau[7:9] = 0.04
            if np.linalg.norm(self.data.body("target").xpos - self.X) > 2:
                self.patience = 300
                self.eef = "end_effector"
                return 0
        return super().__call__(tau)
    
        
