import mujoco
import mujoco.viewer
from icecream import ic
import numpy as np

class Simulation:
    def __init__(self,model,data,controller):
        self.model = model
        self.data = data
        self.controller = controller

    def set_states(self,states=None):
            pass