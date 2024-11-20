import mujoco 
import mujoco.viewer
from icecream import ic
from controller.impedance import Impedance
import numpy as np

model_path = "/home/autrio/college-linx/RDC/project/models/panda.xml";

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False)


controller = Impedance(model, data, viewer)
controller.get_params()
controller.set_params(np.diag([1500,1500,1500,0,0,0]),np.diag([1,1,1,1,1,1,1,1,1]))
#! orientational impedance set to 0 since a ball is spatially symmetric
controller.resetViewer(False)

model.opt.gravity[2] = -1

flag = 0
key = "start_0"
key_i = 1
ts = 1000
controller.patience = 400

while viewer.is_running():
    target = np.zeros(7)
    target[:3] = data.body("target").xpos 
    target[3:] = data.body("target").xquat

    keys = ["start_"+str(i) for i in range(8)]

    launch_mapping = {
        "start_0":np.array([0,-40,65,0,0,0]),
        "start_1":np.array([0,40,65,0,0,0]),
        "start_2":np.array([-40,0,65,0,0,0]),
        "start_3":np.array([40,0,65,0,0,0]),
        "start_4":np.array([-30,-30,65,0,0,0]),
        "start_5":np.array([30,-30,65,0,0,0]),
        "start_6":np.array([-30,30,65,0,0,0]),
        "start_7":np.array([30,30,65,0,0,0])
    }



    if not controller(target,catch=True):
        # key = np.random.choice(keys)
        key = keys[key_i%8]
        ic(key)
        controller.resetViewer(False,key)
        key_i += 1
        flag = False

    if(not flag):
        ts -= 1
        if ts == 0:
            data.xfrc_applied[model.body('target').id] = launch_mapping[key]
            flag = True
            ts = 1000
