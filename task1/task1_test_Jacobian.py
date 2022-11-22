# Please don't submit this file.
# This file is only for helping you while development

import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation
from Pybullet_Simulation_template import Simulation_template

# specific settings for this task
taskId = 1

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True


### You may want to change the code since here
pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": True,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains_template.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": True
}

verbose = False
debugLine = True

# sim = Simulation_template(pybulletConfigs, robotConfigs)
# print(sim.joints)

# def getMotorJointStates(p, robot):
#     joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
#     joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
#     joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
#     joint_positions = [state[0] for state in joint_states]
#     joint_velocities = [state[1] for state in joint_states]
#     joint_torques = [state[3] for state in joint_states]
#     return joint_positions, joint_velocities, joint_torques


# mpos, mvel, mtorq = getMotorJointStates(bullet_simulation, sim.robot)
# res = bullet_simulation.getLinkState(sim.robot,
#                      sim.jointIds['LARM_JOINT5'],
#                      computeLinkVelocity=1,
#                      computeForwardKinematics=1)
# link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = res
# j_geo, j_rot = bullet_simulation.calculateJacobian(
#     sim.robot, 
#     sim.jointIds['LARM_JOINT5'],
#     [0,0,0], 
#     mpos,
#     [0.0] * len(mpos),
#     [0.0] * len(mpos),
# )
# print()
# for col in j_geo:
#     print(col)
# print()
# print(len(j_geo[0]))
# for col in j_rot:
#     print(col)
# print()

# try:
#     time.sleep(float(sys.argv[1]))
# except:
#     time.sleep(10)

# Our Test Start

sim = Simulation(pybulletConfigs, robotConfigs)
print(sim.joints)

j = sim.jacobianMatrix('LARM_JOINT5')
j_geo = j[0:3,:]
j_v = j[3:,:]
print('Jacobian start')
for col in j_geo:
    print(col)
print('Jacobian end')

print('Jacobia vector start')
for col in j_v:
    print(col)
print('Jacobian vector end')

try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)
# End

# # TODO: Add your code here to start simulation

# ref = [0, 0, 1]
# sim = Simulation(pybulletConfigs, robotConfigs, refVect=ref)

# # This is an example target position for the left end effector. This target
# # position assumes your world frame is located at the base. If your world
# # frame is located at the waist, you will need to transform this vector using
# # the base_to_waist translation.
# endEffector = "LARM_JOINT5"
# targetPosition = np.array([0.37, 0.23, 1.06385])  # x,y,z coordinates in world frame

# # Example code. Feel free to modify
# pltTime, pltEFPosition = sim.move_without_PD(endEffector, targetPosition, speed=0.01, orientation=None, threshold=1e-3, maxIter=3000, debug=False, verbose=False)


# # Now plot some graphs
# task1_figure_name = "task1_kinematics.png"
# task1_savefig = True
# # ...

# fig = plt.figure(figsize=(6, 4))

# plt.plot(pltTime, pltEFPosition, color='blue')
# plt.xlabel("Time s")
# plt.ylabel("Distance to target position")

# plt.suptitle("task1 IK without PD", size=16)
# plt.tight_layout()
# plt.subplots_adjust(left=0.15)

# if task1_savefig:
#     fig.savefig(task1_figure_name)
# plt.show()
