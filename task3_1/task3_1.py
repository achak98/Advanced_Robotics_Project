import subprocess, math, time, sys, os, numpy as np
from turtle import left
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.1

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
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
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": False,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    # compile urdfs
    finalTargetPos = np.array([0.7, 0.00, 0.91])
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_1_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition        = [0.8, 0, 0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi/2]),
        useFixedBase        = True,
        globalScaling       = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition        = [0.33, 0, 1.0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase        = False,
        globalScaling       = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition        = finalTargetPos,
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase        = True,
        globalScaling       = 1
    )
    for _ in range(200):
        sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution(cubeId, targetId):
    # TODO: Add
    leftWrist = "LARM_JOINT5"
    targetPosition = [0.33, 0, 1.0]
    print("TargetPosition: ", sim.p.getBasePositionAndOrientation(cubeId)[0], "Dumbbell at:", sim.p.getBasePositionAndOrientation(targetId)[0])
    print("DISTANCE: ", np.linalg.norm(np.array(sim.p.getBasePositionAndOrientation(cubeId)[0])-np.array(sim.p.getBasePositionAndOrientation(targetId)[0])))
    #print('Initial Position: {}'.format(sim.getJointPosition(leftWrist).flatten()))
    #print('Initial Orientation: {}'.format(sim.getJointAxis(leftWrist).flatten()))
    
    scaleP = 1
    scaleD = 0.95
    scaleI = 1000000000000000
    
    wayPoint1 = sim.getJointPosition(leftWrist).flatten()
    wayPoint1[0]-=0.3
    wayPoint1[1]-=0.115
    wayPoint1[2]+=0.1
    interp_steps = 500
    sim.move_with_PD(leftWrist, wayPoint1,interpolationSteps=interp_steps, speed=0.01, orientation=[0,0,1], threshold=1e-3, debug=False, verbose=False, scaleP = scaleP, scaleD = scaleD, scaleI = scaleI)
    #print('End of wp1, target at: {} end effector at : {}'.format(wayPoint1, sim.getJointPosition(leftWrist).flatten()))

    #sim.moveJoint(joint = "LARM_JOINT2", targetPosition = sim.getJointPos("LARM_JOINT2")-np.deg2rad(4), targetVelocity = 0.0, interpolationSteps = 300)    
    #sim.disableVelocityController("LARM_JOINT2")
    sim.moveJoint(joint = "LARM_JOINT5", targetPosition = sim.getJointPos("LARM_JOINT5")+np.deg2rad(45), targetVelocity = 0.0, interpolationSteps = 300)

    wayPoint2 = sim.getJointPosition(leftWrist).flatten()
    wayPoint2[0]-=0.08
    wayPoint2[1]-=0.11
    wayPoint2[2]=.96
    sim.move_with_PD(leftWrist, wayPoint2,interpolationSteps=interp_steps*2, speed=0.01, orientation=[0,0,1], threshold=1e-3, debug=False, verbose=False, scaleP = scaleP, scaleD = scaleD, scaleI = scaleI)
    #print('End of wp2, target at: {} end effector at : {}'.format(wayPoint2, sim.getJointPosition(leftWrist).flatten()))
    
    

    #pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = \
    #    sim.moveJoint(
    #         "LARM_JOINT5", np.deg2rad(45), 0.0, False)
    #wayPoint3 = sim.getJointPosition(leftWrist).flatten()
    #sim.move_with_PD(leftWrist, wayPoint3,interpolationSteps=interp_steps, speed=0.01, orientation=[0,0,1], threshold=1e-3, debug=False, verbose=False)
    #print('End of wp3, end effector at : {}'.format(sim.getJointPosition(leftWrist).flatten()))

    joint_axis = sim.getJointOrientation(leftWrist)
    current_posi = sim.getJointPosition(leftWrist).flatten()
    for i in range (5, 0, -1):
        wayPoint3 = sim.getJointPosition(leftWrist).flatten()
        wayPoint3[0] = current_posi[0] + (0.540-current_posi[0])/i
        wayPoint3[1] = current_posi[1] + (-0.13-current_posi[1])/i
        wayPoint3[2] = current_posi[2] + (.94-current_posi[2])/i
        x_vector = joint_axis[0]+ (1-joint_axis[0])/i
        y_vector = joint_axis[1]-(joint_axis[1])/i
        z_vector = joint_axis[2]-(joint_axis[2])/i
        sim.move_with_PD(leftWrist, wayPoint3,interpolationSteps=250, speed=0.01, orientation=[0,0,1], threshold=1e-3, debug=False, verbose=False)
       # print('End of wp3.{}, target at: {} end effector at : {}'.format(6-i,wayPoint3,sim.getJointPosition(leftWrist).flatten()))
    time.sleep(2)
    """wayPoint5 = sim.getJointPosition(leftWrist).flatten()
    wayPoint5[0] = 0.6
    wayPoint5[1] = 0
    sim.move_with_PD(leftWrist, wayPoint5,interpolationSteps=interp_steps, speed=0.01, orientation=[1,0,0], threshold=1e-3, debug=False, verbose=False)
    print('End of wp5, end effector at : {}'.format(sim.getJointPosition(leftWrist).flatten()))"""
    print("TargetPosition: ", sim.p.getBasePositionAndOrientation(cubeId)[0], "Dumbbell at:", sim.p.getBasePositionAndOrientation(targetId)[0])
    print("DISTANCE: ", np.linalg.norm(np.array(sim.p.getBasePositionAndOrientation(cubeId)[0])-np.array(sim.p.getBasePositionAndOrientation(targetId)[0])))
tableId, cubeId, targetId = getReadyForTask()
solution(cubeId, targetId)
