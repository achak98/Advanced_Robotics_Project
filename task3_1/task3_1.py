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
    
    
    wayPoint1 = sim.getJointPosition(leftWrist).flatten()
    wayPoint1[0]-=0.3
    wayPoint1[2] = 1.04

    interp_steps = 1000
    sim.move_with_PD(leftWrist, wayPoint1,interpolationSteps=interp_steps//2, speed=0.01, orientation=[1,0,1], threshold=1e-3, debug=False, verbose=False)
    #print('End of wp1, target at: {} end effector at : {}'.format(wayPoint1, sim.getJointPosition(leftWrist).flatten()))
    
    wayPoint2 = sim.getJointPosition(leftWrist).flatten()
    wayPoint2[0] -= 0.12
    wayPoint2[1] -= 0.205
    wayPoint2[2] = .97
    sim.move_with_PD(leftWrist, wayPoint2,interpolationSteps=interp_steps//2, speed=0.01, orientation=[1,1,1], threshold=1e-3, debug=False, verbose=False)

    wayPoint3 = sim.getJointPosition(leftWrist).flatten()
    wayPoint3[0] = 0.56
    wayPoint3[1] = -0.01
    wayPoint3[2] = 0.96

    sim.move_with_PD(leftWrist, wayPoint3,interpolationSteps=interp_steps, speed=0.01, orientation=[0,0,1], threshold=1e-3, debug=False, verbose=False)

    time.sleep(2)

    print("TargetPosition: ", sim.p.getBasePositionAndOrientation(cubeId)[0], "Dumbbell at:", sim.p.getBasePositionAndOrientation(targetId)[0])
    print("DISTANCE: ", np.linalg.norm(np.array(sim.p.getBasePositionAndOrientation(cubeId)[0])-np.array(sim.p.getBasePositionAndOrientation(targetId)[0])))
    
    
tableId, cubeId, targetId = getReadyForTask()
solution(cubeId, targetId)
