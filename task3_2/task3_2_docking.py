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

# specific settings for this task

taskId = 3.2

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
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35,0.38,1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_2_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)

    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition          = [0.8, 0, 0],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/2]),                                  
        useFixedBase          = True,             
        globalScaling         = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf", 
        basePosition          = [0.5, 0, 1.1],            
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,0]),                                  
        useFixedBase          = False,             
        globalScaling         = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1,1,0,1])
    
    targetId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition          = finalTargetPos,             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )
    obstacle = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition          = [0.43,0.275,0.9],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )

    for _ in range(300):
        sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution(cubeId, targetId):

    print("TargetPosition: ", sim.p.getBasePositionAndOrientation(cubeId)[0], "Dumbbell at:", sim.p.getBasePositionAndOrientation(targetId)[0])
    print("DISTANCE: ", np.linalg.norm(np.array(sim.p.getBasePositionAndOrientation(cubeId)[0])-np.array(sim.p.getBasePositionAndOrientation(targetId)[0])))


    leftWrist = "LARM_JOINT5"
    rightWrist = "RARM_JOINT5"
    initPos = (sim.getJointPosition(leftWrist).flatten() + sim.getJointPosition(rightWrist).flatten())/2
    
    initPos[2] -= 0.02
    initPos[0] -= 0.08
    
    
    
    finalTargetPosition = np.array([0.345, 0.375, 1.04])
    
    initPosUp = 0 + initPos
    initPosUp[2] += 0.12
    initPos[0] += 0.05

    finalTargetPositionUp = 0 + finalTargetPosition
    finalTargetPositionUp[2] = 0 + initPosUp[2]
    print("Ini pos: ", initPos)
    print("Ini pos up : ", initPosUp)
    print("fini pos: ", finalTargetPosition)
    interp_steps = 4000

    y_diff=0.08
    x_diff=0.04
    z_diff=0.12

    mid_point_up = (initPosUp+finalTargetPositionUp)/2
    mid_point_up[0] = finalTargetPos[0]
    
    finaltargetmidup = (finalTargetPositionUp+finalTargetPos)/2
    
    targetPositions = np.empty((0,3))

    targetPositions = np.vstack([targetPositions, initPos])

    targetPositions = np.vstack([targetPositions, initPosUp])
   
    targetPositions = np.vstack([targetPositions, finalTargetPositionUp])

    targetPositions = np.vstack([targetPositions, finaltargetmidup])
    
    targetPositions = np.vstack([targetPositions, finalTargetPosition])
    
    
    sim.clamp(targetPositions = targetPositions, angularSpeed=0.005, interpolationSteps = interp_steps, y_diff= y_diff, x_diff = x_diff, z_diff = z_diff)
    print('End of wp1, end effector at : {} and {}'.format(sim.getJointPosition(leftWrist).flatten(), sim.getJointPosition(rightWrist).flatten()))
    print("TargetPositionDumbbell: ", sim.p.getBasePositionAndOrientation(cubeId)[0], "TargetPosition at:", sim.p.getBasePositionAndOrientation(targetId)[0])
    print("DISTANCE: ", np.linalg.norm(np.array(sim.p.getBasePositionAndOrientation(cubeId)[0])-np.array(sim.p.getBasePositionAndOrientation(targetId)[0])))

    
tableId, cubeId, targetId = getReadyForTask()
solution(cubeId = cubeId, targetId = targetId)
