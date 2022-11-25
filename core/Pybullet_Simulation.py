from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline, CubicHermiteSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base

class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics

    # To be used as entries for the Jacobian Matrix
    jointList = [
        'CHEST_JOINT0',
        'HEAD_JOINT0',
        'HEAD_JOINT1',
        'LARM_JOINT0',
        'LARM_JOINT1',
        'LARM_JOINT2',
        'LARM_JOINT3',
        'LARM_JOINT4',
        'LARM_JOINT5',
        'RARM_JOINT0',
        'RARM_JOINT1',
        'RARM_JOINT2',
        'RARM_JOINT3',
        'RARM_JOINT4',
        'RARM_JOINT5'
        #'RHAND',
        #'LHAND'
    ]

    #Used to figure out kinematic chain for end effector
    jointPathDict = {
        'base_to_waist': ['base_to_waist'],  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': ['base_to_waist', 'CHEST_JOINT0'],
        'HEAD_JOINT0':  ['base_to_waist', 'CHEST_JOINT0', 'HEAD_JOINT0'],
        'HEAD_JOINT1':  ['base_to_waist', 'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1'],
        'LARM_JOINT0':  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0'],
        'LARM_JOINT1':  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1'],
        'LARM_JOINT2':  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2'],
        'LARM_JOINT3':  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3'],
        'LARM_JOINT4':  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4'],
        'LARM_JOINT5':  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5'],
        'RARM_JOINT0':  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0'],
        'RARM_JOINT1':  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1'],
        'RARM_JOINT2':  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2'],
        'RARM_JOINT3':  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3'],
        'RARM_JOINT4':  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4'],
        'RARM_JOINT5':  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
        #'RHAND'      :  ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5', 'RHAND'],
        #'LHAND'      :  ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'LHAND']
    }

    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.array([0,0,1]),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1])
        #'RHAND'      : np.array([0, 0, 1]),
        #'LHAND'      : np.array([0, 0, 1])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.array([0,0,0.85]),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335])
        #'RHAND'      : np.array([0.0095, 0, 0]), # optional
        #'LHAND'      : np.array([0.0095, 0, 0]) # optional
    }

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        # TODO modify from here
        # Hint: the output should be a 3x3 rotational matrix as a numpy array
        # Get the axis from dictionary
        axis = self.jointRotationAxis[jointName]

        # Compute the x,y and z rotational matrix
        rx = np.eye(3)
        ry = np.eye(3)
        rz = np.eye(3)

        # If there is rotation in the axis, recalculate the value
        if axis[0] == 1:
            rx = np.matrix([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        if axis[1] == 1:
            ry = np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])

        if axis[2] == 1:
            rz = np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        # Compute the rotational matrix
        r = rz*ry*rx

        # Check that the rotational matrix computed is 3x3
        assert(len(r) == 3)

        return r

    def getTransformationMatrices(self, thetas = None):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        # TODO modify from here
        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.

        # Loop through the joints from existing dictionary
        for jointName in self.jointRotationAxis:
            # Get the rotation matrix and translation from parent
            if thetas == None or thetas[jointName] == None:
                r = self.getJointRotationalMatrix(jointName, self.getJointPos(jointName))
            else:
                r = self.getJointRotationalMatrix(jointName, thetas[jointName])
            p = np.array(self.frameTranslationFromParent[jointName])

            # Concatenate the rotation, translation and augmentation to get transformation matrix
            t = np.hstack((r,np.transpose(np.matrix(p))))
            t = np.vstack((t,[0,0,0,1]))

            # Ensure the size of the matrix is correct
            assert(len(t) == 4)
            assert(len(t[0] == 4))

            # Add back to dictionary
            transformationMatrices[jointName] = t

        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName, thetas = None):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        #TODO modify from here
        # Hint: return two numpy arrays, a 3x1 array for the position vector,
        # and a 3x3 array for the rotation matrix

        # Make sure joint name is valid
        if jointName not in self.jointRotationAxis:
            raise Exception('jointName does not exist.')
        transformationMatrices = self.getTransformationMatrices(thetas)
        result = np.identity(4)

        # Find the path of the endEffector
        path = self.jointPathDict[jointName]

        # Loop through the path and multiply to get the end matrix
        for joint in path:
            result = result*transformationMatrices[joint]

        # Slice the result and return
        pos = np.transpose([result[0:3,3]])
        rotmat = result[0:3,0:3]

        return pos, rotmat

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""

        # Initialise an empty Jacobiab Matrix (3xN)
        J = np.empty((0,3))

        # Vector Jacobian
        J_V = np.empty((0,3))

        # Obtain the end effector position
        endEffectorPos = self.getJointPosition(endEffector)
        # Obtain the end effector rotation axis
        endEffectorRot = self.getJointAxis(endEffector)
       

        # Loop through the path from origin to the end effector
        path = self.jointPathDict[endEffector]
        for joint in path:
            ai = self.getJointAxis(joint)
            pi = endEffectorPos - self.getJointPosition(joint)
            pi = pi.reshape(1,3)

            # Cross the rotational matrix and positional vector of each link
            cross_product = np.cross(ai,pi)
            cross_vector = np.cross(ai,endEffectorRot)
            

            # Add the entry to the Jacobian Matrix
            J = np.vstack([J,cross_product])
            J_V = np.vstack([J_V, cross_vector])

        
        # Make sure to transpose the matrix before returning
        assert J.T.shape == (3,len(path))
        assert J_V.T.shape == (3,len(path))

        return np.vstack([J.T,J_V.T])

    # Task 1.2 Inverse Kinematics
    def inverseKinematics(self, endEffector, targetPosition, threshold, orientation=None):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            interpolationSteps: number of interpolation steps
            maxIterPerStep: maximum iterations per step
            threshold: accuracy threshold
        Return: \\
            Vector of x_refs
        """

        # Get the path to endEffector
        path = self.jointPathDict[endEffector]

        # To store current revolut position of each step
        current_q = []

        # Loop through the affected links
        for joint in path:
            current_q.append(self.getJointPos(joint))
        assert(len(current_q) == len(path))

        endEffectorPos = self.getJointPosition(endEffector).flatten()

        # Obtain the Jacobian, use the current joint configurations and E-F position
        J = self.jacobianMatrix(endEffector)
        
        if type(orientation) == type(None) or (orientation[0]) == (None): #if we dont care about orientation
            J = J[:3,:]

        # Compute the dy steps
        deltaStep = targetPosition - endEffectorPos #This gets updated every step of the way

        # Define the dy
        if(type(orientation)!=type(None) and (orientation[0])!=(None)):
            subtarget = np.array([deltaStep[0], deltaStep[1], deltaStep[2], orientation[0], orientation[1], orientation[2]])
        else:
            subtarget = np.array([deltaStep[0], deltaStep[1], deltaStep[2]])
        # Compute dq from dy and pseudo-Jacobian
        pseudoJacobian = np.linalg.pinv(J)
        rad_q = np.dot(pseudoJacobian,subtarget)

        # Update the robot configuration
        current_q = current_q + rad_q

        return current_q

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, debug=False, verbose=False, interpolationSteps = 500):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        # Obtain path to end effector
        path = self.jointPathDict[endEffector]

        # Calculate the positions the end effector should go to
        endEffectorPos = self.getJointPosition(endEffector).flatten()
        step_positions = np.linspace(endEffectorPos, targetPosition, interpolationSteps)
        assert(len(step_positions == interpolationSteps))

        # To store initial revolut position of each step
        init_q = []
        # Loop through the affected links
        for joint in path:
            init_q.append(self.getJointPos(joint))
        assert(len(init_q) == len(path))

        # Store the full trajectory here
        traj = np.empty((0,len(path)))
        traj = np.vstack((traj,init_q))

        # Compute the initial distance to target
        distanceToTaget = np.linalg.norm(endEffectorPos - targetPosition)
        pltDistance = [distanceToTaget] # Take z axis here
        pltTime = [0]


        orientation_current = self.getJointAxis(endEffector)
        interpolationOrientation = np.linspace(orientation_current, orientation, interpolationSteps)
        # Loop through interpolation steps
        for i in range(interpolationSteps):

            # Return revolut angles for the next one interpolation step
            nextStep = self.inverseKinematics(endEffector, step_positions[i], threshold, interpolationOrientation[i])

            # Update target joint positions with next step to shared dictionary
            for j, joint in enumerate(path):
                self.jointTargetPos[joint] = nextStep[j]
                # self.p.resetJointState(self.robot, self.jointIds[joint], nextStep[j])

            # One step in the simulation
            self.tick_without_PD(path)
            #print('One Step', i)

            traj = np.vstack((traj,nextStep))

            # Compute the endEffector position after the move
            endEffectorPos = self.getJointPosition(endEffector).flatten()
            distanceToTaget = np.linalg.norm(endEffectorPos - targetPosition)

            # Update Plotting
            pltTime.append(pltTime[-1]+ 1/240)
            pltDistance.append(distanceToTaget)

            # Check if we are already within accuracy threshold
            err = np.absolute(endEffectorPos - targetPosition)
            err_res = err < threshold
            if err_res.all():
                print('Target reached within threshold')
                print('iteration', i, 'of', interpolationSteps)
                break

        # Return plotting
        pltTime = np.array(pltTime)
        pltDistance = np.array(pltDistance)
        return pltTime, pltDistance

    def tick_without_PD(self, path):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
            # For each joint, you can use the shared variable self.jointTargetPos.
        for joint in path:
            self.p.resetJointState(self.robot, self.jointIds[joint], self.jointTargetPos[joint])

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)


    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd, joint = None):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        # TODO: Add your code here
        u_t = kp*(x_ref - x_real) + kd*(dx_ref - dx_real) + ki*(integral)
        #print("TORQUE: {} Joint: {} P: {} x_diff: {} D: {} dx_diff: {}".format(u_t, joint, kp*(x_ref - x_real), (x_ref - x_real), kd*(dx_ref - dx_real), (dx_ref - dx_real)))
        return u_t

    # functions for testing
    def trajectoryCubicTuning(self, joint, targetPosition, targetVelocity, toy_tick):

        interpolationSteps = 2000
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [0], [], [0], [0], [], [0]

        # Calculate the positions the end effector should go to
        endEffectorRevPos = self.getJointPos(joint)
        
        #Just how Cubic Hermite Spline works
        xs = np.array([endEffectorRevPos, targetPosition])
        xs.sort() #The control points Xs need to be in an increasing order.
        ys = [np.sin(x + targetPosition) + 1  for x in xs ] #changes representation in the spline
        dydx = [0]*2 #the spline thing-y needs the rate of change at the ends of the said control points, which in our case is 0 -> 0 velocity; doesn't really work, hence padding is needed. more on this below.
        interp_func = CubicHermiteSpline(xs, xs, dydx)
        x_points = np.linspace(endEffectorRevPos, targetPosition, interpolationSteps)
        pltTarget = np.append(pltTarget,[interp_func(endEffectorRevPos)]*2) #pads the initial position in the target to ensure that zero velocity condition actually exists at the beginning
        pltTarget = np.append(pltTarget,interp_func(x_points))

        pltTarget = np.append(pltTarget,[interp_func(targetPosition)]*4) #pads final "target" position at the end to ensure that the difference with targer position is VERY negligible, and also helps reaching (tending towards) a 0 velocity state
        assert(len(pltTarget == interpolationSteps+1))

        pltPosition.append(self.getJointPos(joint)) #this is done to maintain the length of the plot given all the padding that's done in this method
        pltPosition.append(self.getJointPos(joint))

        # Loop through interpolation steps
        for i in range(1,interpolationSteps+5): #"+5" is due to the said pads
        
            x_ref = pltTarget[i]
            x_real = self.getJointPos(joint)
            
            approximatedRealVelocity = (pltPosition[i]-pltPosition[i-1])/self.dt #approximates real velocity, since we can't call the actual method RE:prohibited api. Turns out, works pretty well! Diff in order of e-4 or less!
            refVelocity = (pltTarget[i]-pltTarget[i-1])/self.dt #ref velocity; what is actually used as a make-do way of attaining 0 velocity conditions
            toy_tick(x_ref, x_real, refVelocity, approximatedRealVelocity, 0, pltTorque) #OP-est of all methods!
            if len(pltTime) == 0:
                pltTime.append(self.dt)
            else:
                pltTime.append(self.dt + pltTime[-1])
            pltPosition.append(self.getJointPos(joint))
            pltVelocity.append(self.getJointVel(joint))
       
        pltTorqueTime = pltTime
        print('Goal: {}, Current: {}'.format(np.rad2deg(targetPosition), np.rad2deg(self.getJointPos(joint))))

        pltPosition = pltPosition[1:]
        pltTarget = pltTarget[1:]
        
        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    # functions for testing
    def trajectoryLinearTuning(self, joint, targetPosition, targetVelocity, toy_tick):

        interpolationSteps = 2000
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        # Calculate the positions the end effector should go to
        endEffectorRevPos = self.getJointPos(joint)
        print('Initial joint angle: {}'.format(np.rad2deg(endEffectorRevPos)))
        subtargets = np.linspace(endEffectorRevPos, targetPosition, interpolationSteps) # The reference small target steps the effector should try to reach
        assert(len(subtargets == interpolationSteps))
    
        # Find the targetjointRev velocity for each step_positions
        velRefs = []
        for i in range(len(subtargets)-1):
            diff = subtargets[i+1] - subtargets[i]
            vel = diff/self.dt
            velRefs.append(vel)
        velRefs.append(targetVelocity)
        assert(len(velRefs) == len(subtargets))

        # Loop through all the subtargets
        for i in range(len(subtargets)):
            # For the first 2 data points where you cannot approximate
            if i < 2:
                approximatedRealVelocity = 0
            else:
                approximatedRealVelocity = (pltPosition[i-1]-pltPosition[i-2])/self.dt

            toy_tick(subtargets[i], self.getJointPos(joint), velRefs[i], approximatedRealVelocity, 0, pltTorque)
            
            if len(pltTime) == 0:
                 pltTime.append(self.dt)
            else:
                pltTime.append(self.dt + pltTime[-1])
            pltTarget.append(subtargets[i])
            pltPosition.append(self.getJointPos(joint))
            pltVelocity.append(self.getJointVel(joint))
       
        pltTorqueTime = pltTime
        print('Goal: {}, Current: {}'.format(np.rad2deg(targetPosition), np.rad2deg(self.getJointPos(joint))))

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity
    
    # Function for tuning under a single end target angle
    def targetTuning(self, joint, targetPosition, targetVelocity, toy_tick):

        steps = 1000
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        for i in range(steps):
            # For the first 2 data points where you cannot approximate
            if i < 2:
                approximatedRealVelocity = 0
            else:
                approximatedRealVelocity = (pltPosition[i-1]-pltPosition[i-2])/self.dt
            
            # Tick one step of simulation
            toy_tick(targetPosition, self.getJointPos(joint), targetVelocity, approximatedRealVelocity , 0, pltTorque)
            
            if len(pltTime) == 0:
                 pltTime.append(self.dt)
            else:
                pltTime.append(self.dt + pltTime[-1])
            pltTarget.append(targetPosition)
            pltPosition.append(self.getJointPos(joint))
            pltVelocity.append(self.getJointVel(joint))
       
        pltTorqueTime = pltTime
        print('Goal: {}, Current: {}'.format(np.rad2deg(targetPosition), np.rad2deg(self.getJointPos(joint))))
        
        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False, interpolationSteps = 2000):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral, pltTorque):
            # Loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)
        print('Target angle of the joint: {}'.format(np.rad2deg(targetPosition)))

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        # Calculate the positions the end effector should go to
        endEffectorRevPos = self.getJointPos(joint)
        print('Initial joint angle: {}'.format(np.rad2deg(endEffectorRevPos)))

        ################### Three tuning methods, Uncomment to select ########################

        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = self.trajectoryCubicTuning(joint, targetPosition, targetVelocity, toy_tick)
        # pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = self.trajectoryLinearTuning(joint, targetPosition, targetVelocity, toy_tick)
        # pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = self.targetTuning(joint, targetPosition, targetVelocity, toy_tick)

        # Check the joint angle against target       
        distThreshold = 0.035 # Given on the lab guidebook
        deltaAngle = abs(self.getJointPos(joint) - targetPosition)
        if deltaAngle <= distThreshold:
            print('Threhold met, difference is {}'.format(np.rad2deg(deltaAngle)))
        else:
            print('Goal Failed, difference is {}'.format(np.rad2deg(deltaAngle)))

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    '''
    Function computes hermite interpolation given 2 points, and returns a function
    '''
    def hermiteInterpolation(self, point1, point2, dydx = None):
        xs = np.array([point1, point2])
        xs.sort() #The control points Xs need to be in an increasing order.
        ys = [np.sin(x + point2) + 1  for x in xs ] #changes representation in the spline
        if(type(dydx) == type(None)):
            dydx = [0]*2 #the spline thing-y needs the rate of change at the ends of the said control points, which in our case is 0 -> 0 velocity; doesn't really work, hence padding is needed. more on this below.
        interp_func = CubicHermiteSpline(xs, xs, dydx)
        return interp_func
        
    def move_with_PD(self, endEffector, targetPosition, interpolationSteps=2000, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Obtain path to end effector
        path = self.jointPathDict[endEffector]

        # Calculate the positions the end effector should go to
        endEffectorPos = self.getJointPosition(endEffector).flatten()
        
        # Matrix that will store the interpolated path
        step_angles= np.empty((0,3))

        # Initialise arrays for interpolation on x, y and z axis
        xpoints, ypoints, zpoints = [], [], []
        
        # Points on all 3 axis interpolated over steps defined
        points = np.linspace(endEffectorPos, targetPosition, interpolationSteps)
        
        # If the end efector does not move in a certain axis, hold the points the same,
        # Else use cubic hermit spline

        # x-axis
        if(endEffectorPos[0]==targetPosition[0]):
            xpoints = points[:, 0]
        else:
            xHermite = self.hermiteInterpolation(endEffectorPos[0],targetPosition[0])
            xpoints = xHermite(points[:,0])
        # y-axis
        if(endEffectorPos[1]==targetPosition[1]):
            ypoints = points[:,1]
        else:
            yHermite = self.hermiteInterpolation(endEffectorPos[1],targetPosition[1])
            ypoints = yHermite(points[:,1])
        # z-axis
        if(endEffectorPos[2]==targetPosition[2]):
            zpoints = points[:,2]
        else:
            zHermite = self.hermiteInterpolation(endEffectorPos[2],targetPosition[2])
            zpoints = zHermite(points[:,2])
        
        # Fill in the matrix with interpolation steps
        step_angles = np.hstack((np.c_[xpoints], np.c_[ypoints], np.c_[zpoints])) 
        step_angles = np.vstack([step_angles[0], step_angles])
        step_angles = np.vstack([step_angles[0], step_angles])
        #pads final "target" position at the end to ensure that the difference with targer position is VERY negligible, and also helps reaching (tending towards) a 0 velocity state
        for _ in range(4):
            step_angles = np.vstack([step_angles, step_angles[-1]])

        # To store initial revolut position of each step
        init_q = []
        init_q_const = {}
        # Loop through the affected links
        for joint in self.jointList:
            if joint in path:
                init_q.append(self.getJointPos(joint))
            else:
                init_q_const[joint] = self.getJointPos(joint)

        # Compute the initial distance to target
        distanceToTaget = np.linalg.norm(endEffectorPos - targetPosition)
        pltDistance = [distanceToTaget] 
        pltTime = [0]
        
        # Dictionary stores revolut angle velocity histories for each joint
        jointPosiHist = {}
        for joint in path:
            jointPosiHist[joint] = [self.getJointPos(joint)]
        
        # Dictionary stores the approximated real joint revolut velocity
        jointVelReal = {}
        for joint in path:
            jointVelReal[joint] = 0

        orientation_current = self.getJointAxis(endEffector)
        interpolatedOrientation = [orientation_current]
        interpolatedOrientation = np.vstack([interpolatedOrientation, np.linspace(orientation_current, orientation, len(step_angles)-1)])
        
        # Loop through interpolation steps
        for i in range(1, len(step_angles)):

            # Return revolut angles for the next one interpolation step
            nextStep = self.inverseKinematics(endEffector, step_angles[i], threshold, interpolatedOrientation[i])
            # Update target joint positions with next step to shared dictionary
            for j, joint in enumerate(path):
                self.jointTargetPos[joint] = nextStep[j]

                # Calculate the approximated revolut angle
                jointPosiHist[joint] = np.append(jointPosiHist[joint], self.getJointPos(joint))
                jointVelReal[joint] =  (jointPosiHist[joint][i]-jointPosiHist[joint][i-1])/self.dt
            
            for joint in init_q_const:
                self.jointTargetPos[joint] = init_q_const[joint]

            # One step in the simulation
            self.tick(path=path,realVelDict= jointVelReal)

            # Compute the endEffector position after the move
            endEffectorPos = self.getJointPosition(endEffector).flatten()
            distanceToTaget = np.linalg.norm(endEffectorPos - targetPosition)

            # Update Plotting
            pltTime.append(pltTime[-1]+ self.dt)
            pltDistance.append(distanceToTaget)

            # Check if we are already within accuracy threshold
            err = np.absolute(endEffectorPos - targetPosition)
            err_res = err < threshold
            if err_res.all():
                print('Target reached within threshold')
                print('iteration', i, 'of', len(step_angles))
                break

        # Return plotting
        pltTime = np.array(pltTime)
        pltDistance = np.array(pltDistance)

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).
        return pltTime, pltDistance

    def tick(self, path=None, realVelDict=None):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)
            # print(self.ctrlConfig)
            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p'] 
            ki = self.ctrlConfig[jointController]['pid']['i'] 
            kd = self.ctrlConfig[jointController]['pid']['d'] 
            
            
            torque = 0
            if path != None:
                #print(path)
                if joint in path:
                    if realVelDict != None:
                        torque = self.calculateTorque(self.jointTargetPos[joint], self.getJointPos(joint), (self.jointTargetPos[joint]-self.getJointPos(joint))/self.dt, realVelDict[joint], 0, kp, ki, kd, joint = joint)
                    else:
                        torque = 0

                elif joint not in path:
                    torque = self.calculateTorque(self.jointTargetPos[joint], self.getJointPos(joint), (self.jointTargetPos[joint]-self.getJointPos(joint))/self.dt, 0, 0, kp, ki, kd, joint = joint)

            #print('Joint {} with torque {}'.format(joint, torque))
            ### ... to here ###
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = 9.81
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )

            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)
    

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100): #don't really use as we are using hermite cubic interpolation
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        #TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        #return xpoints, ypoints
        x = points[0]
        y = points[1]
        # apply cubic spline interpolation
        cs = CubicSpline(x, y, bc_type=((2, 0.0), (2, 0.0)))
        xpoints = np.linspace(points[0], points[-1], nTimes)
        ypoints = cs(xpoints)

        return xpoints, ypoints

    #returns interpolated points in 3d
    def hermiteInterpolation3D(self, endEffectorPos, targetPosition, interpolationSteps, dydx):
        # Points on all 3 axis interpolated over steps defined
        points = np.linspace(endEffectorPos, targetPosition, interpolationSteps)
        # If the end efector does not move in a certain axis, hold the points the same,
        # Else use cubic hermit spline
        xpoints, ypoints, zpoints = [], [], []
        # x-axis
        if(endEffectorPos[0]==targetPosition[0]):
            xpoints = points[:, 0]
        else:
            xHermite = self.hermiteInterpolation(endEffectorPos[0],targetPosition[0], dydx = dydx)
            xpoints = xHermite(points[:,0])
        # y-axis
        if(endEffectorPos[1]==targetPosition[1]):
            ypoints = points[:,1]
        else:
            yHermite = self.hermiteInterpolation(endEffectorPos[1],targetPosition[1], dydx = dydx)
            ypoints = yHermite(points[:,1])
        # z-axis
        if(endEffectorPos[2]==targetPosition[2]):
            zpoints = points[:,2]
        else:
            zHermite = self.hermiteInterpolation(endEffectorPos[2],targetPosition[2], dydx = dydx)
            zpoints = zHermite(points[:,2])
        
        # Fill in the matrix with interpolation steps
        trajectory = np.hstack((np.c_[xpoints], np.c_[ypoints], np.c_[zpoints])) 
        trajectory = np.vstack([trajectory[0], trajectory])
        trajectory = np.vstack([trajectory[0], trajectory])
        #pads final "target" position at the end to ensure that the difference with targer position is VERY negligible, and also helps reaching (tending towards) a 0 velocity state
        for _ in range(4):
            trajectory = np.vstack([trajectory, trajectory[-1]])
        return trajectory

    def moveTwoEndEffectors (self, left_pos, right_pos, pathLeft, pathRight, jointPosiHist, jointVelReal, left_orientation, right_orientation, threshold = 0.0035,iterSteps = 1000, effLeft = "LARM_JOINT5", effRight = "RARM_JOINT5"):
        for i in range(0, iterSteps):
            if(i%100 == 0):
                print("iter {} of {}".format(i,iterSteps))
            # Return revolut angles for left endeff for the next one interpolation step
            nextStepLeft = self.inverseKinematics(effLeft, left_pos[i], threshold, left_orientation[i])
            # Return revolut angles for right endeff for the next one interpolation step
            nextStepRight = self.inverseKinematics(effRight, right_pos[i], threshold, right_orientation[i])
            # Update target joint positions, position history and real velocities with next step to shared dictionary for right end effector
            for joint in self.jointList:
                self.jointTargetPos[joint] = self.getJointPos(joint)
                jointPosiHist[joint] = np.append(jointPosiHist[joint], self.getJointPos(joint))
                jointVelReal[joint] =  (jointPosiHist[joint][i]-jointPosiHist[joint][i-1])/self.dt
            for jRight, jointRight in enumerate(pathRight):
                self.jointTargetPos[jointRight] = nextStepRight[jRight]
            #give torque to right end effector
            self.tick(path=pathRight, realVelDict=jointVelReal)


            # Update target joint positions, position history and real velocities with next step to shared dictionary for left end effector
            for jLeft, jointLeft in enumerate(pathLeft): 
                self.jointTargetPos[jointLeft] = nextStepLeft[jLeft]
            for joint in self.jointList:
                jointPosiHist[joint] = np.append(jointPosiHist[joint], self.getJointPos(joint))
                jointVelReal[joint] =  (jointPosiHist[joint][i]-jointPosiHist[joint][i-1])/self.dt
            #give torque to left end effector
            self.tick(path=pathLeft, realVelDict=jointVelReal)
            
            
            #print("left :", self.getJointPosition("LARM_JOINT5")," right :", self.getJointPosition("RARM_JOINT5"))
            


    # Task 3.2 Grasping & Docking
    def clamp(self, targetPositions, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False, interpolationSteps = 2000, y_diff= 0.2, x_diff = 0.2, z_diff = 0.2):
        # Obtain path to end effector
        effLeft = "LARM_JOINT5"
        effRight = "RARM_JOINT5"
        pathLeft = self.jointPathDict[effLeft]
        pathRight = self.jointPathDict[effRight]


        # Matrix that will store the interpolated path
        trajectory_post_clamping_left_eff= np.empty((0,3))
        trajectory_post_clamping_right_eff= np.empty((0,3))
        targetPositionsLeft, targetPositionsRight= np.array(targetPositions), np.array(targetPositions)
        
        #the x,y,z differences are used to accomodate for the dumbbell being in the middle with the two "hands" on the sides
        targetPositionsLeft[:,0] -= x_diff
        targetPositionsLeft[:,1] += y_diff
        
        targetPositionsRight[:,1] -= y_diff
    
        targetPositionsLeft[:-1,2] += z_diff
        targetPositionsRight[:-1,2] += z_diff
        
         # Calculate the positions the end effector should go to
        leftStartPos = targetPositionsLeft[0]
        rightStartPos = targetPositionsRight[0]
        

        #interpolate the trajectories in terms of 3D coordinates from clamping to docking
        for i, left_waypoint in enumerate(targetPositionsLeft[1:]):
            dydx = np.ones(2)
            trajectory_post_clamping_left_eff = np.vstack([trajectory_post_clamping_left_eff, self.hermiteInterpolation3D(leftStartPos, left_waypoint, interpolationSteps//len(targetPositions), dydx)])
            leftStartPos = left_waypoint
        for i,right_waypoint in enumerate(targetPositionsRight[1:]):
            dydx = np.ones(2)
            trajectory_post_clamping_right_eff = np.vstack([trajectory_post_clamping_right_eff, self.hermiteInterpolation3D(rightStartPos, right_waypoint, interpolationSteps//len(targetPositions),dydx)])
            rightStartPos = right_waypoint
            
        
        # Dictionary stores revolut angle velocity histories for each joint
        jointPosiHist = {}
        for joint in self.jointList:
            jointPosiHist[joint] = [self.getJointPos(joint),self.getJointPos(joint)]
        
        # Dictionary stores the approximated real joint revolut velocity
        jointVelReal = {}
        for joint in self.jointList:
            jointVelReal[joint] = 0
    
        #to reduce non-zero velocity errors that may destabilise the system
        iterations_to_stability = 20
        
        #divides the iteration steps and the quotient is used to decide how many iterations for both fixing orientation and for clamping (applicable separately)
        initial_clamp_iteration_scale = 10
        
        #bent orientation is extremely useful in effectively transporting the dumbbell without changing torso configuration explicitly
        left_orientation_goal = [-1/4,1/1.4,0]
        right_orientation_goal = [1/4,-1/1.4,0]
        left_orientation_current = self.getJointAxis(effLeft) #returns current end effector orientation for left
        right_orientation_current = self.getJointAxis(effRight) #returns current end effector orientation for right
        left_orientation = np.linspace(left_orientation_current, left_orientation_goal, len(trajectory_post_clamping_left_eff)//initial_clamp_iteration_scale) #interpolates orientation
        left_orientation = np.vstack([left_orientation, [left_orientation[-1]]*iterations_to_stability]) #padding for stabilising orientation
        right_orientation = np.linspace(right_orientation_current, right_orientation_goal, len(trajectory_post_clamping_left_eff)//initial_clamp_iteration_scale) #interpolates orientation
        right_orientation = np.vstack([right_orientation, [right_orientation[-1]]*iterations_to_stability]) #padding for stabilising orientation

        # moves end effectors up and away from the dumbbell to make the orientation change stabler (especially because we dont want the left end effector to collide with the obstacle)
        left_init_pos = self.getJointPosition(effLeft).flatten()
        left_first_goal_pos = 0 + left_init_pos
        left_first_goal_pos[1] += 0.04
        left_first_goal_pos[2] += 0.12
        left_pos = np.linspace(left_init_pos, left_first_goal_pos, len(trajectory_post_clamping_left_eff)//initial_clamp_iteration_scale)
        left_pos = np.vstack([left_pos, [left_pos[-1]]*iterations_to_stability])
        right_init_pos = self.getJointPosition(effRight).flatten()
        right_first_goal_pos = 0 + right_init_pos
        right_first_goal_pos[1] -= 0.04
        right_first_goal_pos[2] += 0.12
        right_pos = np.linspace(right_init_pos, right_first_goal_pos, len(trajectory_post_clamping_left_eff)//initial_clamp_iteration_scale)
        right_pos = np.vstack([right_pos, [right_pos[-1]]*iterations_to_stability])
       
        #after orientation is fixed, we move to clamp the dumbbell
        left_pos_to_bring_back = self.hermiteInterpolation3D(left_pos[-1], np.array(targetPositionsLeft[0]), len(trajectory_post_clamping_left_eff)//initial_clamp_iteration_scale,[0,0])
        right_pos_to_bring_back =  self.hermiteInterpolation3D(right_pos[-1], np.array(targetPositionsRight[0]), len(trajectory_post_clamping_left_eff)//initial_clamp_iteration_scale,[0,0])
        left_pos = np.vstack([left_pos, left_pos_to_bring_back])
        right_pos = np.vstack([right_pos, right_pos_to_bring_back])
        
        #we add the steps needed to reach the actual target and dock
        left_pos = np.vstack([left_pos, trajectory_post_clamping_left_eff])
        right_pos = np.vstack([right_pos, trajectory_post_clamping_right_eff])

        left_orientation = np.vstack([left_orientation,[left_orientation[-1]]* ((int)(len(left_pos) - len(left_orientation)))])
        right_orientation = np.vstack([right_orientation,[right_orientation[-1]]* ((int)(len(right_pos) - len(right_orientation)))])
        
        #moves the two end effectors once provided with the required coordinate trajectories and orientations required
        self.moveTwoEndEffectors (left_pos, right_pos, pathLeft, pathRight,  jointPosiHist, jointVelReal, left_orientation, right_orientation,iterSteps = len(left_pos))
        time.sleep(4)
 ### END

