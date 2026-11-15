#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    DT = 1e-1  
    q = qcurrent.copy() 

    idL = robot.model.getFrameId(LEFT_HAND)               
    idR = robot.model.getFrameId(RIGHT_HAND)
    oMhookL = getcubeplacement(cube, LEFT_HOOK)           
    oMhookR = getcubeplacement(cube, RIGHT_HOOK)

    for i in range(1000):
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q) 

        oMhandL = robot.data.oMf[idL]
        oMhandR = robot.data.oMf[idR] 

        ## Hand L task  
        o_handl = pin.computeFrameJacobian(robot.model, robot.data, q, idL,pin.LOCAL)
        handltoMgoal = oMhandL.inverse()* oMhookL 
        handl_nu = pin.log(handltoMgoal).vector

        # HandR task
        o_handr = pin.computeFrameJacobian(robot.model,robot.data,q,idR,pin.LOCAL)
        handrtoMgoal = oMhandR.inverse()* oMhookR
        handr_nu = pin.log(handrtoMgoal).vector

        if norm(handl_nu) < EPSILON and norm(handr_nu) < EPSILON:                 
            break

        vq = pinv(o_handl) @ handl_nu
        Ptool = np.eye(robot.nv)-pinv(o_handl) @ o_handl
        vq += pinv(o_handr @ Ptool) @ (handr_nu - o_handr @ vq)

        q = pin.integrate(robot.model,q, vq * DT)
        q = projecttojointlimits(robot, q)
        
        if viz is not None:
            viz.display(q)
        
    success = (norm(handl_nu) < EPSILON and norm(handr_nu) < EPSILON
            and not collision(robot, q))
    return q, success

            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
