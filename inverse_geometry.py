#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
import time
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()

    lhand_id = robot.model.getFrameId(LEFT_HAND)
    rhand_id = robot.model.getFrameId(RIGHT_HAND)
    
    oMlhook = getcubeplacement(cube, LEFT_HOOK)
    oMrhook = getcubeplacement(cube, RIGHT_HOOK)

    success = False

    for i in range(1000):
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        oMlhand = robot.data.oMf[lhand_id]
        oMrhand = robot.data.oMf[rhand_id]

        lhandMlhook = oMlhand.inverse()*oMlhook
        lhand_nu = pin.log(lhandMlhook).vector

        rhandMrhook = oMrhand.inverse()*oMrhook
        rhand_nu = pin.log(rhandMrhook).vector

        oRlhand = oMlhand.rotation
        lhand_Jlhand = pin.computeFrameJacobian(robot.model,robot.data,q,lhand_id,pin.LOCAL_WORLD_ALIGNED)
        o_Jlhand = pin.computeFrameJacobian(robot.model,robot.data,q,lhand_id, pin.WORLD)

        oRrhand = oMrhand.rotation
        rhand_Jrhand = pin.computeFrameJacobian(robot.model,robot.data,q,rhand_id,pin.LOCAL_WORLD_ALIGNED)
        o_Jrhand = pin.computeFrameJacobian(robot.model,robot.data,q,rhand_id, pin.WORLD)

        vq = pinv(lhand_Jlhand)@lhand_nu
        Prhand = np.eye(robot.nv)-pinv(o_Jrhand) @ o_Jrhand
        vq += pinv(rhand_Jrhand @ Prhand) @ (o_Jlhand - o_Jrhand @ vq)
        
        q = pin.integrate(robot.model,q, vq)

        lerror = norm(oMlhand.translation - oMlhook.translation)
        rerror = norm(oMrhand.translation - oMrhook.translation)

        if viz:
            viz.display(q)
            time.sleep(1e-3)
        
        if lerror < EPSILON and not collision(robot, q):
            success = True
            break
    
    return q, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
