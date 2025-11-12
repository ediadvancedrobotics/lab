#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import copy
import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK, CUBE_PLACEMENT
import time

from inverse_geometry_old import solve_dual_ik_3d
from tools import collision, distanceToObstacle, setcubeplacement

from inverse_geometry_old import computeqgrasppose

import matplotlib.pyplot as plt
from pinocchio.utils import rotate


left_ids = [3,4,5,6,7,8]
right_ids = [9,10,11,12,13,14]


def RAND_CONF(robot, cube, checkcollision=True):
    '''
    Return a random position for the cube and a random configuration for the robot
    '''

    while True:
        cube_location = pin.SE3(rotate('z', 0),np.random.rand(3))
        cube_location.translation[0] *= 0.6
        cube_location.translation[1] -= 0.8
        cube_location.translation[2] *= 0.6
        cube_location.translation[2] += 0.93

        # print(cube_location)

        q, success = computeqgrasppose(robot, robot.q0, cube, cube_location)

        get_cube_out_of_the_way = pin.SE3(rotate('z', 0),np.array([5,5,5]))
        setcubeplacement(robot, cube, get_cube_out_of_the_way)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
        # print(q)

        if success and not collision(robot, q):
            # updatevisuals(viz, robot, cube, q)
            return q
    

def distance(q1,q2):    
    '''Return the euclidian distance between two configurations'''
    return np.linalg.norm(q2-q1)
        
def NEAREST_VERTEX(G,q_rand):
    '''returns the index of the Node of G with the configuration closest to q_rand  '''
    min_dist = 10e4
    idx=-1
    # print('nearest vertex search...')
    for (i,node) in enumerate(G):
        # print(q_rand)
        dist = distance(node[1],q_rand) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx

def ADD_EDGE_AND_VERTEX(G,parent,q):
    G += [(parent,q)]

def lerp(q0,q1,t):    
    return q0 * (1 - t) + q1 * t

def NEW_CONF(robot, q_near,q_rand,discretisationsteps, delta_q = None):
    '''Return the closest configuration q_new such that the path q_near => q_new is the longest
    along the linear interpolation (q_near,q_rand) that is collision free and of length <  delta_q'''
    q_end = q_rand.copy()
    dist = distance(q_near, q_rand)
    if delta_q is not None and dist > delta_q:
        #compute the configuration that corresponds to a path of length delta_q
        q_end = lerp(q_near,q_rand,delta_q/dist)
        # now dist == delta_q
    dt = 1 / discretisationsteps
    for i in range(1,discretisationsteps):
        q = lerp(q_near,q_end,dt*i)
        if collision(robot, q):
            q = lerp(q_near,q_end,dt*(i-1))
            return q
    print("no collision detected along the edge?")
    return q_end


def VALID_EDGE(q_new,q_goal,discretisationsteps):
    return np.linalg.norm(q_goal -NEW_CONF(robot, q_new, q_goal,discretisationsteps)) < 1e-3

from math import ceil
from time import sleep

def displayedge(q0,q1,vel=2.): #vel in sec.    
    '''Display the path obtained by linear interpolation of q0 to q1 at constant velocity vel'''
    dist = distance(q0,q1)
    duration = dist / vel    
    nframes = ceil(48. * duration)
    f = 1./48.
    for i in range(nframes-1):
        viz.display(lerp(q0,q1,float(i)/nframes))
        sleep(f)
    viz.display(q1)
    sleep(f)
    
def displaypath_tutorial(path):
    for q0, q1 in zip(path[:-1],path[1:]):
        displayedge(q0,q1)
    


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(robot, qinit,qgoal,cubeplacementq0, cubeplacementqgoal):

    discretisationsteps_newconf = 200 #To tweak later on
    discretisationsteps_validedge = 200 #To tweak later on
    k = 1000  #To tweak later on
    delta_q = .2 #To tweak later on

    def rrt(q_init, q_goal, k, delta_q):
        G = [(None,q_init)]
        for count in range(k):
            q_rand = RAND_CONF(robot, cube)
            q_near_index = NEAREST_VERTEX(G,q_rand)
            q_near = G[q_near_index][1]        
            q_new = NEW_CONF(robot, q_near,q_rand,discretisationsteps_newconf, delta_q)
            ADD_EDGE_AND_VERTEX(G,q_near_index,q_new)
            if VALID_EDGE(q_new,q_goal,discretisationsteps_validedge):
                print ("Path found!")
                ADD_EDGE_AND_VERTEX(G,len(G)-1,q_goal)
                return G, True
            
            print(count)
            # print(q_new)
            updatevisuals(viz, robot, cube, q_new)
        print("path not found")
        return G, False
    
    def getpath(G):
        path = []
        node = G[-1]
        while node[0] is not None:
            path = [node[1]] + path
            node = G[node[0]]
        path = [G[0][1]] + path
        return path
    
    G, foundpath = rrt(qinit, qgoal, k, delta_q)

    path = foundpath and getpath(G) or []

    # print(path)

    displaypath_tutorial(path)

    # def sampleSpace(nbSamples=500):
    #     '''
    #     Sample nbSamples configurations and store them in two lists depending
    #     if the configuration is in free space (hfree) or in collision (hcol), along
    #     with the distance to the target and the distance to the obstacles.
    #     '''
    #     hcol = []
    #     hfree = []
    #     for i in range(nbSamples):
    #         q = RAND_CONF(False)
    #         if not collision(robot,q):
    #             hfree.append( list(q.flat) + [ distance(q,qgoal), distanceToObstacle(robot,q) ])
    #         else:
    #             hcol.append(  list(q.flat) + [ distance(q,qgoal), 1e-2 ])
    #     return hcol,hfree

    # def plotConfigurationSpace(hcol,hfree,markerSize=20):
    #     '''
    #     Plot 2 "scatter" plots: the first one plot the distance to the target for 
    #     each configuration, the second plots the distance to the obstacles (axis q1,q2, 
    #     distance in the color space).
    #     '''
    #     htotal = hcol + hfree
    #     h=np.array(htotal)
    #     plt.subplot(2,1,1)
    #     plt.scatter(h[:,0],h[:,1],c=h[:,2],s=markerSize,lw=0)
    #     plt.title("Distance to the target")
    #     plt.colorbar()
    #     plt.subplot(2,1,2)
    #     plt.scatter(h[:,0],h[:,1],c=h[:,3],s=markerSize,lw=0)
    #     plt.title("Distance to the obstacles")
    #     plt.colorbar()
    #     plt.tight_layout(pad=0.8)
    #     plt.show()

    # hcol,hfree = sampleSpace(2000) #increase to improve resolution
    # # print(hcol,hfree)
    # plotConfigurationSpace(hcol,hfree)


    return path

# def getpath_to_node(G, node_idx):
#     path = []
#     node = G[node_idx]
#     while node[0] is not None:
#         path = [node[1]] + path
#         node = G[node[0]]
#     path = [G[0][1]] + path
#     return path

# #returns a collision free path from qinit to qgoal under grasping constraints
# #the path is expressed as a list of configurations
# def computepath(robot, qinit,qgoal,cubeplacementq0, cubeplacementqgoal):

#     discretisationsteps_newconf = 200 #To tweak later on
#     discretisationsteps_validedge = 200 #To tweak later on
#     k = 1000  #To tweak later on
#     delta_q = .2 #To tweak later on

#     def rrt(q_init, q_goal, k, delta_q):
#         G_start = [(None,q_init)]
#         G_goal = [(None,q_goal)]
        
#         for count in range(k):
#             q_rand = RAND_CONF(robot, cube)

#             q_near_index_start = NEAREST_VERTEX(G_start,q_rand)
#             q_near_start = G_start[q_near_index_start][1]        
#             q_new_start = NEW_CONF(robot, q_near_start,q_rand,discretisationsteps_newconf, delta_q)
            
#             ADD_EDGE_AND_VERTEX(G_start,q_near_index_start,q_new_start)


#             q_near_index_goal = NEAREST_VERTEX(G_goal,q_new_start)
#             q_near_goal = G_goal[q_near_index_goal][1]
#             q_new_goal = NEW_CONF(robot, q_near_goal,q_new_start,discretisationsteps_newconf, delta_q)

#             ADD_EDGE_AND_VERTEX(G_goal,q_near_index_goal,q_new_goal)


#             if VALID_EDGE(q_new_start,q_new_goal,discretisationsteps_validedge):
#                 print ("Path found!")
#                 path_start = getpath_to_node(G_start, len(G_start)-1)
#                 path_goal  = getpath_to_node(G_goal, len(G_goal)-1)
#                 path_goal.reverse()
#                 return path_start + path_goal, True
            
#             # if count % 2 == 0:
#             #     G_start, G_goal = G_goal, G_start

#             print(count)
#             # print(q_new)
#             updatevisuals(viz, robot, cube, q_new_start)
#         print("path not found")
#         return G_start, False
    
#     def getpath(G):
#         path = []
#         node = G[-1]
#         while node[0] is not None:
#             path = [node[1]] + path
#             node = G[node[0]]
#         path = [G[0][1]] + path
#         return path
    
#     path, foundpath = rrt(qinit, qgoal, k, delta_q)

#     # path = foundpath and getpath(G) or []


#     displaypath_tutorial(path)


#     return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry_old import computeqgrasppose
    from setup_meshcat import updatevisuals

    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()

    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    # qe[9:] = [0,0,-np.pi/4,0,0,0]
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    
    # RAND_CONF(robot, cube)

    
    path = computepath(robot, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    # displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
