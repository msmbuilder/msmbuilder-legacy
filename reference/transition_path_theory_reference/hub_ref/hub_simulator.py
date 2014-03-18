
"""
A simple (dumb) simulator to check the hub score calculation.
"""
from __future__ import print_function
import sys

import numpy as np
from msmbuilder import msm_analysis

# manual parameters
steps = 10**6 # sample steps
triples = [ (0, 1, 2),     # (waypoint, source, sink) to test
            (0, 1, 3),
            (0, 1, 4) ] 


# load in the transition matrx
N = 11
T = np.transpose( np.genfromtxt('mat_1.dat')[:,:-3] )
print(T)
print(T.shape)
print(T.sum(1))

# sample from it
print("Making a len: %d traj" % steps)
traj = msm_analysis.sample(T, np.random.randint(11), steps, force_dense=True)
print("Generated traj")

# count the fraction visits

n_visited_waypoint = 0
n_notvisited_waypoint = 0

started = False
visited = False

for (waypoint, source, sink) in triples:
    
    for n,i in enumerate(traj):
        
        if n % 10000 == 0:
            print("%d/%d" % (n, steps))
        
        if i == source:
            started = True
            visited = False
            
        elif i == waypoint:
            visited = True
            
        elif i == sink:
            if started:
                if visited: n_visited_waypoint += 1
                else: n_notvisited_waypoint += 1
                started = False
                
    hc = float(n_visited_waypoint) / float(n_visited_waypoint + n_notvisited_waypoint)
    
    print("For triple:", (waypoint, source, sink))
    print("hc =", hc)
            