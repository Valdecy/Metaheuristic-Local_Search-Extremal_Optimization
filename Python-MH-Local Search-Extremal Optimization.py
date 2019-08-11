############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Extremal Optimization

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-Extremal_Optimization, File: Python-MH-Local Search-Extremal Optimization.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-Extremal_Optimization>

############################################################################

# Required Libraries
import pandas as pd
import random
import numpy  as np
import copy
import os
from matplotlib import pyplot as plt 

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def buid_distance_matrix(coordinates):
    Xdata = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates[i,:]
                y = coordinates[j,:]
                Xdata[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = np.copy(Xdata)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m[i,j] = (1/2)*(Xdata[0,j]**2 + Xdata[i,0]**2 - Xdata[i,j]**2)    
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Rank Cities
def ranking(Xdata, city = 0, tau = 1.8):
    rank = np.zeros((Xdata.shape[0], 4))
    for i in range(0, rank.shape[0]):
        rank[i,0] = Xdata[i,city]
        rank[i,1] = i + 1
    rank = rank[rank[:,0].argsort()]
    for i in range(0, rank.shape[0]):
        rank[i,2] = i
        if (i> 0):
            rank[i,3] = i**(-tau)
    sum_prob = rank[:, 3].sum()
    for i in range(0, rank.shape[0]):
        rank[i, 3] = rank[i, 3]/sum_prob
    rank = rank[rank[:,-1].argsort()]
    for i in range(1, rank.shape[0]):
        rank[i,3] = rank[i,3] + rank[i-1,3]
    return rank

# Function: Selection
def roulette_wheel(rank, city_tour, tau = 1.8):
    fitness = np.zeros((rank.shape[0], 5))
    fitness[:,0]  = city_tour[0][0:-1]
    fitness[:,1]  = city_tour[0][-2:-1] + city_tour[0][0:-2]
    fitness[:,2] = city_tour[0][1:]
    for i in range(0, fitness.shape[0]):
        left  = rank[np.where(rank[:,1] == fitness[i, 1])]
        right = rank[np.where(rank[:,1] == fitness[i, 2])]
        fitness[i, 3] = 3/(left[0,2] + right[0,2])    
        fitness[i, 4] = fitness[i, 3]**(-tau) 
    sum_prob = fitness[:, 4].sum()
    for i in range(0, fitness.shape[0]):
        fitness[i, 4] = fitness[i, 4]/sum_prob
    fitness = fitness[fitness[:,-1].argsort()]
    for i in range(1, fitness.shape[0]):
        fitness[i,4] = fitness[i,4] + fitness[i-1,4]
    ix =  1
    iy = -1 # left
    iz = -1 # rigth
    iw =  1 # change
    rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (rand <= fitness[i, 4]):
          ix    = fitness[i, 0]
          iw    = fitness[i, 0]
          left  = rank[np.where(rank[:,1] == fitness[i, 1])]
          right = rank[np.where(rank[:,1] == fitness[i, 2])]
          if (left[0,0] > right[0,0]):
              iy = fitness[i, 1]
              iz = -1
          else:
              iy = -1
              iz = fitness[i, 2]              
          break
    while (ix == iw):
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        for i in range(0, rank.shape[0]):
            if (rand <= rank[i, 3]):
              iw = fitness[i, 0]
              break      
    return iy, ix, iz, iw

# Function: Exchange
def exchange(Xdata, city_tour, iy = 1, ix = 2, iz = 3, iw = 4):
    best_route = copy.deepcopy(city_tour)    
    tour = copy.deepcopy(city_tour)
    if (iy == -1 and city_tour[0].index(iw) < city_tour[0].index(ix)):
        i = city_tour[0].index(ix) - 1
        j = city_tour[0].index(ix)
        best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
        best_route[0][-1]  = best_route[0][0]   
        i = city_tour[0].index(iw)
        j = city_tour[0].index(ix) - 1
        best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
        best_route[0][-1]  = best_route[0][0] 
        best_route[1] = distance_calc(Xdata, city_tour = best_route)
    elif (iy == -1 and city_tour[0].index(iw) > city_tour[0].index(ix)):  
        i = city_tour[0].index(ix)
        j = city_tour[0].index(iw)
        best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
        best_route[0][-1]  = best_route[0][0] 
        best_route[1] = distance_calc(Xdata, city_tour = best_route)
    elif (iz == -1 and city_tour[0].index(iw) < city_tour[0].index(ix)): 
        i = city_tour[0].index(iw)
        j = city_tour[0].index(ix)
        best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
        best_route[0][-1]  = best_route[0][0] 
        best_route[1] = distance_calc(Xdata, city_tour = best_route)
    elif (iz == -1 and city_tour[0].index(iw) > city_tour[0].index(ix)):  
        i = city_tour[0].index(ix)
        j = city_tour[0].index(ix) + 1
        best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
        best_route[0][-1]  = best_route[0][0]   
        i = city_tour[0].index(ix) + 1
        j = city_tour[0].index(iw)
        best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
        best_route[0][-1]  = best_route[0][0] 
        best_route[1] = distance_calc(Xdata, city_tour = best_route)       
    if (best_route[1] < tour[1]):
        tour[1] = copy.deepcopy(best_route[1])
        for n in range(0, len(tour[0])): 
            tour[0][n] = best_route[0][n]                        
    return tour

# Function: Extremal Optimization
def extremal_optimization(Xdata, city_tour, iterations = 50, tau = 1.8):
    count = 0
    best_solution = copy.deepcopy(city_tour)
    while (count < iterations):
        for i in range(0, Xdata.shape[0]):
            rank = ranking(Xdata, city = i, tau = tau)
            iy, ix, iz, iw = roulette_wheel(rank, city_tour, tau = tau)
            city_tour = exchange(Xdata, city_tour, iy = iy, ix = ix, iz = iz, iw = iw)
        if (city_tour[1] < best_solution[1]):
            best_solution = copy.deepcopy(city_tour) 
        count = count + 1
        city_tour = copy.deepcopy(best_solution)
        print("Iteration = ", count, "-> Distance = ", best_solution[1])
    print("Best Solution = ", best_solution)
    return best_solution

######################## Part 1 - Usage ####################################

# Load File - A Distance Matrix (17 cities,  optimal = 1922.33)
X = pd.read_csv('Python-MH-Local Search-Extremal Optimization-Dataset-01.txt', sep = '\t') 
X = X.values

# Start a Random Seed
seed = seed_function(X)

# Call the Function. tau [1.2, 1.8]
lseo = extremal_optimization(X, city_tour = seed, iterations = 250, tau = 1.5)

# Plot Solution. Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses
plot_tour_distance_matrix(X, lseo)

######################## Part 2 - Usage ####################################

# Load File - Coordinates (Berlin 52,  optimal = 7544.37)
Y = pd.read_csv('Python-MH-Local Search-Extremal Optimization-Dataset-02.txt', sep = '\t') 
Y = Y.values

# Build the Distance Matrix
X = buid_distance_matrix(Y)

# Start a Random Seed
seed = seed_function(X)

# Call the Function. tau [1.2, 1.8]
lseo = extremal_optimization(X, city_tour = seed, iterations = 500, tau = 1.8)

# Plot Solution. Red Point = Initial city; Orange Point = Second City
plot_tour_coordinates(Y, lseo)
