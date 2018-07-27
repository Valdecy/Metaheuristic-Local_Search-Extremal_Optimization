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
        distance = distance + Xdata.iloc[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x.iloc[j] - y.iloc[j])**2 + distance   
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
    Xdata = pd.DataFrame(np.zeros((coordinates.shape[0], coordinates.shape[0])))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates.iloc[i,:]
                y = coordinates.iloc[j,:]
                Xdata.iloc[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = Xdata.copy(deep = True)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m.iloc[i,j] = (1/2)*(Xdata.iloc[0,j]**2 + Xdata.iloc[i,0]**2 - Xdata.iloc[i,j]**2)    
    m = m.values
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    coordinates = coordinates.values
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Rank Cities
def ranking(Xdata, city = 0, tau = 1.8):
    rank = pd.DataFrame(np.zeros((Xdata.shape[0], 4)), columns = ['Distance', 'City', 'Rank', 'Probability'])
    for i in range(0, rank.shape[0]):
        rank.iloc[i,0] = Xdata.iloc[i,city]
        rank.iloc[i,1] = i + 1
    rank = rank.sort_values(by = 'Distance')
    for i in range(0, rank.shape[0]):
        rank.iloc[i,2] = i
        if (i> 0):
            rank.iloc[i,3] = i**(-tau)
    sum_prob = rank.iloc[:, 3].sum()
    for i in range(0, rank.shape[0]):
        rank.iloc[i, 3] = rank.iloc[i, 3]/sum_prob
    rank = rank.sort_values(by = 'Probability')
    for i in range(1, rank.shape[0]):
        rank.iloc[i,3] = rank.iloc[i,3] + rank.iloc[i-1,3]
    return rank

# Function: Selection
def roulette_wheel(rank, city_tour, tau = 1.8):
    fitness = pd.DataFrame(np.zeros((rank.shape[0], 5)), columns = ['City', 'Left', 'Right', 'Fitness', 'Probability'])
    fitness['City']  = city_tour[0][0:-1]
    fitness['Left']  = city_tour[0][-2:-1] + city_tour[0][0:-2]
    fitness['Right'] = city_tour[0][1:]
    for i in range(0, fitness.shape[0]):
        left  = rank.loc[rank['City'] == fitness.iloc[i, 1]]
        right = rank.loc[rank['City'] == fitness.iloc[i, 2]]
        fitness.iloc[i, 3] = 3/(left.iloc[0,2] + right.iloc[0,2])    
        fitness.iloc[i, 4] = fitness.iloc[i, 3]**(-tau) 
    sum_prob = fitness.iloc[:, 4].sum()
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i, 4] = fitness.iloc[i, 4]/sum_prob
    fitness = fitness.sort_values(by = 'Probability')
    for i in range(1, fitness.shape[0]):
        fitness.iloc[i,4] = fitness.iloc[i,4] + fitness.iloc[i-1,4]
    ix =  1
    iy = -1 # left
    iz = -1 # rigth
    iw =  1 # change
    rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (rand <= fitness.iloc[i, 4]):
          ix = fitness.iloc[i, 0]
          iw = fitness.iloc[i, 0]
          left  = rank.loc[rank['City'] == fitness.iloc[i, 1]]
          right = rank.loc[rank['City'] == fitness.iloc[i, 2]]
          if (left.iloc[0,0] > right.iloc[0,0]):
              iy = fitness.iloc[i, 1]
              iz = -1
          else:
              iy = -1
              iz = fitness.iloc[i, 2]              
          break
    while (ix == iw):
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        for i in range(0, rank.shape[0]):
            if (rand <= rank.iloc[i, 3]):
              iw = fitness.iloc[i, 0]
              #iw = random.sample(list(range(1,rank.shape[0]+1)), 1)[0]
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

X = pd.read_csv('Python-MH-Local Search-Extremal Optimization-Dataset-01.txt', sep = '\t') #17 cities = 1922.33
seed = seed_function(X)
lseo = extremal_optimization(X, city_tour = seed, iterations = 100, tau = 1.5) # tau [1.2, 1.8]
plot_tour_distance_matrix(X, lseo) # Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses.

Y = pd.read_csv('Python-MH-Local Search-Extremal Optimization-Dataset-02.txt', sep = '\t') # Berlin 52 = 7544.37
X = buid_distance_matrix(Y)
seed = seed_function(X)
lseo = extremal_optimization(X, city_tour = seed, iterations = 250, tau = 1.8) # tau [1.2, 1.8]
plot_tour_coordinates (Y, lseo) # Red Point = Initial city; Orange Point = Second City
