import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def load_data(filepath):
    pokemon = []
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(len(pokemon) < 20):
                pokemon.append(row)
            else:
                break
        csvfile.close()
    # cast needed values to int
    for dict in pokemon:
        dict['#'] = int(dict['#'])
        dict['Total'] = int(dict['Total'])
        dict['HP'] = int(dict['HP'])
        dict['Attack'] = int(dict['Attack'])
        dict['Defense'] = int(dict['Defense'])
        dict['Sp. Atk'] = int(dict['Sp. Atk'])
        dict['Sp. Def'] = int(dict['Sp. Def'])
        dict['Speed'] = int(dict['Speed'])
        dict.pop('Generation')
        dict.pop('Legendary')
    return pokemon

def calculate_x_y(stats):
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return (x, y)

def hac(dataset):
    #create dictionary of pokemon as tuples
    dataset = filterData(dataset)
    m = len(dataset)
    mxn = {} # index of pokemon will return its tuple
    rtrnArr = np.empty((0, 4))
    index = 0
    for row in dataset:
        mxn[index] = []
        mxn[index].append(row)
        index += 1

    #create and fill distance matrix:
    dMatrix = np.empty((2*m, 2*m))
    dMatrix[:] = np.nan
    i = 0
    while(i <= m-1):
        j = 0
        while(i>j):
            dMatrix[i][j] = distance(mxn[i][0], mxn[j][0])
            j += 1
        i += 1
    
    q = m
    while(True):
        # Get coordinates of min distance and add a new row to the return matrix
        minCoord = find_min(dMatrix)
        if(minCoord == (-1, -1)):
            break
        minD = np.nanmin(dMatrix)
        rtrnArr = np.append(rtrnArr, np.array([[minCoord[0], minCoord[1], minD, (len(mxn[minCoord[0]]) + len(mxn[minCoord[1]]))]]), axis=0)
        # update mxn
        mxn[q] = []
        mxn[q].extend(mxn[minCoord[0]])
        mxn[q].extend(mxn[minCoord[1]])
        mxn.pop(minCoord[0])
        mxn.pop(minCoord[1])
        # Set the value of this coord to nan
        dMatrix[minCoord[0]][minCoord[1]] = np.nan

        #update distances based off distance matrix
        i = 0
        while(i <= q):
            if(i == minCoord[0]):
                dMatrix[i][i] = np.nan
            if(i == minCoord[1]):
                dMatrix[i][i] = np.nan
            if(i > minCoord[0] and i > minCoord[1]):
                dMatrix[q][i] = min(dMatrix[i][minCoord[0]], dMatrix[i][minCoord[1]])
            elif(i > minCoord[0] and i < minCoord[1]):
                dMatrix[q][i] = min(dMatrix[i][minCoord[0]], dMatrix[minCoord[1]][i])
            elif(i < minCoord[0] and i < minCoord[1]):
                dMatrix[q][i] = min(dMatrix[minCoord[0]][i], dMatrix[minCoord[1]][i])
            else:
                dMatrix[q][i] = min(dMatrix[minCoord[0]][i], dMatrix[i][minCoord[1]])
            i+=1
        # clear old distance values
        dMatrix[minCoord[0], :] = np.nan
        dMatrix[minCoord[1], :] = np.nan
        dMatrix[:, minCoord[0]] = np.nan
        dMatrix[:, minCoord[1]] = np.nan
        q+=1
        
    return rtrnArr

def random_x_y(m):
    rtrn = []
    i = 0
    while(i < m):
        rtrn.append( (np.random.randint(360), np.random.randint(360)) )
        i+=1
    return rtrn

def imshow_hac(dataset):
    #create dictionary of pokemon as tuples
    dataset = filterData(dataset)
    m = len(dataset)
    mxn = {} # index of pokemon will return its tuple
    rtrnArr = np.empty((0, 4))
    index = 0
    for row in dataset:
        mxn[index] = []
        mxn[index].append(row)
        index += 1
 
    #create and fill distance matrix:
    dMatrix = np.empty((2*m, 2*m))
    dMatrix[:] = np.nan
    i = 0
    while(i <= m-1):
        j = 0
        while(i>j):
            dMatrix[i][j] = distance(mxn[i][0], mxn[j][0])
            j += 1
        i += 1
    
    x = np.array([row[0] for row in dataset])
    y = np.array([row[1] for row in dataset])
    plt.scatter(x, y)
 
    q = m
    while(True):
        plt.pause(.1)
        # Get coordinates of min distance and add a new row to the return matrix
        minCoord = find_min(dMatrix)
        if(minCoord == (-1, -1)):
            break
        minD = np.nanmin(dMatrix)
        rtrnArr = np.append(rtrnArr, np.array([[minCoord[0], minCoord[1], minD, (len(mxn[minCoord[0]]) + len(mxn[minCoord[1]]))]]), axis=0)
        # Get min distance for dMatrix, find out which points are being compared, use those two in x and yValues
        minDistance = float('inf')
        point1 = {}
        point2 = {}
        # Dijkstra's algo could optimize the mapping, but would entail rewriting entire hac method
        for point in mxn[minCoord[0]]:
            for coord in mxn[minCoord[1]]:
                if(distance(point, coord) < minDistance):
                    point1[0] = point
                    point2[0] = coord
                    minDistance = distance(point, coord)
        plt.plot([point1[0][0], point2[0][0]], [point1[0][1], point2[0][1]])
        # update mxn
        mxn[q] = []
        mxn[q].extend(mxn[minCoord[0]])
        mxn[q].extend(mxn[minCoord[1]])
        mxn.pop(minCoord[0])
        mxn.pop(minCoord[1])
        # Set the value of this coord to nan
        dMatrix[minCoord[0]][minCoord[1]] = np.nan
        #update distances based off distance matrix
        i = 0
        while(i <= q):
            if(i == minCoord[0]):
                dMatrix[i][i] = np.nan
            if(i == minCoord[1]):
                dMatrix[i][i] = np.nan
            if(i > minCoord[0] and i > minCoord[1]):
                dMatrix[q][i] = min(dMatrix[i][minCoord[0]], dMatrix[i][minCoord[1]])
            elif(i > minCoord[0] and i < minCoord[1]):
                dMatrix[q][i] = min(dMatrix[i][minCoord[0]], dMatrix[minCoord[1]][i])
            elif(i < minCoord[0] and i < minCoord[1]):
                dMatrix[q][i] = min(dMatrix[minCoord[0]][i], dMatrix[minCoord[1]][i])
            else:
                dMatrix[q][i] = min(dMatrix[minCoord[0]][i], dMatrix[i][minCoord[1]])
            i+=1
        # clear old distance values
        dMatrix[minCoord[0], :] = np.nan
        dMatrix[minCoord[1], :] = np.nan
        dMatrix[:, minCoord[0]] = np.nan
        dMatrix[:, minCoord[1]] = np.nan
        q+=1

    plt.show()
 
#Helper Functions
#___________________________________________________________________________________________________

def distance(x, y):
    return math.sqrt( math.pow((y[0] - x[0]), 2) + math.pow((y[1] - x[1]), 2) )

def filterData(dataset):
    for tup in dataset:
        if(len(tup)!= 2):
            dataset.remove(tup)
        elif(np.isnan(tup[0]) or np.isnan(tup[1]) or np.isinf(tup[0]) or np.isinf(tup[1])):
            dataset.remove(tup)
    return dataset

def find_min(dMatrix):
    # numpy nanmin was confusing so I wrote my own
    row = len(dMatrix[0, :]) - 1
    column = len(dMatrix[:, 0]) - 1
    minVal = float('inf')
    minRow = 0
    minCol = 0
    # get min val, go from high index to low to return minimum with lowest index (auto tie-breaker)
    while(row >= 0):
        while(column >= 0):
            if(np.isnan(dMatrix[row][column])):
                column -= 1
            else:
                if(dMatrix[row][column] < minVal):
                    minVal = dMatrix[row][column]
                    minRow = row
                    minCol = column
                column -= 1
        row -= 1
        column = len(dMatrix[:, 0]) - 1
    if(math.isinf(minVal)):
        return (-1, -1)
    if(minRow > minCol):
        return (minCol, minRow)
    else:
        return (minRow, minCol)
    
    

