from cmath import inf, sqrt
from codecs import unicode_escape_decode
from heapq import heapify, heappop, heappush
from queue import PriorityQueue
from PIL import Image, ImageDraw
import numpy as np
import sys
import random
from bresenham import bresenham
import networkx as nx

occupancy_map_img = Image.open('occupancy_map.png')
M = (np.asarray(occupancy_map_img) > 0).astype(int)
print(M)


###########################################
#           PART A
###########################################
def RecoverPath(start, goal, pred):
    path = RecoverPathHelper(start, goal, [], pred)
    return path


def RecoverPathHelper(start, goal, path, pred):
    if (start == goal):
        return [(start[0], start[1])]
    else:
        return RecoverPathHelper(start, pred[''.join(map(str, goal))], path, pred) + [(goal[0], goal[1])]


def A_Star(V, start, goal, N, w, h):
    cost_to = {}
    est_total_cost = {}
    Q = []
    pred = {}

    for vertex in V:
        cost_to[''.join(map(str, vertex))] = inf
        est_total_cost[''.join(map(str, vertex))] = inf

    cost_to[''.join(map(str, start))] = 0
    est_total_cost[''.join(map(str, start))] = h(start, goal)
    Q.append([h(start, goal), start])

    while not (len(Q) == 0):
        vertexToExplore = heappop(Q)[1]

        if (vertexToExplore == goal):
            return RecoverPath(start, goal, pred)

        for neighbor in N(vertexToExplore):
            pvi = cost_to[''.join(map(str, vertexToExplore))] + \
                w(vertexToExplore, neighbor)

            if pvi < cost_to[''.join(map(str, neighbor))]:
                pred[''.join(map(str, neighbor))] = vertexToExplore
                cost_to[''.join(map(str, neighbor))] = pvi
                est_total_cost[''.join(map(str, neighbor))
                               ] = pvi + h(neighbor, goal)

                found = False
                for vertex in Q:
                    if (vertex[1] == neighbor):
                        vertex[0] = est_total_cost[''.join(map(str, neighbor))]
                        found = True

                if (not found):
                    heappush(
                        Q, [est_total_cost[''.join(map(str, neighbor))], neighbor])

    return []

###########################################
#               PART B
###########################################


def unoccupiedNeighbors(v):
    unoccupiedNeighbors = []
    # top left
    if (not (v[0] == 0) and not (v[1] == 0)):
        if (M[v[0]-1][v[1]-1]) == 1:
            unoccupiedNeighbors.append([v[0]-1, v[1]-1])
    # left
    if (not (v[0] == 0)):
        if (M[v[0]-1][v[1]]) == 1:
            unoccupiedNeighbors.append([v[0]-1, v[1]])
    # bottom left
    if (not (v[0] == 0) and not (v[1] == len(M[0])-1)):
        if (M[v[0]-1][v[1]+1]) == 1:
            unoccupiedNeighbors.append([v[0]-1, v[1]+1])
    # below
    if (not (v[1] == 0)):
        if (M[v[0]][v[1]-1]) == 1:
            unoccupiedNeighbors.append([v[0], v[1]-1])
    # top right
    if (not (v[0] == len(M)-1) and not (v[1] == 0)):
        if (M[v[0]+1][v[1]-1]) == 1:
            unoccupiedNeighbors.append([v[0]+1, v[1]-1])
    # right
    if (not (v[0] == len(M)-1)):
        if (M[v[0]+1][v[1]]) == 1:
            unoccupiedNeighbors.append([v[0]+1, v[1]])
    # bottom right
    if (not (v[0] == len(M)-1) and not (v[1] == len(M[0])-1)):
        if (M[v[0]+1][v[1]+1]) == 1:
            unoccupiedNeighbors.append([v[0]+1, v[1]+1])
    # above
    if (not (v[1] == len(M[0])-1)):
        if (M[v[0]][v[1]+1]) == 1:
            unoccupiedNeighbors.append([v[0], v[1]+1])

    return unoccupiedNeighbors


def distance(v1, v2):
    if (v1[0] == v2[0] or v1[1] == v2[1]):
        return 1
    else:
        return 2**0.5


def euclideanDistance(v1, v2):
    return (((v1[0] - v2[0]) * (v1[0] - v2[0])) + ((v1[1] - v2[1]) * (v1[1] - v2[1])))**0.5


start = [635, 140]
end = [350, 400]

vertexSet = []
# for x in range(len(M)):
#     for y in range(len(M[0])):
#         vertexSet.append([x, y])

# path = A_Star(vertexSet, start, end, unoccupiedNeighbors, distance, heuristic)

# with Image.open('occupancy_map.png') as im:
#     draw = ImageDraw.Draw(im)
#     # draw.ellipse((635-50, 140-50, 635+50, 140+50),
#     #              fill='red', outline='blue')
#     # draw.line([(635, 140), (350, 400)], width=10, fill='red')
#     draw.line(path, width=1, fill='red')

#     im.show()

###########################################
#               PART C
###########################################


def getRandomSample(Map):
    xSize = len(Map)
    ySize = len(Map[0])
    samplePointValue = 0
    while samplePointValue == 0:
        randomX = random.sample(range(xSize), 1)
        randomY = random.sample(range(ySize), 1)
        samplePointValue = Map[randomX[0]][randomY[0]]

    return (randomX[0], randomY[0])


def reachableByStraightLine(Map, vertex1, vertex2):
    pointsToCheck = list(
        bresenham(vertex1[0], vertex1[1], vertex2[0], vertex2[1]))

    for point in pointsToCheck:
        if (Map[point[0]][point[1]] == 0):
            return False

    return True


def addVertex(graph, newVertex, searchRadius, map):
    nextNodeNum = graph.number_of_nodes() + 1
    graph.add_node(nextNodeNum, {'pos': (newVertex[0], newVertex[1])})

    for vertex in graph.nodes():
        if (not (vertex.index == nextNodeNum)):
            if (reachableByStraightLine(map, newVertex, vertex)):
                distanceBetweenNodes = euclideanDistance(newVertex, vertex)
                if (distanceBetweenNodes < searchRadius):
                    graph.add_edge(nextNodeNum, vertex.index,
                                   weight=distanceBetweenNodes)


def createPRM(numSamples, searchRadius, map):
    graph = nx.graph()

    for i in range(numSamples):
        newVertex = getRandomSample(map)
        addVertex(graph, newVertex, searchRadius, map)

    return graph

    ###########################################
    #                   TESTS
    ###########################################
    # # binary map
    # M = [[1, 1, 1, 1, 1, 0],
    #      [1, 1, 1, 0, 0, 0],
    #      [1, 0, 1, 1, 0, 1],
    #      [1, 0, 0, 1, 0, 1],
    #      [1, 0, 0, 1, 0, 1],
    #      [1, 0, 1, 1, 1, 1],
    #      [1, 0, 1, 1, 1, 1],
    #      [1, 0, 1, 1, 1, 1],
    #      [1, 0, 0, 0, 0, 1],
    #      [1, 0, 0, 0, 0, 1],
    #      [1, 0, 0, 0, 0, 1],
    #      [1, 0, 0, 0, 0, 1],
    #      [0, 1, 1, 1, 1, 1], ]


sample = getRandomSample(M)
print(sample)
# print(list(bresenham(0, 0, 2, 8)))


# M = [[1, 1, 0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 1, 1, 1, 1, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 1, 1, 1], ]

# M = [[1, 1, 1],
#      [1, 1, 1],
#      [1, 1, 1]]


start = [0, 0]
end = [12, 5]

path = A_Star(vertexSet, start, end, unoccupiedNeighbors, distance, heuristic)
print(path)

# print(h([635, 140], [350, 400]))

# print(N([3, 3]))

# def h(s,g):
#     sqr(s.)


# print(''.join(map(str, [0, 1])))


# pred1 = {
#     "F": "E",
#     "E": "D",
#     "D": "C",
#     "C": "B",
#     "B": "A"
# }

# pred2 = {
#     "F": "K",
#     "G": "E",
#     "E": "F",
#     "H": "G"
# }


# path = RecoverPath("K", "H", pred2)
# print(path)
print("finished")