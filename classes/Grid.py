import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse

from .Graph import Graph, dijkstra, to_array
from .Utils import Utils

class Grid:
    def __init__(self, grid_rows, grid_cols, start, destination, allObs):
        self.grid_size = (grid_rows, grid_cols)
        self.start = np.array(start)
        self.destination = np.array(destination)

        self.allObs = allObs

        self.graph = Graph()
        self.utils = Utils()

        self.dijkstra_dist = []
        self.dijkstra_prev = []

    def runGrid(self):
        '''
        Run the grid
        '''
        self.genCoords()
        self.checkIfCollisonFree()
        # self.formGrid()
        self.createGridGraph()
        self.calcAllPathCost()
        return self.collisionFreePoints, self.graph.edges
    

    def genCoords(self):
        x = np.linspace(0, self.grid_size[0]-1, self.grid_size[0])
        y = np.linspace(0, self.grid_size[1]-1, self.grid_size[1])
        self.node_coords = np.array([[i, j] for i in x for j in y])
        self.start = self.start.reshape(1, 2)
        self.destination = self.destination.reshape(1, 2)
        self.node_coords = np.concatenate(
            (self.destination, self.start, self.node_coords), axis=0)

    def createGridGraph(self):
        rows, cols = self.grid_size
        for i in range(rows):
            for j in range(cols):
                current_node = (i, j)
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]  # Grid-like neighbors
                for neighbor in neighbors:
                    if 1 <= neighbor[0] < rows-1 and 1 <= neighbor[1] < cols-1:  # Check if neighbor is within grid boundaries
                        if not self.checkPointCollision(current_node) and not self.checkPointCollision(neighbor):
                            if not self.checkLineCollision(current_node, neighbor):
                                a = str(self.findNodeIndex(current_node))
                                b = str(self.findNodeIndex(neighbor))
                                self.graph.add_node(a)
                                self.graph.add_edge(a, b, 1)

    def formGrid(self):
        '''
        return a grid of size grid_size as a Graph() (self.graph)
        '''
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                i = str(self.findNodeIndex(i))
                j = str(self.findNodeIndex(j))
                self.graph.add_node(i)
                node  = (i, j)
                if i > 0:
                    self.graph.add_edge(node, (i - 1, j), 1)  # Up
                if i < self.grid_size[0] - 1:
                    self.graph.add_edge(node, (i + 1, j), 1)  # Down
                if j > 0:
                    self.graph.add_edge(node, (i, j - 1), 1)  # Left
                if j < self.grid_size[1] - 1:
                    self.graph.add_edge(node, (i, j + 1), 1)  # Right
        return self.graph
    
    def calcAllPathCost(self):
        for coord in self.collisionFreePoints:
            startNode = str(self.findNodeIndex(coord))
            dist, prev = dijkstra(self.graph, startNode)
            self.dijkstra_dist.append(dist)
            self.dijkstra_prev.append(prev)

    def checkLineCollision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.allObs:
            if(self.utils.isWall(obs)):
                uniqueCords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(
                    uniqueCords)
                if(line.intersection(wall)):
                    collision = True
            else:
                obstacleShape = shapely.geometry.Polygon(
                    obs.allCords)
                collision = line.intersects(obstacleShape)
            if(collision):
                return True
        return False

    def checkIfCollisonFree(self):
        collision = False
        self.collisionFreePoints = np.array([])
        for point in self.node_coords:
            collision = self.checkPointCollision(point)
            if(not collision):
                if(self.collisionFreePoints.size == 0):
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point])

    def findPointsFromNode(self, node):
        '''
        Find the points from a node
        '''
        return self.collisionFreePoints[int(node)]
    
    def checkCollision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if(obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.bottomLeft[1] <= p_y <= obs.topLeft[1]):
            return True
        else:
            return False

    def checkPointCollision(self, point):
        for obs in self.allObs:
            collision = self.checkCollision(obs, point)
            if(collision):
                return True
        return False
    
    def calcDistance(self, current, destination):
        startNode = str(self.findNodeIndex(current))
        endNode = str(self.findNodeIndex(destination))
        if startNode == endNode:
            return 0
        pathToEnd = to_array(self.dijkstra_prev[int(startNode)], endNode)
        if len(pathToEnd) <= 1: # not expand this node
            return 1000

        distance = self.dijkstra_dist[int(startNode)][endNode]
        distance = 0 if distance is None else distance
        return distance
    
    def findNodeIndex(self, p):
        # return np.where((self.collisionFreePoints == p).all(axis=1))[0][0]
        # print(np.linalg.norm(self.collisionFreePoints - p, axis=1), p)
        # print(p)
        return np.where(np.linalg.norm(self.collisionFreePoints - p, axis=1) < 1e-5)[0][0]
