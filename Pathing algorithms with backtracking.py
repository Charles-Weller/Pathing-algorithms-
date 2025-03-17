import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time
import pandas as pd
from collections import deque
import math

startPoint = (51.754816, -1.222991)
middlePoint = (51.735, -1.205)
endPoint = (51.72627, -1.19996)


roadGraph = ox.graph_from_point(middlePoint, dist=3000, network_type="drive")
startNode = ox.distance.nearest_nodes(roadGraph, X=startPoint[1], Y=startPoint[0])
endNode = ox.distance.nearest_nodes(roadGraph, X=endPoint[1], Y=endPoint[0])

fig, ax = ox.plot_graph(roadGraph, node_size=10, edge_linewidth=0.5, show=False, close=False)
plt.scatter([roadGraph.nodes[startNode]["x"]], [roadGraph.nodes[startNode]["y"]], c="green", s=100, zorder=5)
plt.scatter([roadGraph.nodes[endNode]["x"]], [roadGraph.nodes[endNode]["y"]], c="blue", s=100, zorder=5)
plt.title("Road network graph represonataion")
fig.set_size_inches(15, 15)
plt.savefig("roadNetwork.png", dpi=300, bbox_inches="tight")
plt.show()

numNodes = len(roadGraph.nodes)
numEdges = len(roadGraph.edges)
print("There are ", numNodes, "nodes in this graph.")
print("There are ", numEdges, "edges in this graph.\n")

restrictedRoads = ["Sandford Road", "Garsington Road"]
roadSafetyStatus = {"Sandford Road": "unsafe","Garsington Road": "unsafe"}

def isRoadOpen(roadName):
    if isinstance(roadName, list):
        return all(roadSafetyStatus.get(name, "safe") == "safe" for name in roadName)
    return roadSafetyStatus.get(roadName, "safe") == "safe"

def isRoadSafe(roadName):
    if isinstance(roadName, list):
        return all(roadSafetyStatus.get(name, "safe") == "safe" for name in roadName)
    return roadSafetyStatus.get(roadName, "safe") == "safe"

def backtrackingPathfinding(graph, startNode, endNode, visited=None, path=None):

    if visited is None:
        visited = set()
    if path is None:
        path = [startNode]

    visited.add(startNode)
    if startNode == endNode:
        return path

    for neighbor in graph.neighbors(startNode):
        roadName = graph.edges[startNode, neighbor, 0].get("name", "")

        if isRoadOpen(roadName) and isRoadSafe(roadName):
            if neighbor not in visited:
                newPath = backtrackingPathfinding(graph, neighbor, endNode, visited, path + [neighbor])
                if newPath:
                    return newPath

    return None

def bfs(graph, start, goal):
    startTime = time.time()
    queue = deque([(start, [start])])
    visited = set()
    visitedNodes = 0

    while queue:
        currentNode, path = queue.popleft()

        if currentNode == goal:
            pathLength = sum(graph[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1))
            executionTime = time.time() - startTime
            return path, pathLength, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None, None, visitedNodes, executionTime

def bfsBT(graph, start, goal):
    startTime = time.time()
    queue = deque([(start, [start])])
    visited = set()
    visitedNodes = 0

    while queue:
        currentNode, path = queue.popleft()

        if currentNode == goal:
            pathLength = sum(graph[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1))
            executionTime = time.time() - startTime
            return None, pathLength, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                roadName = graph.edges[currentNode, neighbor, 0].get("name", "")

                if isRoadOpen(roadName) and isRoadSafe(roadName):
                    queue.append((neighbor, path + [neighbor]))

    return backtrackingPathfinding(graph, start, goal)

def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    visitedNodes = 0
    startTime = time.time()

    while stack:
        currentNode, path = stack.pop()

        if currentNode == goal:
            pathLength = sum(graph[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1))
            executionTime = time.time() - startTime
            return path, pathLength, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None, None, visitedNodes, executionTime

def dfsBT(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    visitedNodes = 0
    startTime = time.time()

    while stack:
        currentNode, path = stack.pop()

        if currentNode == goal:
            pathLength = sum(graph[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1))
            executionTime = time.time() - startTime
            return None, pathLength, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                roadName = graph.edges[currentNode, neighbor, 0].get("name", "")

                if isRoadOpen(roadName) and isRoadSafe(roadName):
                    stack.append((neighbor, path + [neighbor]))

    return backtrackingPathfinding(graph, start, goal)

def dijkstra(graph, start, goal):
  startTime = time.time()
  openSet = [(0, start, [start])]
  heapq.heapify(openSet)
  visited = set()
  minCost = {start: 0}
  visitedNodes = 0

  while openSet:
        cost, currentNode, path = heapq.heappop(openSet)

        if currentNode == goal:
            executionTime = time.time() - startTime
            return path, cost, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                edgeWeight = graph[currentNode][neighbor][0]["length"]
                newCost = cost + edgeWeight

                if neighbor not in minCost or newCost < minCost[neighbor]:
                    minCost[neighbor] = newCost
                    heapq.heappush(openSet, (newCost, neighbor, path + [neighbor]))

  return None, None, visitedNodes, executionTime

def dijkstraBT(graph, start, goal):
    startTime = time.time()
    openSet = [(0, start, [start])]
    heapq.heapify(openSet)
    visited = set()
    minCost = {start: 0}
    visitedNodes = 0

    while openSet:
        cost, currentNode, path = heapq.heappop(openSet)

        if currentNode == goal:
            executionTime = time.time() - startTime
            return None, None, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                roadName = graph.edges[currentNode, neighbor, 0].get("name", "")

                if isRoadOpen(roadName) and isRoadSafe(roadName):
                    edgeWeight = graph[currentNode][neighbor][0]["length"]
                    newCost = cost + edgeWeight

                    if neighbor not in minCost or newCost < minCost[neighbor]:
                        minCost[neighbor] = newCost
                        heapq.heappush(openSet, (newCost, neighbor, path + [neighbor]))

    return backtrackingPathfinding(graph, start, goal)


def euclideanDistance(node, goal):
    nodeCoords = (roadGraph.nodes[node]["y"], roadGraph.nodes[node]["x"])
    goalCoords = (roadGraph.nodes[goal]["y"], roadGraph.nodes[goal]["x"])

    x1, y1 = nodeCoords
    x2, y2 = goalCoords
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def aStar(graph, start, goal):
    startTime = time.time()
    openSet = [(0 + euclideanDistance(start, goal), 0, start, [start])]
    heapq.heapify(openSet)
    visited = set()
    minCost = {start: 0}
    visitedNodes = 0

    while openSet:
        fCost, cost, currentNode, path = heapq.heappop(openSet)

        if currentNode == goal:
            executionTime = time.time() - startTime
            return path, cost, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                edgeWeight = graph[currentNode][neighbor][0]["length"]
                newCost = cost + edgeWeight

                fCost = newCost + euclideanDistance(neighbor, goal)

                if neighbor not in minCost or newCost < minCost[neighbor]:
                    minCost[neighbor] = newCost
                    heapq.heappush(openSet, (fCost, newCost, neighbor, path + [neighbor]))


    return None, None, visitedNodes, executionTime

def aStarBT(graph, start, goal):
    startTime = time.time()
    openSet = [(0 + euclideanDistance(start, goal), 0, start, [start])]
    heapq.heapify(openSet)
    visited = set()
    minCost = {start: 0}
    visitedNodes = 0

    while openSet:
        fCost, cost, currentNode, path = heapq.heappop(openSet)

        if currentNode == goal:
            executionTime = time.time() - startTime
            return None, cost, visitedNodes, executionTime

        if currentNode not in visited:
            visited.add(currentNode)
            visitedNodes += 1

            for neighbor in graph.neighbors(currentNode):
                roadName = graph.edges[currentNode, neighbor, 0].get("name", "")

                if isRoadOpen(roadName) and isRoadSafe(roadName):
                    edgeWeight = graph[currentNode][neighbor][0]["length"]
                    newCost = cost + edgeWeight
                    fCost = newCost + euclideanDistance(neighbor, goal)

                    if neighbor not in minCost or newCost < minCost[neighbor]:
                        minCost[neighbor] = newCost
                        heapq.heappush(openSet, (fCost, newCost, neighbor, path + [neighbor]))

    return backtrackingPathfinding(graph, start, goal)

def bidirectionalDijkstra(graph, start, goal):
    startTime = time.time()

    fQueue = [(0, start, [start])]
    heapq.heapify(fQueue)
    fVisited = set()
    fDist = {start: 0}
    fPrev = {start: None}
    fVisitedNodes = 0

    bQueue = [(0, goal, [goal])]
    heapq.heapify(bQueue)
    bVisited = set()
    bDist = {goal: 0}
    bPrev = {goal: None}
    bVisitedNodes = 0

    shortestPathLength = float("inf")
    meetingNode = None
    visitedNodes = 0

    while fQueue and bQueue:
        if fQueue:
            fCost, fNode, fPath = heapq.heappop(fQueue)

            if fNode in bVisited:
                totalDist = fDist[fNode] + bDist[fNode]
                if totalDist < shortestPathLength:
                    shortestPathLength = totalDist
                    meetingNode = fNode

                    bPath = []
                    bNode = fNode
                    while bNode is not None:
                        bPath.append(bNode)
                        bNode = bPrev.get(bNode)
                    bPath.reverse()
                    path = fPath + bPath[1:]
                    break

            if fNode not in fVisited:
                fVisited.add(fNode)
                fVisitedNodes += 1
                visitedNodes += 1

                for neighbor in graph.neighbors(fNode):
                    edgeWeight = graph[fNode][neighbor][0]["length"]
                    newCost = fCost + edgeWeight

                    if neighbor not in fDist or newCost < fDist[neighbor]:
                        fDist[neighbor] = newCost
                        fPrev[neighbor] = fNode
                        heapq.heappush(fQueue, (newCost, neighbor, fPath + [neighbor]))

        if bQueue:
            bCost, bNode, bPath = heapq.heappop(bQueue)

            if bNode in fVisited:
                totalDist = fDist[bNode] + bDist[bNode]
                if totalDist < shortestPathLength:
                    shortestPathLength = totalDist
                    meeting_node = bNode

                    fPath = []
                    fNode = bNode
                    while fNode is not None:
                        fPath.append(fNode)
                        fNode = fPrev.get(fNode)
                    fPath.reverse()
                    path = fPath + bPath[1:]
                    break

            if bNode not in bVisited:
                bVisited.add(bNode)
                bVisitedNodes += 1
                visitedNodes += 1

                for neighbor in graph.neighbors(bNode):
                    edgeWeight = graph[bNode][neighbor][0]["length"]
                    newCost = bCost + edgeWeight

                    if neighbor not in bDist or newCost < bDist[neighbor]:
                        bDist[neighbor] = newCost
                        bPrev[neighbor] = bNode
                        heapq.heappush(bQueue, (newCost, neighbor, bPath + [neighbor]))

    executionTime = time.time() - startTime
    return None, shortestPathLength, visitedNodes, executionTime

def bidirectionalDijkstraBT(graph, start, goal):
    startTime = time.time()

    fQueue = [(0, start, [start])]
    heapq.heapify(fQueue)
    fVisited = set()
    fDist = {start: 0}
    fPrev = {start: None}
    fVisitedNodes = 0

    bQueue = [(0, goal, [goal])]
    heapq.heapify(bQueue)
    bVisited = set()
    bDist = {goal: 0}
    bPrev = {goal: None}
    bVisitedNodes = 0

    shortestPathLength = float("inf")
    meetingNode = None
    visitedNodes = 0

    while fQueue and bQueue:
        if fQueue:
            fCost, fNode, fPath = heapq.heappop(fQueue)

            if fNode in bVisited:
                totalDist = fDist[fNode] + bDist[fNode]
                if totalDist < shortestPathLength:
                    shortestPathLength = totalDist
                    meetingNode = fNode

                    bPath = []
                    bNode = fNode
                    while bNode is not None:
                        bPath.append(bNode)
                        bNode = bPrev.get(bNode)
                    bPath.reverse()
                    path = fPath + bPath[1:]
                    break

            if fNode not in fVisited:
                fVisited.add(fNode)
                fVisitedNodes += 1
                visitedNodes += 1

                for neighbor in graph.neighbors(fNode):
                    road_name = graph.edges[fNode, neighbor, 0].get("name", "")
                    if isRoadOpen(road_name) and isRoadSafe(road_name):
                        edgeWeight = graph[fNode][neighbor][0]["length"]
                        newCost = fCost + edgeWeight

                        if neighbor not in fDist or newCost < fDist[neighbor]:
                            fDist[neighbor] = newCost
                            fPrev[neighbor] = fNode
                            heapq.heappush(fQueue, (newCost, neighbor, fPath + [neighbor]))

        if bQueue:
            bCost, bNode, bPath = heapq.heappop(bQueue)

            if bNode in fVisited:
                totalDist = fDist[bNode] + bDist[bNode]
                if totalDist < shortestPathLength:
                    shortestPathLength = totalDist
                    meetingNode = bNode

                    fPath = []
                    fNode = bNode
                    while fNode is not None:
                        fPath.append(fNode)
                        fNode = fPrev.get(fNode)
                    fPath.reverse()
                    path = fPath + bPath[1:]
                    break

            if bNode not in bVisited:
                bVisited.add(bNode)
                bVisitedNodes += 1
                visitedNodes += 1

                for neighbor in graph.neighbors(bNode):
                    road_name = graph.edges[bNode, neighbor, 0].get("name", "")
                    if isRoadOpen(road_name) and isRoadSafe(road_name):
                        edgeWeight = graph[bNode][neighbor][0]["length"]
                        newCost = bCost + edgeWeight

                        if neighbor not in bDist or newCost < bDist[neighbor]:
                            bDist[neighbor] = newCost
                            bPrev[neighbor] = bNode
                            heapq.heappush(bQueue, (newCost, neighbor, bPath + [neighbor]))

    executionTime = time.time() - startTime
    return None, shortestPathLength, visitedNodes, executionTime

def compareAlgorithms(graph, startNode, endNode):
    results = {}
    algorithms = {"BD": bidirectionalDijkstra, "BD BT": bidirectionalDijkstraBT, "A*": aStar,"A* BT": aStarBT,"DFS": dfs, "DFS BT": dfsBT, "BFS": bfs, "BFS BT": bfsBT,"D": dijkstra, "D BT": dijkstraBT}

    for algo_name, algo_func in algorithms.items():
        path, cost, visited_nodes, exec_time = algo_func(graph, startNode, endNode)
        results[algo_name] = {"Visited Nodes": visited_nodes,"Path Length (meters)": cost,"Execution Time (seconds)": exec_time}
    return results

print("All the BT will have varying results due to the limitation on roads that they can take")
print("")

results = compareAlgorithms(roadGraph, startNode, endNode)
dfResults = pd.DataFrame(results).T
print(dfResults)

print("")
print("BD stands for biodiregical dijkstra")
print("D syands for  dijkstra")
print("BT represents when backtracking is being used")

algorithms = dfResults.index
nodesVisited = dfResults["Visited Nodes"]
plt.bar(algorithms, nodesVisited, color=["blue"])
plt.title("Nodes Visited by pathfinding algorithms")
plt.xlabel("Pathing methord")
plt.ylabel("Number of Nodes Visited")
plt.show()


executionTimes = dfResults["Execution Time (seconds)"]
plt.bar(algorithms, executionTimes, color=["blue"])
plt.title("Time for the pathfinding algorithms to find path")
plt.xlabel("Pathing methord")
plt.ylabel("Execution time")
plt.show()

pathLengths = dfResults["Path Length (meters)"]
plt.bar(algorithms, pathLengths, color=["blue"])
plt.title("Lengths of paths found by pathfinding algorithms")
plt.xlabel("Pathing methord")
plt.ylabel("Path Length")
plt.show()
