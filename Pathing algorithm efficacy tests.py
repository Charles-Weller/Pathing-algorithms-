import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time
import pandas as pd
from collections import deque

# Coordinates of Headington Campus, Oxford Brookes University
start_point = (51.754816, -1.222991)  # Oxford Brookes University
# Coordinates of Gatwick Airport
end_point = (51.742027, -1.228216)    

# Get the road network within a 10000-meter radius of Oxford Brookes University
G = ox.graph_from_point(start_point, dist=10000, network_type='drive')

# Calculate the number of nodes and edges in the graph
num_nodes = len(G.nodes)
num_edges = len(G.edges)

print(f"Number of nodes in the graph: {num_nodes}")
print(f"Number of edges in the graph: {num_edges}")

# Find the nearest nodes to the start and end points in the graph
start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])

# Plot the graph with the start and end points highlighted
fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=0.5, show=False, close=False)
plt.scatter([G.nodes[start_node]['x']], [G.nodes[start_node]['y']], c='green', s=100, zorder=5, label="Start: Oxford Brookes")
plt.scatter([G.nodes[end_node]['x']], [G.nodes[end_node]['y']], c='blue', s=100, zorder=5, label="End: Gatwick Airport")
plt.title('Road Network between Oxford Brookes and Gatwick Airport')

# Add a legend for start and end points
plt.legend(loc='upper right')

# Save the plot as a high-resolution image
fig.set_size_inches(14, 12)
plt.savefig("road_network_oxford_brookes_gatwick.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

#BFS
def bfs(graph, start, goal):
    start_time = time.time()  # Start timing
    queue = deque([(start, [start])])
    visited = set()
    visited_nodes = 0  # To track visited nodes

    while queue:
        current_node, path = queue.popleft()

        if current_node == goal:
            # Calculate the total path length in meters by summing edge weights
            path_length = sum(graph[path[i]][path[i + 1]][0]['length'] for i in range(len(path) - 1))
            execution_time = time.time() - start_time  # Stop timing
            return path, path_length, visited_nodes, execution_time

        if current_node not in visited:
            visited.add(current_node)
            visited_nodes += 1  # Count the visited nodes

            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None, None, visited_nodes, time.time() - start_time

    #DFS
def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    visited_nodes = 0  # To track visited nodes
    start_time = time.time()  # Start timing

    while stack:
        current_node, path = stack.pop()

        if current_node == goal:
            # Calculate the total path length in meters by summing edge weights
            path_length = sum(graph[path[i]][path[i + 1]][0]['length'] for i in range(len(path) - 1))
            execution_time = time.time() - start_time  # Stop timing
            return path, path_length, visited_nodes, execution_time

        if current_node not in visited:
            visited.add(current_node)
            visited_nodes += 1  # Count the visited nodes

            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None, None, visited_nodes, time.time() - start_time

# Dijkstra algorithm
def dijkstra(graph, start, goal):
  start_time = time.time()  # Start timing

    ####
  print(graph)
  print(start)
  print(goal)
    ####

  open_set = [(0, start, [start])]
  heapq.heapify(open_set)
  visited = set()

  min_cost = {start: 0}
  visited_nodes = 0  # To count the number of visited nodes

  while open_set:
        cost, current_node, path = heapq.heappop(open_set)

        if current_node == goal:
            execution_time = time.time() - start_time  # Stop timing
            return path, cost, visited_nodes, execution_time

        if current_node not in visited:
            visited.add(current_node)
            visited_nodes += 1  # Increment visited nodes count

            for neighbor in graph.neighbors(current_node):
                edge_weight = graph[current_node][neighbor][0]['length']
                new_cost = cost + edge_weight

                if neighbor not in min_cost or new_cost < min_cost[neighbor]:
                    min_cost[neighbor] = new_cost
                    heapq.heappush(open_set, (new_cost, neighbor, path + [neighbor]))

  return None, None, visited_nodes, time.time() - start_time
####
# Bidirectional Dijkstra Algorithm (Updated)
def bidirectional_dijkstra(graph, start, goal):
    start_time = time.time()  # Start timing
    
    # Forward direction: start -> goal
    forward_queue = [(0, start, [start])]
    heapq.heapify(forward_queue)
    forward_visited = set()
    forward_dist = {start: 0}
    forward_prev = {start: None}  # To track the path in the forward search
    forward_visited_nodes = 0  # Count of nodes visited in forward search
    
    # Backward direction: goal -> start
    backward_queue = [(0, goal, [goal])]
    heapq.heapify(backward_queue)
    backward_visited = set()
    backward_dist = {goal: 0}
    backward_prev = {goal: None}  # To track the path in the backward search
    backward_visited_nodes = 0  # Count of nodes visited in backward search
    
    # Initialize best distance and meeting node
    shortest_path_length = float('inf')
    meeting_node = None
    visited_nodes = 0  # Total number of visited nodes
    
    while forward_queue and backward_queue:
        # Expand the forward search
        if forward_queue:
            f_cost, f_node, f_path = heapq.heappop(forward_queue)

            if f_node in backward_visited:
                total_dist = forward_dist[f_node] + backward_dist[f_node]
                if total_dist < shortest_path_length:
                    shortest_path_length = total_dist
                    meeting_node = f_node
                    
                    # Reconstruct the full path by combining forward and backward paths
                    # Forward path is from start to meeting node
                    # Backward path is from goal to meeting node (reverse it)
                    forward_path = f_path
                    backward_path = []
                    b_node = f_node
                    while b_node is not None:
                        backward_path.append(b_node)
                        b_node = backward_prev.get(b_node)
                    backward_path.reverse()
                    path = forward_path + backward_path[1:]  # Avoid double-counting meeting node
                    break

            if f_node not in forward_visited:
                forward_visited.add(f_node)
                forward_visited_nodes += 1
                visited_nodes += 1  # Increment total visited nodes

                for neighbor in graph.neighbors(f_node):
                    edge_weight = graph[f_node][neighbor][0]['length']
                    new_cost = f_cost + edge_weight

                    if neighbor not in forward_dist or new_cost < forward_dist[neighbor]:
                        forward_dist[neighbor] = new_cost
                        forward_prev[neighbor] = f_node  # Track the path in the forward search
                        heapq.heappush(forward_queue, (new_cost, neighbor, f_path + [neighbor]))
        
        # Expand the backward search
        if backward_queue:
            b_cost, b_node, b_path = heapq.heappop(backward_queue)

            if b_node in forward_visited:
                total_dist = forward_dist[b_node] + backward_dist[b_node]
                if total_dist < shortest_path_length:
                    shortest_path_length = total_dist
                    meeting_node = b_node
                    
                    # Reconstruct the full path by combining forward and backward paths
                    # Forward path is from start to meeting node
                    # Backward path is from goal to meeting node (reverse it)
                    forward_path = []
                    f_node = b_node
                    while f_node is not None:
                        forward_path.append(f_node)
                        f_node = forward_prev.get(f_node)
                    forward_path.reverse()
                    backward_path = b_path
                    path = forward_path + backward_path[1:]  # Avoid double-counting meeting node
                    break

            if b_node not in backward_visited:
                backward_visited.add(b_node)
                backward_visited_nodes += 1
                visited_nodes += 1  # Increment total visited nodes

                for neighbor in graph.neighbors(b_node):
                    edge_weight = graph[b_node][neighbor][0]['length']
                    new_cost = b_cost + edge_weight

                    if neighbor not in backward_dist or new_cost < backward_dist[neighbor]:
                        backward_dist[neighbor] = new_cost
                        backward_prev[neighbor] = b_node  # Track the path in the backward search
                        heapq.heappush(backward_queue, (new_cost, neighbor, b_path + [neighbor]))

    execution_time = time.time() - start_time  # Stop timing

    #if meeting_node is None:
    return None, None, visited_nodes, execution_time  # No path found

    #return path, shortest_path_length, visited_nodes, execution_time  # Return path, cost, visited nodes, time



####
#Compare funtions 
def compare_algorithms(graph, start_node, end_node):
    results = {}
    algorithms = {
        'Dijkstra': dijkstra,
        'Bidirectional Dijkstra': bidirectional_dijkstra,
        #'A*': a_star,
        'BFS': bfs,
        'DFS': dfs
    }

    for algo_name, algo_func in algorithms.items():
        # Call each algorithm and get the visited nodes, path length, and execution time
        path, cost, visited_nodes, exec_time = algo_func(graph, start_node, end_node)

        # Store results in the dictionary
        results[algo_name] = {
            'Visited Nodes': visited_nodes,
            'Path Length (meters)': cost,
            'Execution Time (seconds)': exec_time
        }

    return results

# Test the algorithms and display the results in a DataFrame
results = compare_algorithms(G, start_node, end_node)
df_results = pd.DataFrame(results).T
print(df_results)

#barchart
algorithms = df_results.index
nodes_visited = df_results['Visited Nodes']

plt.bar(algorithms, nodes_visited, color=['blue', 'green', 'red'])
plt.title('Nodes Visited by Pathfinding Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Number of Nodes Visited')
plt.show()

#line graph
execution_times = df_results['Execution Time (seconds)']

plt.plot(algorithms, execution_times, marker='o', linestyle='-', color='purple')
plt.title('Execution Time of Pathfinding Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (seconds)')
plt.show()

#barchart distance 
path_lengths = df_results['Path Length (meters)']

plt.bar(algorithms, path_lengths, color=['orange', 'blue', 'green'])
plt.title('Path Lengths Found by Pathfinding Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Path Length (meters)')
plt.show()
