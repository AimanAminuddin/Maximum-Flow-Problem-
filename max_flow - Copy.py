import random as rd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

import copy


MAX_LAYERS = 3          # defines max number of layers in the graph
MAX_SIZE_LAYERS = 4     # defines max size for each layer in the graph
MAX_CAPACITY = 40       # defines the max capacity of an edge
EDGE_DENSITY_FORWARD = 0.5   # defines the forward edge density at each layer
EDGE_DENSITY_BACKWARDS = 0.2 # defines the backwards edge density at each layer
EDGE_DENSITY_SIDE = 0.2      # defines the side edge density at each layer

LAYERS = [rd.randint(2,MAX_SIZE_LAYERS) if 0<i<=MAX_LAYERS else 1 for i in range(MAX_LAYERS+2)]
NODES = sum(LAYERS)                          
INFINITY = math.inf     

# function that computes with BFS the shortest path from start_node to end_node
# and returns the corresponding path as a list of nodes 
def BFS_search(graph):
    # start node is always 0 # 
    victory = change_to_dict_of_dict(graph)
    start_node = 0
    length = len(graph)
    end_node = length - 1 
    explored = [False]*length
    explored[start_node] = True
    queue = [start_node]
    my_dict = {start_node:None} 
    while len(queue) > 0:
        current_node = queue.pop(0)
        adjacent = victory[current_node]

        if current_node == end_node:
            temp = end_node
            path = [end_node]
            while start_node != path[0]:
                path = [my_dict[temp]] + path
                temp = my_dict[temp]
            
            converted_path = [] 
            for elem in range(len(path)-1):
                curr = path[elem]
                next_node = path[elem+1]
                temp = [curr,next_node,victory[curr][next_node]]
                converted_path.append(temp)
            
            return converted_path

        else:
            for num in adjacent:
                if not explored[num]:
                    queue.append(num)
                    explored[num] = True
                    my_dict[num] = current_node
                else:
                    continue
    ## No path between start_node and end_node # 
    return [] 


# function that returns the residual graph from the input graph
def change_to_dict_of_dict(graph):
    length = len(graph)
    my_dict = {}
    for num in range(length):
        adjacent = graph[num]
        temp = {elem[0]:elem[1] for elem in adjacent}
        my_dict[num] = temp

    return my_dict 

def build_residual_graph(graph,flow):
    victory = change_to_dict_of_dict(graph)
    length = len(graph)
    # keep track of all the forward edge: capacity - flow
    lst = [] 
    for num in victory:
        adjacent = victory[num]
        for elem in adjacent:
            capacity = victory[num][elem]
            flowing = flow[num][elem] 
            weight = capacity - flowing
            victory[num][elem] = weight
            lst.append([num,elem])
    

    # keep track of all the backward edge: - f #
    for edge in lst:
        node1 = edge[0]
        node2 = edge[1]
        flowing = flow[node1][node2]
        if flowing == 0: 
            continue 
        else:
            victory[node2][node1] = -flowing

    
    # Remove all the 0 residual capacity # 
    my_dict = {}
    for i in victory:
        adjacent = victory[i]
        my_dict[num] = None 
        temp = {} 
        for j in adjacent: 
            residual_capacity = adjacent[j]
            if residual_capacity != 0:
                temp[j] = residual_capacity 
            else:
                continue 
        my_dict[i] = temp 
    # sort dictionary in ascending order # 
    final = {}
    for num in range(length):
        final[num] = my_dict[num]
    
    
    updated_graph = [] 
    for x in final: 
        adjacent = final[x]
        copy = [] 
        for y in adjacent: 
            edge = [y,adjacent[y]]
            copy.append(edge)
        updated_graph.append(copy)
        
    return updated_graph
                
# function that implements the Edmonds Karp algorithm to find max-flow of a network graph


def edmonds_karp(g):
    length = len(g)
    flow = [[0]*length for num in range(length)]
    ## build residual graph ##
    residual_graph = build_residual_graph(g,flow)
    max_flow = 0 

    while True:
        ## find an Augmenting path using BFS ##
        print(residual_graph) 
        path = BFS_search(residual_graph)
        print(path) 
        if path == []:
            break
        else:
            ## find the residual capacity of the path ##
            lst = []
            edges = [] 
            for node in path:
                temp = None
                edges.append([node[0],node[1]]) 
                if node[-1] < 0:
                    temp = -node[-1]
                    lst.append(temp) 
                else:
                    temp = node[-1]
                    lst.append(temp) 

            ## find the minimum of lst ##
            residual_capacity_of_path = min(lst)
            print(residual_capacity_of_path) 
            ## update flow and max_flow ##
            max_flow += residual_capacity_of_path

            for item in edges:
                first = item[0]
                second = item[1]
                flow[first][second] += residual_capacity_of_path
                flow[second][first] -= residual_capacity_of_path

            ## update the residual_graph ##
            residual_graph = build_residual_graph(g,flow)

    return flow 


# function that creates the graph
def make_graph():
    
    print("LAYERS = " + str(LAYERS))
    g = [[] for i in range(len(LAYERS)) for j in range(LAYERS[i])]
    
    for n_out in range(LAYERS[1]): g[0].append([1+n_out, rd.randint(1,MAX_CAPACITY)])
    for n_in in range(LAYERS[-2]): g[len(g)-1+n_in-LAYERS[-2]].append([len(g)-1, rd.randint(1,MAX_CAPACITY)])
    
    start_index = 1
    for l in range(1,MAX_LAYERS):
        for n_in in range(LAYERS[l]):
            for n_out in range(LAYERS[l+1]):
                to_add = (MAX_SIZE_LAYERS*MAX_SIZE_LAYERS-LAYERS[l]*LAYERS[l+1])*(1-EDGE_DENSITY_FORWARD)/(MAX_SIZE_LAYERS*MAX_SIZE_LAYERS)
                if rd.uniform(0, 1)<EDGE_DENSITY_FORWARD+to_add: g[start_index+n_in].append([start_index+LAYERS[l]+n_out, rd.randint(1,MAX_CAPACITY)])
            
            if n_in!=LAYERS[l] and rd.uniform(0, 1)<EDGE_DENSITY_SIDE: 
                if rd.uniform(0, 1)<0.5: g[start_index+n_in].append([start_index+n_in+1, rd.randint(1,MAX_CAPACITY)])
                else: g[start_index+n_in+1].append([start_index+n_in, rd.randint(1,MAX_CAPACITY)])
                
        for n_in in range(LAYERS[l]):
            for n_out in range(LAYERS[l+1]):
                if rd.uniform(0, 1)<EDGE_DENSITY_BACKWARDS: 
                    found = False
                    for j in range(len(g[start_index+n_in])): 
                        if g[start_index+n_in][j][0] == start_index+LAYERS[l]+n_out:                        
                            found = True
                            break
                    if not found: 
                        print("adding "+str(start_index+LAYERS[l]+n_out)+" to " +str(start_index+n_in))
                        g[start_index+LAYERS[l]+n_out].append([start_index+n_in, rd.randint(1,MAX_CAPACITY)])
    
        start_index = start_index + LAYERS[l]

    return g
 

# function that prints the graph
def print_graph(g,flow):
    G = nx.DiGraph()
    for i in range(len(g)): G.add_node(i)
    for i in range(len(g)):
        for j in range(len(g[i])): G.add_edge(i,g[i][j][0],capacity=g[i][j][1], my_flow=flow[i][g[i][j][0]])
    for i in range(len(g)): print("from node %02i: " %(i) + str(g[i]))
    
    pos = []
    [pos.append(np.array([1.5*i, j])) for i in range(len(LAYERS)) for j in range(LAYERS[i])]
    nx.draw(G,pos, with_labels=True)
    
    colors =['r' if d['my_flow']==d['capacity'] else 'k' for u,v,d in G.edges(data=True)]  
    nx.draw_networkx_edges(G,pos,edge_color=colors)       
    labels=dict( [( (u,v,), (str(d['my_flow'])+"/"+str(d['capacity'])) ) for u,v,d in G.edges(data=True)] )
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, font_size=15, label_pos=0.3)
    
    colors =['c' for u in G.nodes()]  
    colors[0]=colors[NODES-1]='g'
    nx.draw_networkx_nodes(G,pos,node_color=colors)
   
    
# function that checks the validity of a flow    
def check_flow_validity(g,flow):
    for i in range(len(g)): 
        for j in range(len(g[i])):
            if g[i][j][1] < flow[i][g[i][j][0]]: 
                print("Flow impossible at edge (" + str(i) + "," + str(g[i][j][0]) + ")")
                return False
    
    return True
    

print("\n\n ******** GENERATING GRAPH ********" )     
g = make_graph()
flow = [[0 for i in range(NODES)] for j in range(NODES)]
plt.figure(1,figsize=(10,10))
print_graph(g,flow)

print("\n ******** PERFORMING EDMONDS-KARP ALGORTIHM ********" )    
flow = edmonds_karp(g)

print("\n ******** PERFORMING FLOW VALIDITY CHECK ********" )    
if check_flow_validity(g,flow): 
    print("FLOW FINAL = " + str(flow))
    print("FLOW VALUE = " + str(sum([flow[0][g[0][i][0]] for i in range(len(g[0]))]  )))
else: print("Invalid flow !")
    
plt.figure(1,figsize=(10,10))
print_graph(g,flow)
