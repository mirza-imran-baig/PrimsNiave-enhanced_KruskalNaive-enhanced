#PRIMS NAIVE

from heapq import *
import time
import sys
import math
import random

def decode(i):
    k = math.floor((1+math.sqrt(1+8*i))/2)
    return k,i-k*(k-1)//2

def rand_pair(n):
    return decode(random.randrange(n*(n-1)//2))

def rand_pairs(n,m):
    return [decode(i) for i in random.sample(range(n*(n-1)//2),m)]



# Naive
class Graph:
    def __init__(self, num_of_nodes):
        self.m_num_of_nodes = num_of_nodes
        # Initialize the adjacency matrix with zeros
        self.m_graph = [[0 for column in range(num_of_nodes)] 
                           for row in range(num_of_nodes)]
        #The above two lines will create and initialize the metrix with zeros and the size of the matrix will depend on the parameter num_of_nodes

    
        
    def add_edge(self, node1, node2, weight):
        self.m_graph[node1][node2] = weight
        self.m_graph[node2][node1] = weight
        #The add_edge function will update the matrix(m_graph) for every edge and we have two lines of code because the graph is undirected

        

    def prims_mst(self):
        postitive_inf = float('inf')
        #First we define a positive infinite number so that we have the maximum number possible for weight comparision and it'll help us in relaxing of nodes
        
        visited_nodes = [False for node in range(self.m_num_of_nodes)]
        #Now, we create an array (visited_nodes) of size (m_num_of_nodes) and initialize every element as false and later on we update the node value to true
        #this will help us in not selecting the same edge multiple times and to actually stop
        
        result = [[0 for column in range(self.m_num_of_nodes)] 
                    for row in range(self.m_num_of_nodes)]
        #we create a new matrix called(result) to store the result
        
        indx = 0
       # for i in range(self.m_num_of_nodes):
        #    print(self.m_graph[i])
        #print(visited_nodes)
        #The above two print statements will print the adjacency matrix(m_graph) and the array visited_nodes 
       
        while(False in visited_nodes):
            #This while loop will run for every false node in the array (visited_nodes)
            minimum = postitive_inf
            #The positive infinite number is used as a minimun number for the relaxatin process


            start = 0
            #The starting node

            end = 0
            # The ending node

            for i in range(self.m_num_of_nodes):
                if visited_nodes[i]==True: #means the node is visited
                #If the visited of node is True then we check the neighbours
                    for j in range(self.m_num_of_nodes):
                    # If the analyzed node have a path to the ending node AND its not included in the MST (to avoid cycles)
                        if (not visited_nodes[j] and self.m_graph[i][j]>0):  
                        #the first part is for checking that the neighbouring node is not visited
                        #second part is for checking if there is an edge between current node and neighbouring node
                            if self.m_graph[i][j] < minimum: 
                            #check if edge is less then minimun
                            #If the weight path analized is less than the minimum of the MST
                                minimum = self.m_graph[i][j]
                                start, end = i, j
                                #Update the minimum weight starting node and ending node
            
            
            visited_nodes[end] = True  
            #Now, we mark the ending vertex as true in the array (visited_node)
            #Since we added the ending vertex to the MST, it's already selected
          
            result[start][end] = minimum
            #Upadte the feilds of resultant matrix (result) with the mminimum values
            
            if minimum == postitive_inf:
                result[start][end] = 0
                #This will allow us to mark the first node as zero since, all the nodes are marked as positive infinite at the strat 
            
            # print("(%d.) %d - %d: %d" % (indx, start, end, result[start][end]))
            indx += 1
            
            result[end][start] = result[start][end]
            
        # Print the resulting MST
        # for node1, node2, weight in result:
        cost=0
        for i in range(self.m_num_of_nodes):
            for j in range(i, self.m_num_of_nodes):
                if result[i][j] > 0:
                    #print("%d - %d: %d" % (i, j, result[i][j]))
                    cost+=result[i][j]
        print("Final Min Cost is :", (cost))

        

# PRIMS WITH HEAP.

class GraphH:

    """ graph class """

    def __init__(self,v):

        self.v=v

        self.adjList={i:[] for i in range(v)}

    def addEdge(self,u,v,w):

        self.adjList[u].append((v,w))

        self.adjList[v].append((u,w))




def primsHeap(g):

    active_edges=[]
    #empty list that acts as heap

    visited=[False for i in g.adjList]

    cost=0

    heappush(active_edges,[0,0,0])
    #inserting in heap

    while len(active_edges)!=0:
    #loops run untill heap empty
        wt,end,start=heappop(active_edges)

        if visited[end]==True:
            continue

        visited[end]=True
        # print(start, "-",end, " : ",wt)
        cost+=wt

        for v,w in g.adjList[end]:
            if not visited[v]:
                heappush(active_edges,[w,v,end])
    print("Final Min Cost is :",(cost))






#KRUSKAL'S NAIVE.


class GraphK:
    def __init__(self, nodes):
        self.nodes = nodes
        # Create Adjacent Matrix of size nodes x nodes with all 0's
        self.adjMat = [[0 for i in range(nodes)] for j in range(nodes)]

    def addEdge(self, u, v, w):
        # undirected graph
        self.adjMat[u][v] = w
        self.adjMat[v][u] = w

    def krushkals(self):
        # Create list of list with (weight, node1, node2) and sort on weight(lowest weight should come first)
        sortWeights = []
        for i in range(self.nodes):
            for j in range(i, self.nodes):
                # Add t  o list if only has an edge using self.adjMat[i][j]>0
                # weight, node1, node2. 'adjMat' is the matrix already initialized in init method
                if (self.adjMat[i][j] > 0):
                    sortWeights.append([self.adjMat[i][j], i, j])

        # Now, sort the list of list
        sortWeights.sort()
        visited = [[0 for i in range(self.nodes)] for j in range(self.nodes)]

        # Final min Spanning tree cost
        cost = 0

        # iterate through the sortWeigth list and check for every element
        for i in range(len(sortWeights)):
            weight = sortWeights[i][0]
            node1 = sortWeights[i][1]
            node2 = sortWeights[i][2]
            # print("KRUSHKALS >>>>:", weight)

            # BFS to check if nodes form a cycle
            queue = []
            map={}
            queue.append(node1)
            check = False
            while(len(queue)!=0):
                front=queue.pop(0)
                if front == node2:
                    check=True
                    break
                map[front]=1
                for i in range(len(visited[front])):
                    if ((i not in map) and visited[front][i]>0):
                        queue.append(i)
            if (check==False):
                cost += weight
                visited[node1][node2]=weight
                visited[node2][node1]=weight
                # Print the visited in this format: {node1 - node2: weight}
                # print(node1, "-", node2, ": ", weight)
        print("Final Min Cost is : " + str(cost))

        
# KRUSKALS UNION FIND


class GraphUnionFind:

	def __init__(self, vertices):
		self.V = vertices 
		self.graph = []

	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])

	def find(self, parent, i):
		if parent[i] != i:
			parent[i] = self.find(parent, parent[i])
		return parent[i]

	# A function that does union of two sets of x and y
	# (uses union by rank)
	def union(self, parent, rank, x, y):
		
		# Attach smaller rank tree under root of
		# high rank tree (Union by Rank)
		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x

		# If ranks are same, then make one as root
		# and increment its rank by one
		else:
			parent[y] = x
			rank[x] += 1

	# The main function to construct MST using Kruskal's
		# algorithm
	def KruskalMST(self):

		result = [] # This will store the resultant MST

		# An index variable, used for sorted edges
		i = 0

		# An index variable, used for result[]
		e = 0

		# Step 1: Sort all the edges in
		# non-decreasing order of their
		# weight. If we are not allowed to change the
		# given graph, we can create a copy of graph
		self.graph = sorted(self.graph,
							key=lambda item: item[2])

		parent = []
		rank = []

		# Create V subsets with single elements
		for node in range(self.V):
			parent.append(node)
			rank.append(0)

		# Number of edges to be taken is equal to V-1
		while e < self.V - 1:

			# Step 2: Pick the smallest edge and increment
			# the index for next iteration
			u, v, w = self.graph[i]
			i = i + 1
			x = self.find(parent, u)
			y = self.find(parent, v)

			# If including this edge doesn't
			# cause cycle, then include it in result
			# and increment the index of result
			# for next edge
			if x != y:
				e = e + 1
				result.append([u, v, w])
				self.union(parent, rank, x, y)
			# Else discard the edge

		minimumCost = 0
		# print("Edges in the constructed MST")
		for u, v, weight in result:
			minimumCost += weight
			# print("%d -- %d == %d" % (u, v, weight))
		print("Final Min Cost is :", minimumCost)

# Create graph object
n = 100
e = 1000
kruskal_naive = GraphK(n)
graphheap=GraphH(n)
prims_naive = Graph(n)
krushkals_union = GraphUnionFind(n)
for (u,v) in rand_pairs(n, e):
    graphheap.addEdge(u, v, u+v)
    prims_naive.add_edge(u ,v ,u+v)
    kruskal_naive.addEdge(u, v, u+v)
    krushkals_union.addEdge(u, v, u+v)

start1=time.time()
prims_naive.prims_mst()
end1=time.time()
print(end1-start1)

start2=time.time()
primsHeap(graphheap)
end2=time.time()
print(end2-start2)

start3=time.time()
kruskal_naive.krushkals()
end3=time.time()
print(end3-start3)

start4 = time.time()
krushkals_union.KruskalMST()
end4 = time.time()
print(end4-start4)