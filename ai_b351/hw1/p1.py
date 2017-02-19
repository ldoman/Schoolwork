#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW1 Drone Navigation + ATTEMPTED EXTRA CREDIT


  -oooooooooooooooooooooooooooooooooooo/ 
 +MMmmmmmmmmmmmmmmMMNmmmmmNMMmmmmmmmMMd 
 +MM:             yyo     oMM-      NMd 
 +MM:                     oMM-      NMd 
 +MM:   A 8       dmy B 6 oMM-  H 5 NMd 
 +MM:             NMd               NMd 
 +MM:             NMd     :oo`      NMd 
 +MMo:::::::::::::NMm:   :yMM+:`  ::NMd 
 +MMMMMMMMMMMMMMMMMMMM` .MMMMMM-  MMMMd 
 /NN:     `mNh   `NMd    `oMM-     `NMd 
                  NMd     -//`      NMd 
 /mm- C 11 dmy I12 Md J 7 `..   G 4 NMd 
 +MM:      NMd    NMd     oMM-      NMd 
 +MM:      NMd    NMd     oMM-      NMd 
 +MMMMMMMMMMMMMMMMMMMMMMMMMMMMM-  MMMMd 
 +MMo::::::NMm::::::::::::yMM+:`  ::NMd 
 +MM:      NMd             ``       NMd 
 +MM:      dmh            :yy.      NMd 
 +MM:  F 9         D 2    oMM-  K 3 NMd 
 +MM:      dmy            oMM-      NMd 
 +MM/`   ``NMd            oMM:``````NMd 
 +MMMM:  NMMMMMMMMMMM/  .MMMMMMMMMMMMMd 
 +MM+:`  -::::::hMM::`   :::::::::::NMd 
 +MM:           yMM`                NMd 
 +MM:           yMM`                NMd 
 +MM:    E 10   yMM`      L 1       NMd 
 +MM:           yMM`                NMd 
 +MM:           yMM`                NMd 
 +MMNmmmmmmmmmmmNMMNmy   ymmmmmmmmmNMMd 
 .+++++++++++++++++++:   :++++++++++++/

"""

# Constants
ROOM_SCAN_TIME = 4

class Node(object):

    def __init__(self, name, neighbors = [], cost = ROOM_SCAN_TIME):
        self.name = name
        self.neighbors = neighbors
        self.cost = cost

    def get_name(self):
        return self.name

    def get_cost(self):
        return self.cost
    
    def get_neighbors(self):
        return self.neighbors
    
    def add_neighbor(self, node):
        self.neighbors.append(node)

    def __str__(self):
        return self.get_name()

    def __eq__(self, other): 
        return self.get_name() == other.get_name()

class Graph(object):

    def __init__(self, adj_list = None):
        self.adj_list = adj_list if adj_list else {}
        self.nodes, self.edges = self.__parse_adj_list(adj_list)

    def __parse_adj_list(self, adj_list):
        """Parses the adjacency list and generates 'Node' objects and an edge weights dictionary.
        
        Args:
            adj_list (list): Adjacency list in format of {node_name: [neighbors],...}

        Returns:
            List of 'Node' objects
            Dict of edge weights with duplicate entries in form of (a,b) = 2 and (b,a) = 2
        """
        nodes = []
        edges = {}
        for node_name, neighbors in adj_list.items():
            nodes.append(Node(node_name, neighbors))
            for node in neighbors:
                if node_name == 'u' or  node == 'u':
                    # Assuming U to be perfectly in between l and c
                    edges[(node, node_name)] = 3.5
                    edges[(node_name, node)] = 3.5
                else:
                    edges[(node, node_name)] = 2
                    edges[(node_name, node)] = 2

        return nodes, edges

    def add_edge(self, node1, node2):
        """ You provide code here
        """

    def get_edges(self):
        edge_list = [(x,y) for x in self.adj_list.keys() for y in self.adj_list[x]]
        return edge_list

    def get_nodes(self):
        node_list = [i.get_name() for i in self.nodes]
        return node_list
        
    def get_node(self, name):
        node = [n for n in self.nodes if n.get_name() == name][0]
        return node
    
    def DFS(self, node):
        visited = []
        reachable = [node]
        s = Stack()
        s.push(node)
        while not s.empty():
            v = s.pop()
            if v not in reachable:
                reachable.append(v)
            if v not in visited:
                visited.append(v)
                for w in self.adj_list[v]:
                    s.push(w)
        return reachable, visited

    def DFS_Mod(self, node_name):
        """Modified DFS that tracks time required to traverse and scan rooms. 
        When DFS jumps after reaching a dead end, min_distance is called to 
        calculate the most efficient route to that room and add that time to
        the total count, because in real life the drone won't magically teleport
        rooms.
        
        Args:
            node_name (str): Name of node to start DFS at

        Returns:
            List of reachable 'Node' objects
            List of visited 'Node' objects
            Time it took drone to scan all visited rooms, accounting for scan + travel time.
        """
        node = self.get_node(node_name)
        visited = []
        reachable = [node]
        s = Stack()
        
        time = 0
        current = node 
        
        s.push(node)
        while not s.empty():
            prev = current
            current = s.pop()
            
            # Add time for travel to current node and scan
            if current.get_name() in prev.get_neighbors():
                try:                
                    time = time + self.edges[(prev.get_name(), current.get_name())] + current.get_cost()
                except KeyError:
                    pass
            else: # TODO: Fix me. Uncomment the print to see partial correctness
                travel_time = self.min_distance(prev, current)
                #print("Time between %s and %s : %d" % (current.get_name(), prev.get_name(), travel_time))
                time = time + current.get_cost() + travel_time
            
            if current not in reachable:
                reachable.append(current)
            if current not in visited:
                visited.append(current)
                for w in (self.get_node(node) for node in current.get_neighbors()):
                    #print(w)
                    s.push(w)
        return reachable, visited, time

    def reach(self, node1, node2):
        return (node1 in self.DFS(node2))

    # TODO: Fix. Returns penalized paths in some instances, and the correct values in others.
    def min_distance(self, current_node, end_node, visited = [], next_node = None):
        if current_node == end_node:
            return 0
        elif end_node.get_name() in current_node.get_neighbors():
            return self.edges[(current_node.get_name(), end_node.get_name())]
        else:
            if next_node is None:
                neighbors = [self.get_node(n) for n in current_node.get_neighbors()]
                ret = map(lambda n: self.min_distance(current_node, end_node, visited, n), neighbors)
                return min(ret)
            elif next_node in visited:                
                return 100 # Penalize cyclic paths
            else:
                visited.append(current_node)           
                return self.edges[(current_node.get_name(), next_node.get_name())] + \
                    self.min_distance(next_node, end_node, visited)
                    

class Stack:

    def __init__(self):
        self.stack = []

    def empty(self):
        return self.stack == []

    def pop(self):
        if self.empty():
            return None
        else:
            item = self.stack.pop()
            return item

    def push(self, item):
        self.stack.append(item)

if __name__ == "__main__":

    # Adjacency list for rooms drone needs to scan
    adj_list = {'u': ['c','l'],
			'c': ['i','u'],
			'i': ['c'],
			'l': ['d','u'],
			'd': ['f','k'],
			'f': ['e','d'],
			'e': ['f'],
			'k': ['g','d'],
			'g': ['j','h','k'],
			'j': ['b','g'],
			'h': ['b','g'],
			'b': ['a','j','h'],
			'a': ['b']}

    rooms = Graph(adj_list)
    start = 'u'

    print(rooms.DFS(start)[1]) # Original DFS path
    
    dfs_mod = rooms.DFS_Mod(start) # Modified DFS
    print([i.get_name() for i in dfs_mod[1]])
    print("Time: " + str(dfs_mod[2]))
