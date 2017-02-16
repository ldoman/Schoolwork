#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW1 Drone Navigation + EXTRA EXTRA CREDIT


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

    def __init__(self, name):
        self.name = name
        self.cost = 4 # Minutes to scan the room

    def get_name(self):
        return name

    def get_cost(self):
        return cost

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

class Graph(object):

    def __init__(self, adj_list=None):
        if adj_list == None:
            adj_list = {}
        else:
            self.adj_list = adj_list

    def add_node(self, node):
        if node not in self.adj_list.keys():
            self.adj_list[node] = []

    def add_edge(self, node1, node2):
        """ You provide code here
        """

    def get_edges(self):
        edge_list = [(x,y) for x in self.adj_list.keys() for y in self.adj_list[x]]
        return edge_list

    def get_nodes(self):
        node_list = [i for i in self.adj_list.keys()]
        return node_list
    
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

    def DFS_Mod(self, node):
        visited = []
        reachable = [node]
		time = 0
        s = Stack()
        s.push(node)
        while not s.empty():
            v = s.pop()
			time = time + ROOM_SCAN_TIME
            if v not in reachable:
                reachable.append(v)
            if v not in visited:
                visited.append(v)
                for w in self.adj_list[v]:
                    s.push(w)
        return reachable, visited

    def reach(self, node1, node2):
        return (node1 in self.DFS(node2))
          
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

	print(rooms.DFS(start)[1])

'''

#mg = {1 : [2], 2: [3,1], 3: [1]}

#f = MyGraph(mg)
#f.draw()

mg2 = {1 : [2], 2: [1,3,4,5], 3: [2,4,5], 4 : [2,3,5,9], 5 : [2,3,4,9], 6 : [7,8], 7 : [6], 8 : [6], 9:[4,5]}

f2 = Graph(mg2)

	
f2.draw()

print(f2.get_edges())
print(f2.get_nodes())
for i in range(1,9):
    print(" Reachable from {0} is {1}".format(i, f2.DFS(i)[0]))


print(f2.reach(1,8))
print(f2.reach(1,9))

#f2.draw()

#f.add_edge(4,1)

#print(f.get_edges())
#print(f.get_nodes())

#f.draw()
#G = nx.Graph()
#G.add_edges_from([(1,2),(2,3),(3,4),(5,3),(1,5)])

#label = {1 : 1, 2 : 2, 3: 3,4:4,5:5}


# Need to create a layout when doing
# separate calls to draw nodes and edges

#pos = nx.spring_layout(G)

#nx.draw_networkx_nodes(G,pos, nodelist=[1,2,3,5], node_color='b', node_size=500, alpha=0.9)
#nx.draw_networkx_edges(G,pos, width = 2, edge_list = None, edge_color='r')
#plt.axis('off')
#nx.draw_networkx_labels(G,pos, label, font_size=16, font_color='w')
#plt.show()
'''
