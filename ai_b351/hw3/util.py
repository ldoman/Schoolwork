class Node(object):

    def __init__(self, name, neighbors = [], cost = 1):
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
                edges[(node, node_name)] = 1
                edges[(node_name, node)] = 1

        return nodes, edges

    def get_edges(self):
        edge_list = [(x,y) for x in self.adj_list.keys() for y in self.adj_list[x]]
        return edge_list

    def get_nodes(self):
        node_list = [i.get_name() for i in self.nodes]
        return node_list
        
    def get_node(self, name):
        query = [n for n in self.nodes if n.get_name() == name]
        node = query[0] if len(query) > 0 else None
        if not Node:
            print("No node found")
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

    def reach(self, node1, node2):
        return (node1 in self.DFS(node2))

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
