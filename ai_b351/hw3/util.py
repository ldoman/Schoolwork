from Queue import PriorityQueue

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

    def min_distance(self, current_node, end_node, visited = [], next_node = None):
        if current_node == end_node:
            return 0
        elif end_node.get_name() in current_node.get_neighbors():
            return self.edges[(current_node.get_name(), end_node.get_name())]
        else:
            if next_node is None:
                neighbors = [self.get_node(n) for n in current_node.get_neighbors()]
                ret = map(lambda n: self.min_distance(current_node, end_node, visited, n), neighbors)
                return min(ret), visited
            elif next_node in visited:
                return 100 # Penalize cyclic paths
            else:
                visited.append(current_node)
                return self.edges[(current_node.get_name(), next_node.get_name())] + \
                    self.min_distance(next_node, end_node, visited)

    # TODO: Not optimal. Need to account for blocks in maze.
    def h_hat(self, node1, node2):
		"""
		Calculates the total number of moves required for node1 to reach node2.

		Args:
			node1 (Node): Start node to measure moves from
			node2 (Node): End node to measure moves to

		Returns:
			Int: number of moves
		"""
		moves_x = node1.get_name()[0] - node2.get_name()[0]
		moves_y = node1.get_name()[1] - node2.get_name()[1]
		ret = abs(moves_x) + abs(moves_y)
		return ret 

    def a_star(self, start_node, end_node):
		"""
		Finds the optimal path between 2 nodes

		Args:
			start_node (Node): Node to start search at
			end_node (Node): End node for search 

		Returns:
			Bool: Whether end node is reachable from start node
			List: Optimal path
		"""
		reachable = False
		visited = []
		frontier = PriorityQueue()
		frontier.put(start_node, 0)
		cost = {}
		cost[start_node] = 0

		while not frontier.empty():
			current_node = frontier.get()
			visited.append(current_node)
			if current_node == end_node:
				reachable = True
				break

			for neigh in [self.get_node(n) for n in current_node.get_neighbors()]:
				g_hat = cost[current_node] + self.edges[(current_node.get_name(), neigh.get_name())]
				if neigh not in cost or g_hat < cost[neigh]:
					cost[neigh] = g_hat
					f_hat = g_hat + self.h_hat(end_node, neigh)
					frontier.put(neigh, f_hat)

		return reachable, visited


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
