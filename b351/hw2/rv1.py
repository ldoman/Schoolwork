#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW2 Maze Navigation

I'm abstracting the robot out of this implemntation because it adds unecessary complexity
without changing the real problem. Please see implementation in get_neighbors() below to 
see how the robots behavior is modeled without using any robot object.
"""

# Imports
from util import Graph,Node,Stack # Hw1 items

# TODO: Use?
class Robot(object):
	def __init__(self, pos):
		self.pos = pos
		self.path = []

	def get_path(self):
		return self.path

	def move(self, move):
		self.pos = move
		self.path.append(move)

class Maze(object):

	def __init__(self, fname):
		"""
		Passed file name must exist in current directory
		"""
		self.maze = self.__parse_maze(fname)

	def __parse_maze(self, fname):
		maze = []
		with open(fname, 'r') as f:
			for line in f:
				maze.append(list(map(int, list(line)[:-1])))
		return maze

	def get_maze(self):
		return self.maze

	def get_start_pos(self):
		"""
		Get all possible starting positions for robot

		Returns: 
			List of tuples (in form of (x,y) coordinate) where each item is
				a viable starting location
		"""
		start = []
		m_len = len(self.maze)
		for i in range(0, m_len):
			if self.maze[m_len-1][i] == 0:
				start.append((m_len-1,i))
		return start

	def get_end_pos(self):
		"""
		Get all possible end positions for robot

		Returns: 
			List of tuples (in form of (x,y) coordinate) where each item is
				a viable end location
		"""
		end = []
		m_len = len(self.maze)
		for i in range(0, m_len):
			# Top row
			if self.maze[0][i] == 0:
				end.append((0,i))
			# Left col
			if self.maze[i][0] == 0:
				end.append((i,0))
			# Right col
			if self.maze[i][m_len-1] == 0:
				end.append((i,m_len-1))
		return end

	def generate_adj_list(self):
		"""
		Generate an adj list in the format that the graph from hw1 expects
		"""
		adj_list = {}
		m_len = len(self.maze)
		for i in range(0, m_len):
			for j in range(0, m_len):
				if self.maze[i][j] == 1:
					continue
				adj_list[(i,j)] = self.get_neighbors((i,j))
		return adj_list

	def get_neighbors(self, pos):
		"""
		Return list of accessible neighbors from given position 
		"""
		neigh = []
		i = pos[0]
		j = pos[1]

		# Up
		if i-1 >= 0 and self.maze[i-1][j] == 0:
			neigh.append((i-1,j))
		# Left
		if j-1 >= 0 and self.maze[i][j-1] == 0:
			neigh.append((i,j-1))
		# Down
		if i+1 < len(self.maze) and self.maze[i+1][j] == 0:
			neigh.append((i+1,j))
		# Right
		if j+1 < len(self.maze) and self.maze[i][j+1] == 0:
			neigh.append((i,j+1))

		return neigh

	def can_navigate(self):
		"""
		Determines is any given maze can be navigated.

		Returns:
			True if possible to navigate
			False otherwise
		"""
		ret = False
		routes = {}
		starts = self.get_start_pos()
		ends = self.get_end_pos()

		graph = Graph(self.generate_adj_list())
		
		for start in starts:
			routes[start] = False
			dfs = graph.DFS(start)[1]
			for end in ends:
				if end in dfs:
					routes[start] = True
					ret = True

		for start,nav in routes.items():
			print("Possible to navigate maze from starting position %s: %s." % (start,nav))

		return ret


if __name__ == '__main__':
	file_name = 'f3.txt'
	maze = Maze(file_name)
	print(maze.can_navigate())


