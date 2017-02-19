#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW2 Maze Navigation
"""

# Imports
from util import Graph,Node,Stack
from rv1 import Maze

class Maze_Mk2(Maze):
	def can_navigate(self):
		"""
		Determines is any given maze can be navigated.

		Returns:
			True: if possible to navigate
			False: otherwise
		"""
		ret = False
		routes = {}
		starts = self.get_start_pos()
		ends = self.get_end_pos()

		graph = Graph(maze.generate_adj_list())
		opt_path = []

		for start in starts:
			routes[start] = False
			for end in ends:
				start_node = graph.get_node(start)
				end_node = graph.get_node(end)
				test = start_node.get_name()
				a_star = graph.a_star(start_node, end_node)
				#print(a_star)
				if a_star[0]:
					routes[start] = True
					ret = True
					opt_path.append(a_star[1])

		test = [i.get_name() for i in opt_path[0]]
		print(test)
		for start,nav in routes.items():
			print("Possible to navigate maze from starting position %s: %s." % (start,nav))

		return ret

if __name__ == '__main__':
	file_name = 'f3.txt'
	maze = Maze_Mk2(file_name)
	graph = Graph(maze.generate_adj_list())
	print(maze.can_navigate())

