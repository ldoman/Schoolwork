#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW3 Goblet
"""

# Imports
from pprint import pformat,pprint
from util import Graph,Node,Stack # Hw1 items

# Constants
BOARD_SIZE = 4

class Goblet(object):
	""" Goblet game peice object """
	def __init__(self, size, color, stack = []:
		self.size = size
		self.color = color
		self.stack = stack

	def get_size(self):
		return self.size

	def get_color(self):
		return self.color

	def get_stack(self):
		return self.stack

	def pop(self):
        if self.stack == []:
            return None
        else:
            item = self.stack.pop()
            return item

    def push(self, item):
        self.stack.append(item)

	def update(self):
		top_gob = self.pop()
		if top_gob[0] != self.size:
			self.size = top_gob[0]
			self.color = top_gob[1]
			self.push(top_gob)

	def __str__(self):
		return "Goblet: %s, %s, %s" % (self.size, self.color, self.stack)

class Board(object):
	""" Goblet board object """
	def __init__(self):
		self.grid = [[None for n in range(0,BOARD_SIZE)] for n in range(0,BOARD_SIZE)]

	def get_board(self):
		return self.board

	def get_gob_at(self, pos):
		return self.grid[pos[0]][pos[1]]

	def move(self, pos_1, pos_2):
		src = get_gob_at(pos_1)
		dest = get_gob_at(pos_2)

		print(src)
		print(dest)

		if not src:
			print("Invalid move. Source goblet does not exist.")
			return -1
		
		src_top = src.pop()
		src_size = src_top[0]
		src_color = src_top[1]
		old_gob = src
		old_gob.update()

		print(old_gob)

		if dest:
			if dest.get_size() >= src.get_size():
				print("Invalid move. Destination goblet of same or larger size.")
				return -1

			new_stack = dest.get_stack()
		else:
			new_stack = []

		new_gob = Goblet(src_size, new_stack)
		new_gob.push((src_size,src_color))

		self.grid[pos1[0]][pos1[1]] = old_gob
		self.grid[pos2[0]][pos2[1]] = new_gob

		return 0

	def check_win(self):
		white_dl, black_dl, white_dr, black_dr = 0
		for i in range(0, BOARD_SIZE):
			white_v, white_h, black_v, black_h = 0
			for j in range(0, BOARD_SIZE):
				# Get current tile colors
				color_v = self.grid[i][j]
				color_h = self.grid[j][i]
				
				# Check horizonal
				white_h = white_h + 1 if color_h == 1 else 0
				black_h = black_h + 1 if color_h == 0 else 0

				# Check vertical
				white_v = white_v + 1 if color_v == 1 else 0
				black_v = black_v + 1 if color_v == 0 else 0

				# Check diagonal left -> right
				white_dl = white_dl + 1 if i == j and color_v == 1 else 0
				black_dl = black_dl + 1 if i == j and color_v == 0 else 0

				# Check diagonal right -> left
				white_dr = white_dr + 1 if i + j == BOARD_SIZE-1 and color_v == 1 else 0
				black_dr = black_dr + 1 if i + j == BOARD_SIZE-1 and color_v == 0 else 0

				# Check for white wins
				if (white_h == BOARD_SIZE or white_v == BOARD_SIZE or 
						white_dl == BOARD_SIZE or white_dr == BOARD_SIZE):
					return 1

				# Check for black wins
				if (black_h == BOARD_SIZE or black_v == BOARD_SIZE or 
						black_dl == BOARD_SIZE or black_dr == BOARD_SIZE):
					return 0

		return -1

	def evaluation(self, player):
		player_dl, player_dl, player_v, player_h = 0
		history = []
		for i in range(0, BOARD_SIZE):
			# Record found proximity to win
			history.append(player_h)
			history.append(player_v)
			player_v, player_h = 0

			for j in range(0, BOARD_SIZE):
				# Get current tile colors
				color_v = self.grid[i][j]
				color_h = self.grid[j][i]

				# Check win scenarios
				player_h = player_h + 1 if color_h == player else 0
				player_v = player_v + 1 if color_v == player else 0
				player_dl = player_dl + 1 if i == j and color_v == player else 0
				player_dr = player_dr + 1 if i + j == BOARD_SIZE-1 and color_v == player else 0

		return max(player_dl, player_dr, max(record))


	def __str__(self):
		return pformat(self.grid)

if __name__ == '__main__':
	# TODO

