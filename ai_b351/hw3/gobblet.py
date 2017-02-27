#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW3 Gobblet
"""

# Imports
from pprint import pformat,pprint
import time
from util import Graph,Node,Stack # Hw1 items

# Constants
BOARD_SIZE = 4
player_modes = {'h2': [0,0], 
				'hr': [0,1],
				'rh': [1,0],
				'r2': [1,1]}

class Gobblet(object):
	""" Gobblet game peice object """
	def __init__(self, size, color, stack = []):
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
		#return "Gobblet: %s, %s, %s" % (self.size, self.color, self.stack)
		return "%s%s" % (self.size, self.color)

class Board(object):
	""" Gobblet board object """
	def __init__(self):
		self.grid = [[None for n in range(0,BOARD_SIZE)] for n in range(0,BOARD_SIZE)]

	def get_board(self):
		return self.grid

	def get_gob_at(self, pos):
		return self.grid[pos[0]][pos[1]]

	def move(self, pos_1, pos_2):
		"""
		Move a Gobblet from one position on the board to another.

		Args:
			pos_1(Tuple): Current postion of gobblet to move. Tuple in form of (x, y) position on board.
			pos_2(Tuple): Gobblet destination. Tuple in form of (x, y) position on board.

		Returns:
			0: If successful
			-1: If move is invalid
		"""
		src = self.get_gob_at(pos_1)
		dest = self.get_gob_at(pos_2)

		print(src)
		print(dest)

		if not src:
			print("Invalid move. Source Gobblet does not exist.")
			return -1
		
		src_top = src.pop()
		src_size = src_top[0]
		src_color = src_top[1]
		old_gob = src
		old_gob.update()

		print(old_gob)

		if dest:
			if dest.get_size() >= src.get_size():
				print("Invalid move. Destination Gobblet of same or larger size.")
				return -1

			new_stack = dest.get_stack()
		else:
			new_stack = []

		new_gob = Gobblet(src_size, new_stack)
		new_gob.push((src_size,src_color))

		self.grid[pos1[0]][pos1[1]] = old_gob
		self.grid[pos2[0]][pos2[1]] = new_gob

		return 0

	def place(self, gob, pos):
		"""
		Places a gobblet from off the board on it

		Args:
			gob (Gobblet): Gobblet to place
			pos (Tuple): Gobblet destination. Tuple in form of (x, y) position on board.

		Returns:
			0: If successful
			-1: If move is invalid
		"""
		dest = self.get_gob_at(pos)

		#print(dest)
		
		src_top = gob.pop()
		src_size = gob.get_size()
		src_color = gob.get_color()
		
		if dest:
			if dest.get_size() >= gob.get_size():
				print("Invalid place. Destination Gobblet of same or larger size.")
				return -1

			new_stack = dest.get_stack()
		else:
			new_stack = []

		new_gob = Gobblet(src_size, new_stack)
		new_gob.push((src_size,src_color))

		self.grid[pos[0]][pos[1]] = new_gob

		return 0

	def check_win(self):
		"""
		Check the board to see if a player has reached a win state.

		Returns:
			-1: No wins
			0: Black wins
			1: White wins
		"""
		white_dl = black_dl = white_dr = black_dr = 0
		for i in range(0, BOARD_SIZE):
			white_v = white_h = black_v = black_h = 0
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
		"""
		Evaluates proximity to a win for the passed player.

		Args:
			player (int): Player to evaluate. (0 = Black, 1 = White)

		Returns:
			Int: Player's largest number of Gobblets in a row
		"""
		player_dl = player_dl = player_v = player_h = 0
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
		temp = [[str(g) for g in r] for r in self.grid]
		return pformat(temp)

class Player(object):
	""" Gobblet board object """
	def __init__(self, player, color):
		self.player = player
		self.color = color
		self.gobblets = [Gobblet(4, self.color, [1,2,3,4]) for n in range(0,3)]

	def get_move(self, board):
		"""
		Returns an action to make for the player.

		Args:
			board (Board): Gobblet board

		Returns:
			Tuple (action_type, (positions)) where action type is either 'move' or 'place'

		"""
		#actions = ['move', 'place']
		if self.player == 0: # Human
			print(board)
			self.print_state()
			action = int(input("Place off board gob (0) or Move one on the board (1): "))
			if action == 0:
				gob = int(input("Which gob? Index 0, 1, or 2: "))
				posx = int(input("x destination: "))
				posy = int(input("y destination: "))
				return (action, (gob, (posx, posy)))
			elif action == 1:
				pos = input("Source position in form (x,y): ")
				pos2 = input("Destination position in form (x,y): ")
				return (action, (pos, pos2))
			else:
				print("Invalid input")
				#self.get_move(board)
		elif self.player == 1: # Robot
			print("Doing robot things")
			# TODO
		else:
			print("Invalid player type")

	def get_gob(self, i):
		return self.gobblets[i]

	def remove_stack(self, i):
		gob = self.get_gob(i)
		if gob:
			new_size = gob.get_size()
			gob.pop()
			if new_size > 0:
				new_gob = Gobblet(new_size, self.color, gob.get_stack())
			else:
				new_gob = None
			self.gobblets[i] = new_gob
		else:
			print("Gob doesn't exist")

	def print_state(self):
		print([str(g) for g in self.gobblets])
		#print("Current off board stacks: %s") % (str(self.gobblets))


def gobby(players, level, time):
	"""
	Plays Gobblet witg passed parameters. 
	Index 0 of turns will always be black
	Index 1 of turns will always be white

	"""
	state = None
	history = []
	turns = player_modes[players]
	board = Board()
	black_player = Player(turns[0], 0)
	white_player = Player(turns[1], 1)

	players = [black_player, white_player]

	while state not in history or board.check_win() != -1:
		#history.append(state)
		current_turn = len(history) % 2
		current_player = players[current_turn]

		start_time = 0#time.time()
		move = -1
		#while start_time < start_time + time and move == -1:
		move = current_player.get_move(board)
		if move == -1:
			print("Time exceeded.")
			return -1

		success = make_move(move, current_player, board)
		while success == -1:
			print("Invalid move. Try again.")
			move = current_player.get_move(board)
			success = make_move(move, current_player, board)

		state = board.get_board()


def make_move(move, player, board):
	if move[0] == 0: # Place
		gob = player.get_gob(move[1][0])
		pos = move[1][1]
		valid = board.place(gob, pos)
	elif move[0] == 1: # Move
		pos = move[1][0]
		pos2 = move[1][1]
		valid = board.move(pos, pos2)
	else:
		valid = -1
		print("Invalid move")

	return valid

if __name__ == '__main__':
	gobby('h2', 4, 20)

