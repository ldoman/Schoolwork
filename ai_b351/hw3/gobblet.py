#!/usr/bin/env python3

__author__ = 'Luke Doman'

""" 
B351 HW3 Gobblet + Attempted extra credit: solution is scalable for any n x n board (given computational power and time).
"""

# Imports
from pprint import pformat,pprint
import sys
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
		"""
		Updates object parameters to match the current topmost in the stack.
		"""
		top_gob = self.pop()
		if top_gob[0] != self.size:
			self.size = top_gob[0]
			self.color = top_gob[1]
			self.push(top_gob)

	def __str__(self):
		color = "b" if self.color == 0 else "w"
		return "%s%s" % (self.size, color)

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

		new_gob = Gobblet(src_size, src_color, new_stack)
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
		if not gob:
			print("Invalid gobblet object.")
			return -1

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

		new_gob = Gobblet(src_size, src_color, new_stack)
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
				v = self.grid[i][j]
				h = self.grid[j][i]
				color_v = v.get_color() if v else -1
				color_h = h.get_color() if h else -1
				
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

	def __str__(self):
		temp = [[str(g) for g in r] for r in self.grid]
		return pformat(temp)

class Player(object):
	""" Gobblet board object """
	def __init__(self, player, color):
		self.player = player
		self.color = color
		self.gobblets = [Gobblet(4, color, [1,2,3,4]) for n in range(0,3)]

	def get_move(self, board, level):
		"""
		Returns an action to make for the player.

		Args:
			board (Board): Gobblet board
			level (int): Level of difficulty for AI

		Returns:
			Tuple (action_type, (positions)) where action type is either 'move' or 'place'

		"""
		# Human
		if self.player == 0:
			print(board)
			print ("")
			self.print_stacks()
			print ("")
			action = int(input("Place off board gobblet (0) or Move one on the board (1): "))
			if action == 0:
				gob = int(input("Gobblet from which stack index? (0,1,2) : "))
				posx = int(input("X destination: "))
				posy = int(input("Y destination: "))
				return (action, (gob, (posy, posx)))
				p1 = int(input("Source X position: "))
			elif action == 1:
				p2 = int(input("Source Y position: "))
				p3 = int(input("Destination X position: "))
				p4 = int(input("Destination Y position: "))

				if board.get_gob_at((p2, p1)).get_color != self.color:
					print("Invalid move. Cannot move opponent gobblet.")
					return self.get_move(board)

				return (action, ((p2, p1),(p4,p3)))
			else:
				print("Invalid input")
				return self.get_move(board)
		# AI
		elif self.player == 1: # Robot
			ab = AlphaBeta(level, self.color, board)
			move = ab.search()
			print(move)
			return move
		else:
			print("Invalid player type")
			return -1

	def get_gob(self, i):
		return self.gobblets[i]

	def get_stacks(self):
		return self.gobblets

	def remove_stack(self, i):
		gob = self.get_gob(i)
		if gob:
			new_size = gob.get_size()-1
			if new_size > 0:
				gob.pop()
				self.gobblets[i] = Gobblet(new_size, self.color, gob.get_stack())
			else:
				self.gobblets.pop(i)
		else:
			print("Gob doesn't exist")

	def print_stacks(self):
		print("Off board: " + str([str(g) for g in self.gobblets]))

	def __str__(self):
		return "Black" if self.color == 0 else "White"

class AlphaBeta(object): # TODO: Complete
	def __init__(self, level, player_color, board):
		self.player_color = player_color
		self.board = board
		self.level = level
		self.state = None
		self.actions = {}

	def cutoff_test(self, depth, state):
		"""
		Determines when to stop expanding alpha beta frontier

		Args:
			depth (int): Current depth

		Returns:
			0: Continue
			1: Cutoff
		"""
		if state.check_win() != -1 or depth > self.level:
			return True
		else:
			return False

	def search(self):
		move_value = self.max_value(state, -sys.maxsize, sys.maxsize)
		return self.actions[move_value]

	def max_value(self, state, alpha, beta, depth):
		"""
		Finds the max state attainable from current actions

		Args:
			state (Board): Current state of board
			alpha (int): Max value reachable from actions
			beta (int): Min value reachable from actions
			depth (int): Current depth of iteration

		Returns:
			Max valued state acheivable
		"""
		if self.cutoff_test(depth):
			return self.evaluation(state, self.player)
		move_value = -sys.maxsize
		for action in self.get_actions(state):
			move_value = max(move_value, self.min_value(self.result_state(state, action), alpha, beta))
			self.actions[move_value] = action
			if move_value >= beta:
				return move_value
			alpha = max(alpha, move_value)
		return move_value

	def min_value(self, state, alpha, beta, depth):
		"""
		Finds the min state attainable from current actions

		Args:
			state (Board): Current state of board
			alpha (int): Max value reachable from actions
			beta (int): Min value reachable from actions
			depth (int): Current depth of iteration

		Returns:
			Min valued state acheivable
		"""
		if self.cutoff_test(depth):
			return self.evaluation(state, self.player)
		move_value = sys.maxsize
		for action in self.get_actions(state):
			move_value = min(move_value, self.max_value(self.result_state(state, action), alpha, beta))
			if move_value >= beta:
				return move_value
			alpha = min(beta, move_value)
		return move_value

	def get_actions(self, state):
		"""
		Get all possible actions reachable from the current state

		Args:
			state (Board): Current state of board

		Returns:
			List of actions in form of (action_type, (positions)) where action type is either 'move' or 'place'
		"""
		actions = []

		for i in range(0, BOARD_SIZE):
			for j in range(0, BOARD_SIZE):
				gob = state.get_board()[i][j]

				# Options for placing off board pieces
				for stack in self.player.get_stacks():
					new_gob = stack.pop()
					if gob is not None and gob.get_size() < new_gob.get_size():
						actions.append((0, (new_gob, (i,j))))

				# Options for moving on board pieces
				# TODO: Implement

		return actions

	def result_state(self, state, action):
		"""
		Returns the state with the action applied to it
		"""
		# TODO: Implement

	def evaluation(self, state, player):
		"""
		Evaluates proximity to a win for the passed player. TODO: Improve by adding
		situational awareness (Ex. 3 in a row with no gobblet in the last spot is 
		better than 3 in a row with an oppononent gobblet of size 4 in the last spot.)

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
				v = state.get_board()[i][j]
				h = state.get_board()[j][i]
				color_v = v.get_color() if v else -1
				color_h = h.get_color() if h else -1

				# Check win scenarios
				player_h = player_h + 1 if color_h == player else 0
				player_v = player_v + 1 if color_v == player else 0
				player_dl = player_dl + 1 if i == j and color_v == player else 0
				player_dr = player_dr + 1 if i + j == BOARD_SIZE-1 and color_v == player else 0

		return max(player_dl, player_dr, max(history))


def gobby(players, level, time, display = False):
	"""
	Plays Gobblet with passed parameters. Assume index 0 of turns is always black player
	"""
	state = None
	history = []
	turns = player_modes[players]
	board = Board()
	black_player = Player(turns[0], 0)
	white_player = Player(turns[1], 1)

	players = [black_player, white_player]
	winner = -1

	while winner == -1:
		history.append(state)
		turn = len(history) % 2
		current_player = players[turn]

		if display:
			print ("")
			print("Current player : %s" % (current_player))
			print ("")

		start_time = 0#time.time()
		move = -1
		#while start_time < start_time + time and move == -1:
		move = current_player.get_move(board, level)
		if move == -1:
			print("Time exceeded.")
			return -1

		success = make_move(move, current_player, board)
		while success == -1:
			print("Invalid move. Try again.")
			move = current_player.get_move(board)
			success = make_move(move, current_player, board)

		state = board.get_board()
		winner = board.check_win()

	if winner == 0:
		print("Black wins!")
	elif winner == 1:
		print("White wins!")
	else:
		print("Tie")

def make_move(move, player, board):
	if move[0] == 0: # Place
		gob = player.get_gob(move[1][0])
		pos = move[1][1]
		valid = board.place(gob, pos)
		player.remove_stack(move[1][0])
	elif move[0] == 1: # Move
		pos = move[1][0]
		pos2 = move[1][1]
		valid = board.move(pos, pos2)
	else:
		valid = -1
		print("Invalid move")

	return valid

if __name__ == '__main__':
	gobby('h2', 4, 20, True)

