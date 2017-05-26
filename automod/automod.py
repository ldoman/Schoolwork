"""
Facebook AutoModerator for groups.
"""

__author__ = 'Luke Doman'

import csv
import facebook
import os
import requests
import sys

valid_actions = ['comment'] #TODO: tag, log, warn, ban

class Automod(object):
	"""
	Automod object capable of parsing Facebook group feeds and moderating when appropriate.

	Attrs:
		token (str): Token required for FB account
		fb (FB Graph): Instance of FB graph API
		id (int): Facebook user ID of automod
		g_id (int): ID of group to "moderate"
		rules (Dict): Dictionary of automod rules, in format of 'query':[(action, args)]
	"""
	def __init__(self, group_name = 'test_env', csv_name = 'Automod Rules - Sheet1.csv'):
		self.token = self.__get_token()
		self.fb = facebook.GraphAPI(self.token)
		self.id = self.fb.get_object("me")['id']
		self.g_id = self.__get_group_id(group_name)
		self.rules = self.__parse_rules(csv_name)

	def __get_token(self):
		token = None
		with open('token.txt') as tok:
			token = tok.readlines()[0]
		if not token:
			print "Error getting token. Aborting..."
			sys.exit(1)
		return token

	def __get_group_id(self, mod_group):
		g_id = None
		groups = self.fb.get_connections("me", "groups")['data']
		for group in groups:
			if group['name'] == mod_group:
				g_id = group['id']
		if not g_id:
			print "Error getting group id. Aborting..."
			sys.exit(1)
		return g_id

	def __parse_rules(self, csv_name):
		"""
		Parses logic for Automod from the csv to build the rules dictionary. 
		The dictionary is of the format: 'query':[(action, args)]
		"""
		rules = {}
		with open(csv_name) as data:
			csv_reader = csv.DictReader(data)
			for line in csv_reader:
				rule = []
				actions = line['Action(s)'].replace(" ", "").lower().split(',')
				for action in actions:
					if action not in valid_actions:
						continue
					rule.append((action, line['Args']))
				rules[line['Query']] = rule

		return rules

	def scan(self):
		""" Scan the given group for posts and respond when required """
		feed = self.fb.get_connections(self.g_id, "feed")['data']
		for post in feed:
			reply_flag = True
			post_data = []
			try:
				post_data.append(post['message'])
				#print 'Post: ' + post['message']
				comments = self.fb.get_connections(post['id'], 'comments')['data']
				for com in comments:
					post_data.append(com['message'])
					#print 'Comment: ' + com['message']
					if com['from']['id'] == self.id:
						reply_flag = False # Currently assumes if it already commented to not reply again
				response = self.get_actions(post_data)
				if reply_flag and response:
					self.action_router(post['id'], response)
			except KeyError:
				continue

	def get_actions(self, data):
		"""
		Given a list of texts determine appropriate response(s).

		Args:
			data (List of str): A single post message and all of it's comments

		Returns:
			List of actions if response required
			None otherwise
		"""
		actions = []
		for query, action in self.rules.iteritems():
			for message in data:
				if query in message:
					actions = actions + action
		return actions if actions else None

	def comment(self, post_id, message):
		"""
		Comment on a specific post with specified message.

		Args:
			post_id (int): ID of post to reply to - NOT comment ID
			message (str): Reply to publish
		"""
		print "Commenting..."
		url = "https://graph.facebook.com/{0}/comments".format(post_id)
		params = {'access_token' : self.token, 'message' : message}
		s = requests.post(url, data = params)

	def action_router(self, post_id, actions):
		""" 
		Receives a list of actions to perform and routes them to correct functions.
		This function must be below any functions it routes to for the action map to work.

		Args:
			post_id (int): ID of post to take action on.
			actions (list): List of actions to take
		"""
		action_map = {'comment': self.comment}
		for action in actions:
			action_map[action[0]](post_id, action[1])

if __name__ == "__main__":
	Automod().scan()

