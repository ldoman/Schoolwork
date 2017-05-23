import facebook
import os
import requests
import sys

responses = {'ayy': 'lmao',
			 'what the ay': 'What the ayy did you just say to me you little lmao?'}

class Automod(object):
	"""
	Automod object capable of parsing Facebook group feeds and replying when appropriate.

	Attrs:
		token (str): Token required for FB account
		fb (FB Graph): Instance of FB graph API
		id (int): Facebook user ID of automod
		g_id (int): ID of group to "moderate"
	
	"""
	def __init__(self, group_name = 'test_env'):
		self.token = self.get_token()
		self.fb = facebook.GraphAPI(self.token)
		self.id = self.fb.get_object("me")['id']
		self.g_id = self.get_group_id(group_name)

	def get_token(self):
		token = None
		with open('token.txt') as tok:
			token = tok.readlines()[0]
		if not token:
			print "Error getting token. Aborting..."
			sys.exit(1)
		return token

	def get_group_id(self, mod_group):
		g_id = None
		groups = self.fb.get_connections("me", "groups")['data']
		for group in groups:
			if group['name'] == mod_group:
				g_id = group['id']
		if not g_id:
			print "Error getting group id. Aborting..."
			sys.exit(1)
		return g_id

	def scan(self):
		#self.fb = facebook.GraphAPI(self.token)
		feed = self.fb.get_connections(self.g_id, "feed")['data']
		for post in feed:
			reply_flag = True
			post_data = []
			try:
				post_data.append(post['message'])
				print 'Post: ' + post['message']
				comments = self.fb.get_connections(post['id'], 'comments')['data']
				for com in comments:
					post_data.append(com['message'])
					print 'Comment: ' + com['message']
					if com['from']['id'] == self.id:
						reply_flag = False # TODO: fix
				response = self.get_action(post_data)
				print 'Reply: ' + str(response)
				print ''
			except KeyError:
				continue

	def get_action(self, data):
		"""
		Given a list of texts determine appropriate response.

		Args:
			data (List of str): The original post message and all comments

		Returns:
			String if response required
			None otherwise
		"""
		for query, resp in responses.iteritems():
			for mes in data:
				if query in mes:
					return resp
		return None


if __name__ == "__main__":
	Automod().scan()

