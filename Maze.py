import numpy as np 
import matplotlib.pyplot as plt

plt.ion()

class game():
	def __init__(self, discount = .2, lr = .5):
		self.board = np.genfromtxt('maze.csv',delimiter = ',')
		self.n, self.m = self.board.shape
		self.Q_table = np.zeros((self.n,self.m,4))
		self.lr = lr
		self.discount = discount
		

	def start_game(self, epsilon = .5):
		self.epsilon = epsilon
		self.board = np.genfromtxt('maze.csv',delimiter = ',')
		self.n, self.m = self.board.shape
		self.goal_pos = tuple(map(int,np.where(np.isnan(self.board) == True)))
		#self.state = (np.random.randint(self.n),np.random.randint(self.m))
		self.state = (self.n-1,self.m-1)
		self.board[self.state] = 3
		self.reward = 0

	def update(self, choice):
		new_state = [0,0]
		if choice == 0:
			new_state[0] = self.state[0] - 1
			new_state[1] = self.state[1]
		elif choice == 1:
			new_state[0] = self.state[0] 
			new_state[1] = self.state[1] + 1
		elif choice == 2:
			new_state[0] = self.state[0] + 1
			new_state[1] = self.state[1] 
		elif choice == 3: 
			new_state[0] = self.state[0] 
			new_state[1] = self.state[1] - 1
		new_state = tuple(new_state)
		self.board[self.state] = 1
		self.board[new_state] = 3
		old_pos = self.state
		self.state = new_state
		return old_pos

	def check_end(self):
		end = False
		if self.state == self.goal_pos:
			print('game over')
			print('reward was: '+ str(self.reward))
			end = True
		return end

	def get_valid_actions(self):
		invalid = []
		if self.state[0] == 0:
			invalid.append(0)
		if self.state[1] == 0:
			invalid.append(3)
		tmp = self.n -1		
		if self.state[0] == tmp:
			invalid.append(2)
		tmp = self.m -1			
		if self.state[1] == tmp:
			invalid.append(1)
		actions = list({0,1,2,3}-set(invalid))
		for a in actions:
			tmp = [0,0]
			if a == 0:
				tmp[0] = self.state[0] - 1
				tmp[1] = self.state[1]
			elif a == 1:
				tmp[0] = self.state[0] 
				tmp[1] = self.state[1] + 1
			elif a == 2:
				tmp[0] = self.state[0] + 1
				tmp[1] = self.state[1] 
			elif a == 3: 
				tmp[0] = self.state[0] 
				tmp[1] = self.state[1] - 1
			if self.board[tuple(tmp)] == 2:
				invalid.append(a)
		actions = list(set(actions)-set(invalid))
		return actions

	def make_choice(self):
		coin_flip = np.random.rand()
		actions = self.get_valid_actions()
		if coin_flip > self.epsilon and np.count_nonzero(self.Q_table[self.state][actions]) > 0:
			tmp = np.argmax(self.Q_table[self.state][actions])
			choice = actions[tmp]
		else:
			choice = np.random.choice(actions)
		return choice
		

	def get_reinforcement(self,old_pos):
		R = 0
		if self.state == self.goal_pos:
		 	R = 100
		elif np.sum(np.abs(np.array(self.state) - np.array(self.goal_pos))) > np.sum(np.abs(np.array(old_pos) - np.array(self.goal_pos))):
			#R = -.1
		self.reward += R
		return R

	def learn(self, old_pos, action, R):
		self.Q_table[old_pos][action] = (1-self.lr)*self.Q_table[old_pos][action] + self.lr*(R + self.discount*np.max(self.Q_table[self.state]))
	
	def show_board(self):
		plt.imshow(self.board)
		plt.pause(.1)











	#0 is up, 1 is right, 2 is down, 3 is left

