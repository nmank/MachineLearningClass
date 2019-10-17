import Maze
import matplotlib.pyplot as plt
import numpy as np





def play(iterations, g, discount = .2, lr = .1, eps = .5, show = False):
	all_rewards = np.zeros(iterations)
	for i in range(iterations):
		print('iteration '+str(i+1)+'...')
		g.start_game(epsilon = eps)
		end = g.check_end()
		while not end:
			if show:
				g.show_board()
			choice = g.make_choice()
			old_pos = g.update(choice)
			# print('STATE')
			# print(g.state)
			R = g.get_reinforcement(old_pos)
			g.learn(old_pos,choice,R)
			end = g.check_end()
			all_rewards[i] += R
	return all_rewards, g.Q_table, g
			

#training
g = Maze.game()
y = []
Q_tables = []
print('round 1 ...')
tmp_y, tmp_Q, g = play(1000,g,eps = 0.5)
y.append(tmp_y)
Q_tables.append(tmp_Q)

print('round 2 ...')
tmp_y, tmp_Q, g = play(1000,g,eps = 0.3)
y.append(tmp_y)
Q_tables.append(tmp_Q)

print('round 3 ...')
tmp_y, tmp_Q, g = play(1000,g,eps = 0.2)
y.append(tmp_y)
Q_tables.append(tmp_Q)

total_reward = np.ndarray.flatten(y).T

plt.plot(total_reward)
plt.show()