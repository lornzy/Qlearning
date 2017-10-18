
import numpy as np
import pandas as pd
import time
np.random.seed(2)
N_states = 6
Actions = ['left','right']
Epsilon = 0.9
Aloha = 0.1
Lambda = 0.9
MAX_episode = 15
Free_time = 0.1

def build_q_table(n_states, actions):
    table =pd.DataFrame(np.zeros((n_states, len(actions))), columns =actions)
    print(table)
    return table

#build_q_table(N_states, Actions)
def choose_action(state, q_table):
	state_actions = q_table.iloc[state, :]
	if (np.random.uniform()>Epsilon) or (state_actions.all()==0):
		action_name = np.random.choice(Actions)
	else:
		action_name = state_actions.argmax()
	return action_name

def get_env_feedback(S, A):
	if A =='right':
		if S == N_states -2:
			S_ = 'terminal'
			R = 1
		else:
			S_ = S + 1
			R = 0
	else:
		R= 0
		if S == 0:
			S_ = S
		else:
			S_ = S - 1
	return S_, R

def update_env(S, episode, step_counter):
	env_list = ['-']*(N_states-1) + ['T']
	if S == 'terminal':
		interaction = '指栽方%s:化方 = %s' %(episode+1, step_counter)
		print('\r{}'.format(interaction))
		time.sleep(2)
		print('\r                          ')
	else:
		env_list[S] = 'i'
		interaction = ''.join(env_list)
		print('\r{}'.format(interaction))
		time.sleep(Free_time)

def rl():
	q_table = build_q_table(N_states, Actions)
	for episode in range(MAX_episode):
		step_counter = 0
		S = 0
		is_terminated = False
		update_env(S, episode, step_counter)
		while not is_terminated:

			A = choose_action(S, q_table)
			S_, R = get_env_feedback(S, A)
			q_predict = q_table.ix[S, A]
			if S_ != 'terminal':
				q_target = R + Lambda* q_table.iloc[S_, :].max()
			else:
				q_target = R
				is_terminated = True

			q_table.ix[S, A] += Aloha * (q_target - q_predict) #update q_table
			S = S_     #move to next state

			update_env(S, episode, step_counter+1)
			step_counter += 1
	return q_table

if __name__ == "__main__":
	q_table = rl()
	print('\r\nQ_table:\n')
	print(q_table)


