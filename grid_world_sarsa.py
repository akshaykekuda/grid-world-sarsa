"""Implementing SARSA on a grid world"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

grid_row_len = 5
grid_col_len = 5
episode_len = 10000
action_list = ['u', 'd', 'r', 'l']
c = [0.1, 0.1, 0, 1, 0.1]
initial_state = (1, 13, 25)


def state_tuple_converter(state_tuple):
    state = grid_row_len * state_tuple[0] + state_tuple[1] + 1
    return state


def state_converter(state):
    q = (state - 1) // grid_row_len
    r = (state - 1) % grid_row_len
    return q, r


def take_action(action, state_tuple):
    if state_tuple == (0, 1):
        return 4, 1
    elif state_tuple == (0, 3):
        return 2, 3
    if action == 'u':
        return max(state_tuple[0] - 1, 0), state_tuple[1]
    if action == 'd':
        return min(state_tuple[0] + 1, grid_row_len - 1), state_tuple[1]
    if action == 'l':
        return state_tuple[0], max(state_tuple[1] - 1, 0)
    if action == 'r':
        return state_tuple[0], min(state_tuple[1] + 1, grid_col_len - 1)


def phi(state_tuple, action):
    arr = np.zeros([grid_col_len * grid_row_len * 4, 1])
    ind = action_list.index(action)
    arr[ind * 25 + state_tuple_converter(state_tuple) - 1] = 1
    return arr


def generate_episode(c, i):
    e = 1
    do_error = np.zeros(episode_len)
    weight = np.zeros([grid_row_len * grid_col_len * 4, episode_len])
    avg_reward = np.zeros(episode_len)
    for y in range(10):
        epsilon = e *(1 / math.ceil((y + 1) / 3))
        start_state = np.random.choice([18])
        curr_state_tuple = state_converter(start_state)
        curr_action = np.random.choice(action_list, p=[0.25, 0.25, 0.25, 0.25])
        for t in range(0, episode_len - 1):
            alpha = 1 / (math.ceil((t + 1) / 10))
            beta = c * alpha
            next_state_tuple = take_action(curr_action, curr_state_tuple)

            if curr_state_tuple == (0, 1):
                reward = 10
            elif curr_state_tuple == (0, 3):
                reward = 5
            elif next_state_tuple == curr_state_tuple:
                reward = -1
            else:
                reward = 0

            random_e = random.uniform(0, 1)
            if random_e < epsilon:
                exploring_action = np.random.choice(action_list, p=[0.25, 0.25, 0.25, 0.25])
                next_action = exploring_action
                # print("random action" + exploring_action)
            else:
                # print("greedy action")
                q_next_u = np.dot(weight[:, t], phi(next_state_tuple, 'u'))
                q_next_d = np.dot(weight[:, t], phi(next_state_tuple, 'd'))
                q_next_r = np.dot(weight[:, t], phi(next_state_tuple, 'r'))
                q_next_l = np.dot(weight[:, t], phi(next_state_tuple, 'l'))
                q_next = [q_next_u, q_next_d, q_next_r, q_next_l]
                # print(q_next)
                q_max = max(q_next)
                # print(q_max)
                q_max_index = [i for i, j in enumerate(q_next) if j == q_max]
                rand_greedy_q = np.random.choice(q_max_index)
                # print("greddy action:" + action_list[rand_greedy_q])
                next_action = action_list[rand_greedy_q]

            do_error[t + 1] = reward - avg_reward[t] + np.dot(weight[:, t],
                                                              phi(next_state_tuple, next_action)) - np.dot(weight[:, t],
                                                                                                           phi(
                                                                                                               curr_state_tuple,
                                                                                                               curr_action))

            avg_reward[t + 1] = avg_reward[t] + beta * do_error[t + 1]
            weight[:, t + 1] = weight[:, t] + alpha * do_error[t + 1] * np.transpose(phi(curr_state_tuple, curr_action))
            curr_state_tuple = next_state_tuple
            curr_action = next_action

        # print(np.around(np.reshape(weight[:, episode_len - 1], (5, 5, 4)), decimals=3))

        q_all = weight[:, episode_len - 1]
        q_max_val = []
        q_max_action = []
        q_final = []
        for x in range(1, 26):
            temp_u = np.dot(q_all, phi(state_converter(x), 'u'))
            temp_d = np.dot(q_all, phi(state_converter(x), 'd'))
            temp_r = np.dot(q_all, phi(state_converter(x), 'r'))
            temp_l = np.dot(q_all, phi(state_converter(x), 'l'))
            temp = [temp_u, temp_d, temp_r, temp_l]
            q_final.append(temp)
            q_max = max(temp)
            q_max_index = [action_list[i] for i, j in enumerate(temp) if j == q_max]
            q_max_val.append(q_max)
            q_max_action.append(q_max_index)

        # print(np.around(np.reshape(q_final, (5, 5, 4)), decimals=2))
        print("start state = " + str(start_state) + " epsilon=" + str(epsilon) + " run = " + str(y + 1))
        print(np.around(np.reshape(q_max_val, (5, 5)), decimals=2))
        print(np.reshape(q_max_action, (5, 5)))
        weight[:, 0] = q_all
        avg_reward[0] = avg_reward[episode_len - 1]


for i in range(1):
    generate_episode.counter = 1
    generate_episode.state_visited = []
    generate_episode(0.1, 1)

    # start_state = 18
    # start_state_tuple = state_converter(start_state)
    # generate_episode.state_visited = []
    # generate_episode(start_state_tuple, c[i], i)

    # start_state = 18
    # start_state_tuple = state_converter(start_state)
    # generate_episode.state_visited = []
    # generate_episode(start_state_tuple, c[i], i)

plt.show()
