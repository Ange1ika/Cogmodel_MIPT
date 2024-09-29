# -*- coding: utf-8 -*-
"""Kochetkova_practice_1_task2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ThOJmTxCcPdmAGyF2yhaE_aI6py6HKWb
"""

import time
import gym # need 17.3
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#env = gym.make('maze-sample-5x5-v0')
env = gym.make('Taxi-v3')

action_n = 6
state_n = 500

class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit_laplace(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        # Laplace smoothing
        alpha = 5
        for state in range(self.state_n):
            new_model[state] = (new_model[state] + alpha) / (np.sum(new_model[state]) + alpha * self.action_n)

        self.model = new_model

    def fit_policy_smoothing(self, elite_trajectories, lambda_):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        # Нормализация новой модели
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        # Применение сглаживания политики
        self.model = lambda_ * new_model + (1 - lambda_) * self.model



def get_state(obs):
    #print(obs)
    return obs

def get_trajectory(env, agent, max_len=1000, visualise=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):

        #state = get_state(obs)
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = get_state(obs)

        if visualise:
            time.sleep(0.5)
            env.render()

        if done:
            break
    return trajectory

def sample_trajectory(env, agent, trajectory_len=1000, visualization=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    obs = env.reset()
        # zero point (0, 0)
        # let's do the action (5 steps)
    for _ in range(trajectory_len):
        state = get_state(obs)
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['action'].append(action)
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        if done:
            break

        if visualization:
            env.render()
            time.sleep(0.2)

    return trajectory

q_param = 0.9
# Initialize agent
agent_laplace = CrossEntropyAgent(state_n, action_n)
agent_policy_smoothing = CrossEntropyAgent(state_n, action_n)

lambda_ = 0.1  # For policy smoothing
iteration_n = 20
trajectory_n = 200

mean_rewards_laplace = []
mean_rewards_policy_smoothing = []

for iteration in range(iteration_n):
    # Laplace Smoothing
    trajectories = [get_trajectory(env, agent_laplace) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    elite_trajectories = [trajectory for trajectory in trajectories if np.sum(trajectory['rewards']) > np.quantile(total_rewards, q_param)]
    mean_rewards_laplace.append(np.mean(total_rewards))
    agent_laplace.fit_laplace(elite_trajectories)

    # Policy Smoothing
    trajectories = [get_trajectory(env, agent_policy_smoothing) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    elite_trajectories = [trajectory for trajectory in trajectories if np.sum(trajectory['rewards']) > np.quantile(total_rewards, q_param)]
    mean_rewards_policy_smoothing.append(np.mean(total_rewards))
    agent_policy_smoothing.fit_policy_smoothing(elite_trajectories, lambda_)

# Plotting the comparison
plt.figure(figsize=(12, 5))

# Mean rewards for Laplace
plt.subplot(1, 2, 1)
plt.plot(range(iteration_n), mean_rewards_laplace, label='Laplace Smoothing', color='b')
plt.plot(range(iteration_n), mean_rewards_policy_smoothing, label='Policy Smoothing', color='g')
plt.xlabel('Iteration')
plt.ylabel('Mean Total Reward')
plt.title('Mean Total Reward Comparison')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

!pip install time
!pip install gym==17.3
!pip install numpy
!pip install matplotlib