import time
import gym # need 17.3
import gym_maze
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#env = gym.make('maze-sample-5x5-v0')
env = gym.make('Taxi-v3')

action_n = 6
state_n = 500

class CrossEntopyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)
    
    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
                #print(f'new_model[state]: {new_model[state]}')
                print(f'new_model[state][action]: {new_model[state][action]}, action: {action}, state: {state}' )
            
            for state in range(self.state_n):
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = self.model[state].copy()

            self.model = new_model
            return None
            


def get_state(obs):
    print(obs)
    return obs

def get_trajectory(env, agent, max_len=1000, visualise=True):
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
    
agent = CrossEntopyAgent(state_n, action_n)
q_param = 0.9
iteration_n = 20
trajectory_n = 50

mean_rewards = []
elite_trajectory_counts = []

for iteration in range(iteration_n):
    #policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]

    #for trajectory in trajectories:
        #print(f"trajectory['rewards']: {trajectory['rewards']}")

    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))
    mean_reward = np.mean(total_rewards)
    mean_rewards.append(mean_reward)

    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    elite_trajectory_counts.append(len(elite_trajectories))
    agent.fit(elite_trajectories)

trajectory = get_trajectory(env, agent, max_len=100, visualise=True)
print('total reward:', sum(trajectory['rewards']))
print('model:')
print(agent.model)
              

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

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(range(iteration_n), mean_reward, label='Mean total reward', color='b')
plt.xlabel('Iteration')
plt.ylabel('Mean Total Reward')
plt.title('Mean Total Reward over Iterations')
plt.legend()

# График количества элитных траекторий
plt.subplot(1, 2, 2)
plt.plot(range(iteration_n), elite_trajectory_counts, label='Elite Trajectories', color='g')
plt.xlabel('Iteration')
plt.ylabel('Number of Elite Trajectories')
plt.title('Elite Trajectories over Iterations')
plt.legend()

plt.tight_layout()
plt.show()