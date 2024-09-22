import time
import gym # need 17.3
import gym_maze
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


env = gym.make('maze-sample-5x5-v0')
#print(env)
action_n = 4
state_n = 25

class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state):
        return np.random.randint(self.action_n)
    
def get_state(obs):
    return 5 * obs[0] + obs[1]
    
agent = RandomAgent(action_n)

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
                print(f'new_model[state]: {new_model[state]}')
                print(f'new_model[state][action]: {new_model[state][action]}, action: {action}, state: {state}' )
            
            for state in range(self.state_n):
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = self.model[state].copy()

            self.model = new_model
            return None
            


def get_state(obs):
    return int(np.sqrt(state_n) * obs[0] + obs[1])

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
    
agent = CrossEntopyAgent(state_n, action_n)
q_param = 0.9
iteration_n = 20
trajectory_n = 50

for iteration in range(iteration_n):
    #policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]

    for trajectory in trajectories:
        print(f"trajectory['rewards']: {trajectory['rewards']}")

    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

trajectory = get_trajectory(env, agent, max_len=100, visualise=True)
print('total reward:', sum(trajectory['rewards']))
print('model:')
print(agent.model)
              

            






    



#print(state)
#env.render() # visualisation

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


