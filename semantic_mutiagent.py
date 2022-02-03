"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import numpy as np
import os
import random
import argparse
import pandas as pd
# from environments.agents_landmarks.env import agentslandmarks
from environments.semantic.env import semantic
from dqn_agent import Agent
import glob

ARG_LIST = ['name','learning_rate', 'memory_capacity', 'batch_size', 'max_timestep', 'target_type', 'prioritization_scale', 'agents_number']


def get_name_brain(args, idx):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_semantic/weights_files/' + file_name_str + '_' + str(idx) + '.h5'


def get_name_rewards(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_semantic/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_semantic/timesteps_files/' + file_name_str + '.csv'


class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = semantic(arguments)
        self.episodes_number = arguments['episode_number']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']

        self.num_agents = arguments['agents_number']
        self.num_RBs = arguments['RBs_number']

    def run(self, agents, file1, file2):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000
        for episode_num in xrange(self.episodes_number):
            state = self.env.reset()

            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()

            reward_all = 0
            time_step = 0
            while time_step < self.max_ts:

                # if self.render:
                #     self.env.render()
                actions = []
                for agent in agents:
                    actions.append(agent.greedy_actor(state))
                next_state, reward = self.env.step(actions,time_step)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                if not self.test:
                    for agent in agents:
                        agent.observe((state, actions, reward, next_state))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()

                total_step += 1
                time_step += 1
                state = next_state
                reward_all += reward

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print("Episode {p}, Score: {s}, Final Step: {t}".format(p=episode_num, s=reward_all,
                                                                               t=time_step))

            if not self.test:
                if episode_num % 10 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)

                    if total_step >= self.filling_steps:
                        if reward_all > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_all


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-name', '--name', help='Name')
    parser.add_argument('-e', '--episode-number', default=5, type=int, help='Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=100, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=50, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.3, type=float, help='Scale for prioritization')
    parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture if "store_false" ')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=10, type=int, help='The number of agents')
    parser.add_argument('-rb', '--RBs-number', default=5, type=int, help='The number of RBs')
    parser.add_argument('-ts', '--max-timestep', default=5, type=int, help='Maximum number of timesteps per episode')

    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']

    env = Environment(args)

    state_size = env.env.state_size
    action_space = env.env.action_space()

    all_agents = []
    for b_idx in xrange(args['agents_number']):

        brain_file = get_name_brain(args, b_idx)
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))

    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)

    env.run(all_agents, rewards_file, timesteps_file)
