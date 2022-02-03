import random
import numpy as np
import pygame


class semantic:

    def __init__(self, args):

        self.max_len = 0
        self.attention = []
        for line in open("./environments/semantic/attention.txt"):
            a = np.array(line.split(',\n')[0].split(','))
            a = map(float, a)
            if len(a) > self.max_len:
                self.max_len = len(a)
            self.attention.append(a)
        self.similarity = []
        for line in open("./environments/semantic/sm_parts.txt"):
            self.sm = np.array(line.split(',\n')[0].split(','))
            self.sm = map(float, self.sm)
            self.sm = [(10 * i)**3 for i in self.sm]
            self.similarity.append(self.sm)

        self.num_agents = args['agents_number']
        self.num_RBs = args['RBs_number']
        # self.rates = [0.5, 0.7, 0.9 ,1.1, 1.3, 1.5, 1.7, 1.9]
        self.rates = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
        # self.rates = [0.5, 0.9, 1.3, 1.7]
        self.P = 0.5
        self.E = []
        self.aton = []

        self.state_size = self.num_agents * (self.max_len + 1)

        self.A = [0, 6, 8, 10, 12, 14, 16, 18, 20, 22]

        self.num_episodes = 0

    def reset(self):  # initialize the world

        initial_state = []
        self.E = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        for i in range(self.num_agents):
            initial_state.append(self.E[i])

        for i in range(self.num_agents):
            for j in range(self.max_len):
                if j < len(self.attention[i]):
                    initial_state.append(self.attention[i][j])
                else:
                    initial_state.append(0.0)

        return initial_state

    def step(self, agents_actions, time_step):
        print(self.E[3])
        print(agents_actions[3])
        print('total_triple', len(self.similarity[3 + (time_step) * self.num_agents]) - 1)
        RB_st = 0
        reward = 0
        for i in range(self.num_agents):
            if self.A[agents_actions[i]] != 0:
                RB_st += 1
            if RB_st <= self.num_RBs :
                triple_len = len(self.similarity[i + (time_step) * self.num_agents]) - 1
                if triple_len >= self.A[agents_actions[i]]:
                    temp = self.E[i] - self.P * (self.A[agents_actions[i]] / self.rates[i])
                    if temp >= 0:
                        self.E[i] = temp
                        reward += self.similarity[i + (time_step) * self.num_agents][triple_len - agents_actions[i]]
                    else:
                        reward += 0
                else:
                    temp = self.E[i] - self.P * (triple_len / self.rates[i])
                    if temp >= 0:
                        self.E[i] = temp
                        reward += self.similarity[i + (time_step) * self.num_agents][0]
                    else:
                        reward += 0
            else:
                reward += 0
        print('letover',self.E[3])
        new_state = []
        for i in range(self.num_agents):
            self.E[i] += random.uniform(0,5)
            # self.E[i] += 3
            new_state.append(self.E[i])

        for i in range(self.num_agents):
            for j in range(self.max_len):
                if j < len(self.attention[i + (time_step + 1) * self.num_agents]):
                    new_state.append(self.attention[i + (time_step + 1) * self.num_agents][j])
                else:
                    new_state.append(0.0)

        return [new_state, reward]

    def action_space(self):
        return len(self.A)

    def render(self):
        if not self.terminal:
            self.step_num += 1

    def gui_setup(self):

        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        screen = pygame.display.set_mode((1, 1))

        myfont = pygame.font.SysFont("monospace", 30)

        return [screen, myfont]

    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq
