import torch
import numpy as np
import matplotlib.pyplot as plt
import random

BOARD_ROWS = 5
BOARD_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1,0), (3,1), (4,2), (1,3)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False        

    def getReward(self):
        if self.state in HOLE_STATE:
            return -5
        elif self.state == WIN_STATE:
            return 1
        else:
            return -1

    def isEndFunc(self):
        if self.state == WIN_STATE or self.state in HOLE_STATE:
            self.isEnd = True

    def nxtPosition(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nxtState = (self.state[0] + moves[action][0], self.state[1] + moves[action][1])
        if 0 <= nxtState[0] < BOARD_ROWS and 0 <= nxtState[1] < BOARD_COLS:
            return nxtState
        return self.state

class Agent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.State = State()
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd
        self.rewards = 0
        self.plot_reward = []

        # Q-table with shape (rows, cols, actions)
        self.Q = torch.zeros((BOARD_ROWS, BOARD_COLS, len(self.actions)), device=device)

    def Action(self):
        i, j = self.State.state
        if random.random() > self.epsilon:
            action = torch.argmax(self.Q[i, j]).item()
        else:
            action = random.choice(self.actions)
        next_pos = self.State.nxtPosition(action)
        return next_pos, action

    def Q_Learning(self, episodes):
        for ep in range(episodes):
            self.State = State()
            self.isEnd = self.State.isEnd
            self.rewards = 0

            while not self.isEnd:
                i, j = self.State.state
                next_state, action = self.Action()
                reward = self.State.getReward()
                self.rewards += reward

                ni, nj = next_state
                max_next_q = torch.max(self.Q[ni, nj]).item()
                td_target = reward + self.gamma * max_next_q
                td_delta = td_target - self.Q[i, j, action].item()

                # Update Q value
                self.Q[i, j, action] += self.alpha * td_delta

                # Move to next state
                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd

            self.plot_reward.append(self.rewards)

    def plot(self):
        plt.plot(self.plot_reward)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title("Reward over Time")
        plt.grid()
        plt.show()

    def showValues(self):
        for i in range(BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                mx_nxt_value = torch.max(self.Q[i, j]).item()
                out += f"{mx_nxt_value:.2f}".ljust(8) + '| '
            print(out)
        print('-----------------------------------------------')

if __name__ == "__main__":
    agent = Agent()
    episodes = 10000
    agent.Q_Learning(episodes)
    agent.plot()
    agent.showValues()

