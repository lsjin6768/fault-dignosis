import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import matplotlib
import matplotlib.pyplot as plt
import random
import math
​
learning_rate = 0.0001
gamma = 0.98
buffer_limit = 10000
batch_size = 32
​
class ReplayBuffer():  # replay buffer
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
​
    def put(self, transition):
        self.buffer.append(transition)
​
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst),dtype=torch.int64), torch.tensor(np.array(r_lst),dtype=torch.int64),torch.tensor(np.array(s_prime_lst),dtype=torch.float)
​
    def size(self):
        return len(self.buffer)
​
class Qnet(nn.Module):  # Q-network
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(56, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 21)
        self.relu = torch.nn.ReLU()
​
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
​
    def sample_action(self, s, epsilon, max_Battery, SOC, check):
        out = self.forward(s)
        coin = random.random()
        if SOC > max_Battery: SOC = max_Battery
        q_list ,idx = [], []
        for i in range(0,21,1):
            if SOC + i>= 0 and SOC + i <= max_Battery:
                q_list.append(out[i].item())
                idx.append(i)
            else:
                q_list.append(np.float('inf'))
​
        if check == 1:
            print(q_list)
            return
​
        if (coin < epsilon):
            a = random.choice(idx)
            return a
        else:
            return q_list.index(min(q_list))
​
def train(q, q_target, memory, optimizer,max_Battery):
    s, a, r, s_prime = memory.sample(batch_size)
    q_a = q(s).gather(1,a)
    print(q_a)
    min_q_prime = q_target(s_prime)
​
    Q_target = list()
​
    for i in range(batch_size):
        SOC = s_prime[i][0].item() - s_prime[i][1].item() + s_prime[i][4].item()
        if SOC > max_Battery: SOC = max_Battery
        q_list = []
        for j in range(0, 21, 1):
            if SOC + j >= 0 and SOC + j <= max_Battery:
                q_list.append(min_q_prime[i][j].item())
            else:
                q_list.append(np.float('inf'))
        Q_target.append(min(q_list))
​
    Q_target = torch.tensor((Q_target)).resize(32,1)
    target = r + gamma * Q_target
    loss = F.smooth_l1_loss(q_a, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
​
def check_result(TOU,L,G):
    greedy_price = 0
    for i in range(0, 24):
        gap = L[i] - G[i]
        if gap < 0: gap = 0
        greedy_price = greedy_price + TOU[i] * gap
    print('greedy price : {}'.format(greedy_price + max(L)*30))
​
    DQN_price = 0
    show2 = np.zeros(epochs)
    for i in range(epochs):
        DQN_price = DQN_price + price_dqn[i]
        show2[i] =  DQN_price / (i + 1)
    print('DQN price : {}'.format(show2[epochs - 1]))
​
    temp = sum(price_dqn[epochs - 100:epochs - 1])
    print(temp/100)
    print(price_dqn[epochs - 100:epochs - 1])
    print(min(price_dqn[epochs - 1000:epochs - 1]))
​
    # price graph
    x = range(0, epochs)
    y1 = [show2[v] for v in x] #dqn
    plt.plot(x,y1,label='dqn_price', color='r')
    plt.legend()
    plt.show()
​
    # action graph
    for i in range(10):
        x = range(0, 24)
        y1 = [v for v in L[0:24]]  # Load
        y2 = [v for v in actions_dqn[epochs - 10 + i]]  # Charge
        y3 = [v for v in TOU[0:24]]  # TOU price
        y4 = [v for v in G[0:24]]
        y5 = [v for v in batterys_dqn[epochs-10+i]]
​
        plt.figure(figsize=(10, 8))
        plt.plot(x, y3, label='TOU', color='gray')
        plt.fill_between(x[0:24], y3[0:24], color='lightgray')
        plt.bar(x, y1, label='Load', color='lightgreen')
        plt.plot(x, y2, label='Charge', color='r')
        plt.plot(x, y4, label='generation', color='b')
        plt.plot(x, y5, label='battery', color='y')
        plt.legend()
        plt.xticks(np.arange(0, 24))
        plt.yticks(np.arange(0, 50))
        plt.grid(True)
        plt.show()
​
​
​
# Load
L = np.zeros(24)
L = [ 9 , 8 , 7 , 7 , 7 , 7 , 8 , 9 ,10 ,10 , 10 , 10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 13, 12, 11] #7월
#L = [10,  8,  8,  8,  7,  8,  8, 10, 10, 10, 10, 10, 10, 10,  9,  9, 10, 11, 13, 13, 13, 13, 12, 11] #1월
TOU = np.zeros(24)
TOU = [5,5,5, 5,5,5, 5,10,10 ,10,10,15, 15,15,15, 15,15,10, 10,5,5,5,5,5]  #여름
#TOU = [5,5,5, 5,5,5, 5,15,15, 15,25,10, 10,10,10, 10,10,15, 15,5,5, 5,5,5]  # 겨울
​
# Generation
G = np.zeros(24)
#G = [0,0,0, 0,0,0, 1,2,4, 6,8,10, 12,12,10, 8,6,4, 2,1,0, 0,0,0] #여름
G = [0,0,0, 0,0,0, 0,1,3, 6,8,10, 10,8,6, 3,1,0 ,0,0,0, 0,0,0] #겨울
​
# time
T = np.zeros(24)
T = np.array(pd.get_dummies(np.array([0,1,2,3,4,5,6,7,8,9,10,11,
                                      12,13,14,15,16,17,18,19,20,21,22,23])))
​
# MAIN
q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
​
actions_dqn = list()
action_dqn = list()
price_dqn = list()
battery_dqn = list()
batterys_dqn = list()
c_r = 0
​
epochs = 100000
max_Battery = 60
​
a,SOC=0,0
for n_epi in range(epochs):
​
    S = np.zeros(56)
    S[0] = SOC + a
    S[1] = L[0]
    S[2] = L[1]
    S[3] = L[2]
    state=[0-3][0-15][5]
    action=[0-100][15]

​
    S[4] = G[0]
    S[5] = G[1]
    S[6] = G[2]
​
    for i in range(24):
        S[7+i] = TOU[i]
    for i in range(24):
        S[31 + i] = T[0][i]
    S[55] = 0  # max(action)
​
    c_r = 0
​
    for time in range(0, 24):
​
        # EPSILON
        epsilon = max(0, 1 - ((1 / epochs) * n_epi))
​
        # ACTION
        SOC = S[0] - S[1] + S[4]
        a = q.sample_action(torch.from_numpy(S).float(), epsilon,max_Battery,SOC,0)
        action_dqn.append(a)
        battery_dqn.append(S[0])
​
        # reward
        r = a * TOU[time]
​
        if time == 23:
            r = r + max(action_dqn) * 30
        c_r = c_r + r
​
        # next State
        S_P = np.zeros(56)
        S_P[0] = SOC + a
        S_P[1] = L[(time+1)%24]
        S_P[2] = L[(time+2)%24]
        S_P[3] = L[(time+3)%24]
​
        S_P[4] = G[(time+1)%24]
        S_P[5] = G[(time+2)%24]
        S_P[6] = G[(time+3)%24]
​
        for i in range(24):
            S_P[7 + i] = TOU[(time+1+i)%24]
        for i in range(24):
            S_P[31 + i] = T[(time + 1) % 24][i]
        S_P[55] = max(action_dqn)  # max(action)
​
        # Save transition
        memory.put((S, a, r, S_P))
        S = S_P
​
        if n_epi % 1000 == 0 or n_epi>epochs-10:
            print('time:{} n_epi:{}  Epsilon:{}'.format(time,n_epi,epsilon))
            print(q_target.sample_action(torch.from_numpy(S).float(),epsilon,max_Battery,SOC,1))
​
​
    price_dqn.append(c_r)
    actions_dqn.append(action_dqn)
    batterys_dqn.append(battery_dqn)
    action_dqn = list()
    battery_dqn = list()
​
    # Start update
    if memory.size() > 500:
        train(q, q_target, memory, optimizer,max_Battery)
​
    # Update Q-target
    if n_epi % 20 == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())
​
check_result(TOU,L,G)