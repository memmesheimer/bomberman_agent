
import numpy as np
from time import sleep
from settings import e 
from time import sleep
from settings import e 
import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, alpha):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        
        self.fc1 = nn.Linear(972, 512)
        self.fc2 = nn.Linear(512, 6)
        self.optimizer = optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')     
        self.to(self.device)

    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
 
        observation = observation.view(-1, 1, 17, 17)      
        observation = F.relu(self.conv1(observation))     
        observation = F.relu(self.conv2(observation))       

        observation = observation.view(-1, 972)    
        observation = F.relu(self.fc1(observation))  
        actions = self.fc2(observation)
        
        return actions
    
class Agent(object):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, minEps = 0.05, actionSpace = [0,1,2,3,4,5]):
        self.gamma = gamma
        self.epsilon = epsilon
        self.minEps = minEps
        self.alpha = alpha
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCounter = 0
        self.Q_eval = Net(alpha)
        self.Q_next = Net(alpha)
        
    def storeTransition(self, state, action, reward, next_state):
        
        if self.memCounter < self.memSize:
            self.memory.append([state, action, reward, next_state])
        else:            
            self.memory[self.memCounter%self.memSize] = [state, action, reward, next_state]
        
        self.memCounter += 1
    
    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        
        if rand < 1 - self.epsilon:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionSpace)            
        self.steps += 1
        return action
    
    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        #getting a random batch of size 'batch_size' of memory
        if self.memCounter+batch_size < self.memSize:            
            memStart = int(np.random.choice(range(self.memCounter)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch=self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)
        
        Qpred = self.Q_eval.forward(list(memory[:,0])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:,3])).to(self.Q_eval.device) 
        maxA = T.argmax(Qnext, dim= 1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)
        
        Qtarget = Qpred
        Qtarget[:, maxA] = rewards + self.gamma * T.max(Qnext[1])
        
        if self.steps > 50000:
            if self.epsilon -1e-6 > self.minEps:
                self.epsilon -= 1e-6
            else:
                self.epsilon = self.min_eps
                
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

class MemoryInitialization(object):
    def __init__(self):
        self.integer = 0
        self.step = 0
        self.firstState = None
        self.nextState = None
        self.lastAction = None
        
        self.coins = 0
        self.blewUp = 0
        self.score= []
        self.score_plot = []
        
        self.numGames = []
        self.numGames_plot = []
        self.games = 0
        
        self.average_score = []
        
        self.epsilon = []
        self.i = 0
        
        self.reward = 0
        self.rewards = []
        self.training = False
        
    def coin_collected(self):
        self.coins += 1
        
    def game_played(self):
        self.games += 1
        self.numGames.append(self.games)
        self.numGames_plot.append(self.games)
    
    def append_score(self):
        self.score.append(self.coins + (self.blewUp * 5))
        self.score_plot.append(self.coins)
        self.coins = 0
    
    def average_plot(self):
        if len(self.numGames_plot) % 10 == 0:
            average_score = 0
            for s in self.score_plot:
                average_score += s
            average_score /= 10
            self.average_score.append(average_score)
            self.numGames_plot = self.numGames
            i = int(len(self.numGames)/10)
            self.numGames_plot = self.numGames_plot[:i]
            
            plt.plot(len(self.numGames_plot), self.average_score)
            plt.xlabel('number of Games / 10 played')
            plt.ylabel('score average achieved')
            plt.savefig('score2.png')
            plt.close()
            self.score_plot = []
            
def setup(self):
    
    self.bomberman = Agent(gamma = 0.9, epsilon = 1.0, alpha = 0.003, maxMemorySize = 10000)

    self.helper = MemoryInitialization()
    
''' The act function is called before every step so the agent can decide, what to do'''
def act(self):
    actionSpace = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
    
    # Gather information about the game state
    arena = self.game_state['arena'] #2d numpy array: 1: crate; 0: wall, -1:free tile
    x, y, _, bombs_left, score = self.game_state['self'] #x-position, y-position, name, bombs available (yes,no), score
    self.logger.info(f'state is: {x,y}')
    bombs = self.game_state['bombs'] #x-position,y position  for all bombs and how much time is left for all bombs
    bomb_xys = [(x,y) for (x,y,t) in bombs] #list of x and y values  
    others = self.game_state['others'] #list of tuples (x,y,_,bombs_left) for other players
    others_xy = [(x,y) for (x,y,n,b,s) in self.game_state['others']] #x and y values of other players  
    coins = self.game_state['coins'] #list of all currently available coins and their coordinates
    
    arena1 = np.copy(arena)
 
    #coin positions
    for coin in coins:
        arena1[coin[1]][coin[0]] = 9
  
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    #explosions  
    rows, cols = np.where(bomb_map == 4)
    for indice in rows:
        arena1[rows, cols] = 8
    rows, cols = np.where(bomb_map == 3)
    for indice in rows:
        arena1[rows, cols] = 7
    rows, cols = np.where(bomb_map == 2)
    for indice in rows:
        arena1[rows, cols] = 6
    rows, cols = np.where(bomb_map == 1)
    for indice in rows:
        arena1[rows, cols] = 5
    for every in others_xy:
        arena1[every[0], every[1]] = 15
    rows, cols = np.where(bomb_map == 0)
    for indice in rows:
        arena1[rows, cols] = 4
    
    #agents positon
    arena1[x,y] = 10
    
    #overwrite the arena so that everywhere where indestructible walls were, they are again there
    rows, cols = np.where(arena == -1)
    for indice in rows:
        arena1[rows, cols] = -1
        
    self.helper.firstState = np.copy(arena1)
    
    #if this flag is set, the agent will initialize memory and is in training mode
    if self.helper.training == True:
        if self.bomberman.memCounter < self.bomberman.memSize:
                self.helper.integer = 1
                self.next_action = np.random.choice(actionSpace)
                self.helper.lastAction = self.next_action
                if self.helper.lastAction == 'UP':    
                    self.helper.xy = [y-1,x]
                    
            
                elif self.helper.lastAction == 'DOWN':
                    self.helper.xy = [y+1,x]
                    
                elif self.helper.lastAction == 'LEFT':
                    self.helper.xy = [y,x-1]
                    
                elif self.helper.lastAction == 'RIGHT':
                    self.helper.xy = [y,x+1]
                   
                elif self.helper.lastAction == 'WAIT':
                    self.helper.xy = [y,x]
                elif self.helper.lastAction == 'BOMB':
                    self.helper.xy = [y,x]
                    
                return self.next_action
        
        #this flag helps in order to only learn, AFTER initializing the memory
        self.helper.integer = 0
        action = self.bomberman.chooseAction(self.helper.firstState)
        self.next_action = actionSpace[action]
        self.helper.lastAction = self.next_action
        return self.next_action
    
    #if not in training mode, then the pretrained network is loaded in
    else:
        bomberman = Agent(gamma = 0.9, epsilon = 0, alpha = 0.003, maxMemorySize = 10000)
        bomberman.Q_eval.load_state_dict(T.load(os.path.abspath("agent_code/user_agent/neural_network.pt")))
        print("in here")
        action = self.bomberman.chooseAction(self.helper.firstState)
        self.next_action = actionSpace[action]
        self.helper.lastAction = self.next_action
        return self.next_action

    
def reward_update(self):
    others3 = self.game_state['others'] #list of tuples (x,y,_,bombs_left) for other players
    others_xy3 = [(x,y) for (x,y,n,b,s) in self.game_state['others']] #x and y values of other players
    arena3 = self.game_state['arena']
    x2, y2, _, bombs_left, score = self.game_state['self']
    bombs3 = self.game_state['bombs']
    bomb_map3 = np.ones(arena3.shape) * 5
    coins3 = self.game_state['coins']
    
    arena2 = np.copy(arena3)
    
    for coin in coins3:
        arena2[coin[1]][coin[0]] = 9
    for xb,yb,t in bombs3:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map3.shape[0]) and (0 < j < bomb_map3.shape[1]):
                bomb_map3[i,j] = min(bomb_map3[i,j], t)
    
    rows, cols = np.where(bomb_map3 == 4)
    for indice in rows:
        arena2[rows, cols] = 8
    
    rows, cols = np.where(bomb_map3 == 3)
    for indice in rows:
        arena2[rows, cols] = 7
    
    rows, cols = np.where(bomb_map3 == 2)
    for indice in rows:
        arena2[rows, cols] = 6
    
    rows, cols = np.where(bomb_map3 == 1)
    for indice in rows:
        arena2[rows, cols] = 5
            
    rows, cols = np.where(bomb_map3 == 0)
    for indice in rows:
        arena2[rows, cols] = 4
    
    arena2[x2,y2] = 10
    
    for every in others_xy3:
        arena2[every[0], every[1]] = 15
        
    rows, cols = np.where(arena3 == -1)
    for indice in rows:
        arena2[rows, cols] = -1

    #rewards     
    reward = -1
    if e.COIN_COLLECTED in self.events:
        reward +=20
        self.helper.coin_collected()
        self.logger.info("Coin Collected")
    if e.INVALID_ACTION in self.events:
        reward -= 1
        self.helper.nextState = self.helper.firstState
        self.logger.info("in invalid action")
    if e.WAITED in self.events:
        reward -= 1
    if e.CRATE_DESTROYED in self.events:
        reward += 1
    if e.BOMB_DROPPED in self.events:
        reward += 1
    if e.KILLED_OPPONENT in self.events:
        reward += 100
        self.helper.blewUp += 1
        self.logger.info("Opponent_Eliminated")
    if e.KILLED_SELF in self.events:
        reward -= 125
        self.logger.info("Killed self")
        self.helper.gotKilled = True
    if e.GOT_KILLED in self.events:
        reward -= 75
        self.logger.info("got Killed")
        self.helper.gotKilled = True
    if e.SURVIVED_ROUND in self.events:
        reward += 20
        self.logger.info("Survived Round")
    self.helper.nextState = np.copy(arena2)
    action = self.helper.lastAction
    self.logger.info(f"reward {reward}")
    
    #this is the flag that was set in the act function beforehand
    #this is basically a distinction between memory initialization and then actual batch learning
    if self.helper.integer == 1:                        
        self.bomberman.storeTransition(self.helper.firstState, action, reward, self.helper.nextState)
    else:
        self.bomberman.storeTransition(self.helper.firstState, action, reward, self.helper.nextState)
        self.bomberman.learn(30)
    
    #used for creating plots
    self.helper.reward += reward
    self.helper.blewUp = 0
    return reward


def end_of_episode(self):
    """
    Called at the end of each game to hand out final rewards and do training.
    
    This is similar to reward_update, exept it is only called at the end of a game.
    self.events will contain all events that occured during your agent's final step.
    
    """
    self.logger.info(f'Epsilon: {self.bomberman.epsilon}') 
    
    self.helper.game_played()
    self.helper.append_score()
    self.helper.rewards.append(self.helper.reward)
    
    self.logger.info(f'The agent has following score as of right now {self.helper.score}')
    self.logger.info(f'The agent has following rewards as of right now {self.helper.rewards}')
    
    T.save(self.bomberman.Q_eval.state_dict(), os.path.abspath("agent_code/user_agent/neural_network.pt"))
    
    
    plt.plot(self.helper.numGames, self.helper.score)
    plt.xlabel('number of Games played')
    plt.ylabel('score achieved')
    plt.savefig('score.png')
    plt.close()
    
    plt.plot(self.helper.numGames, self.helper.rewards)
    plt.xlabel('number of Games played')
    plt.ylabel('rewards achieved')
    plt.savefig('score2.png')
    plt.close()
    
    self.helper.epsilon.append(self.bomberman.epsilon)
    plt.plot(self.helper.epsilon, self.helper.score)
    plt.xlabel('Epsilon')
    plt.ylabel('score achieved')
    plt.savefig('score3.png')
    plt.close()
    print("game played")
    self.helper.reward = 0
    