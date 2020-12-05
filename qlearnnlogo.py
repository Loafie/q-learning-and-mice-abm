import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# parameter_count =  sum(p.numel() for p in model.parameters() if p.requires_grad)

class DeepQNetwork(nn.Module):

    def __init__(self,lr, input_dims, fc1_dims, fc2_dims, n_actions, layers):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.layers = layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        if self.layers > 2:
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        if self.layers > 3:
            self.fc3 = nn.Linear(self.fc2_dims, self.fc2_dims)
        if self.layers > 4:
            self.fc4 = nn.Linear(self.fc2_dims, self.fc2_dims)
        if self.layers > 2:
            self.fc5 = nn.Linear(self.fc2_dims, self.n_actions)
        else:
            self.fc5 = nn.Linear(self.fc1_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        if self.layers > 2:
            x = F.relu(self.fc2(x))
        if self.layers > 3:
            x = F.relu(self.fc3(x))
        if self.layers > 4:            
            x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return actions

class AgentNormalBatch(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, layers=4, fc1_dim = 64, fc2_dim = 64, max_mem_size=1000000, eps_min=0.01, eps_dec=0.999996):
        self.mem_cntr = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dim, fc2_dim, n_actions, layers)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.actions_memory[index] = action
        self.terminal_memory[index] = True if (terminal == 0) else False
        self.mem_cntr += 1

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.Q_eval.device)
            with T.no_grad():
                actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype = np.int32)
            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
            action_batch = self.actions_memory[batch]
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

class AgentOneShot(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, layers=4, fc1_dim = 64, fc2_dim = 64, max_mem_size=1000000, eps_min=0.01, eps_dec=0.999996):
        self.mem_cntr = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dim, fc2_dim, n_actions, layers)


    def store_transition(self, state, action, reward, state_, terminal):
        self.state = np.array(state, dtype=np.float32)
        self.new_state= np.array(state_, dtype=np.float32)
        self.reward = np.array(reward, dtype=np.float32)
        self.actions = np.array(action, dtype=np.int32)
        self.terminal = np.array(True if (terminal == 0) else False, dtype=np.bool)

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.Q_eval.device)
            with T.no_grad():
                actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        return action

    def learn(self):
            self.Q_eval.optimizer.zero_grad()
            the_state = T.tensor(self.state).to(self.Q_eval.device)
            the_new_state = T.tensor(self.new_state).to(self.Q_eval.device)
            the_reward = T.tensor(self.reward).to(self.Q_eval.device)
            the_terminal = T.tensor(self.terminal).to(self.Q_eval.device)
            the_action = self.actions
            q_eval = self.Q_eval.forward(the_state)[the_action]
            q_next = self.Q_eval.forward(the_new_state)
            q_next = T.tensor(np.array(0.0, dtype=np.float32)).to(self.Q_eval.device) if the_terminal.item() else q_next
            q_target = the_reward + self.gamma * T.max(q_next, dim=0)[0]
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

class AgentOneShotRewardProportional(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, layers=4, fc1_dim = 64, fc2_dim = 64, max_mem_size=1000000, eps_min=0.01, eps_dec=0.999996):
        self.mem_cntr = 0
        self.obs_cntr = 1.0
        self.nz_reward_cnt = 0.0
        self.learn_next = False
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dim, fc2_dim, n_actions, layers)


    def store_transition(self, state, action, reward, state_, terminal):
        self.learn_next = False
        rand = np.random.random()
        if reward != 0.0 and rand < (1 - (self.nz_reward_cnt / self.obs_cntr)):
            self.learn_next = True
            self.nz_reward_cnt = self.nz_reward_cnt + 1.0
            self.obs_cntr = self.obs_cntr + 1.0
        
        if reward == 0.0 and rand < (self.nz_reward_cnt / self.obs_cntr):
            self.learn_next = True
            self.obs_cntr = self.obs_cntr + 1.0
        
        if self.learn_next == True:
            self.state = np.array(state, dtype=np.float32)
            self.new_state= np.array(state_, dtype=np.float32)
            self.reward = np.array(reward, dtype=np.float32)
            self.actions = np.array(action, dtype=np.int32)
            self.terminal = np.array(True if (terminal == 0) else False, dtype=np.bool)

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.Q_eval.device)
            with T.no_grad():
                actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.learn_next == True:
            self.Q_eval.optimizer.zero_grad()
            the_state = T.tensor(self.state).to(self.Q_eval.device)
            the_new_state = T.tensor(self.new_state).to(self.Q_eval.device)
            the_reward = T.tensor(self.reward).to(self.Q_eval.device)
            the_terminal = T.tensor(self.terminal).to(self.Q_eval.device)
            the_action = self.actions
            q_eval = self.Q_eval.forward(the_state)[the_action]
            q_next = self.Q_eval.forward(the_new_state)
            q_next = T.tensor(np.array(0.0, dtype=np.float32)).to(self.Q_eval.device) if the_terminal.item() else q_next
            q_target = the_reward + self.gamma * T.max(q_next, dim=0)[0]

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

class AgentBatchRewardProportional(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, layers=4, fc1_dim = 64, fc2_dim = 64, max_mem_size=1000000, eps_min=0.01, eps_dec=0.999996):
        self.mem_cntr = 0
        self.obs_cntr = 1.0
        self.nz_reward_cnt = 0.0
        self.learn_next = False
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dim, fc2_dim, n_actions, layers)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        self.learn_next = False
        rand = np.random.random()
        if reward != 0.0 and rand < (1 - (self.nz_reward_cnt / self.obs_cntr)):
            self.learn_next = True
            self.nz_reward_cnt = self.nz_reward_cnt + 1.0
            self.obs_cntr = self.obs_cntr + 1.0
        
        if reward == 0.0 and rand < (self.nz_reward_cnt / self.obs_cntr):
            self.learn_next = True
            self.obs_cntr = self.obs_cntr + 1.0
        if self.learn_next:
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.actions_memory[index] = action
            self.terminal_memory[index] = True if (terminal == 0) else False
            self.mem_cntr += 1

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.Q_eval.device)
            with T.no_grad():
                actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype = np.int32)
            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
            action_batch = self.actions_memory[batch]
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

class AgentBMRSRMP(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, layers=4, fc1_dim = 64, fc2_dim = 64, max_mem_size=1000000, eps_min=0.01, eps_dec=0.999996):
        self.mem_cntr = 0
        self.learn_next = False
        self.gamma = gamma
        self.max_magnitude = 0.0
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dim, fc2_dim, n_actions, layers)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        self.learn_next = False
        if np.abs(reward) > self.max_magnitude:
            self.max_magnitude = np.abs(reward)
        
        rand = np.random.random()
        if rand < np.sqrt((np.abs(reward) + 1.0) / (self.max_magnitude + 1.0)):
            self.learn_next = True

        if self.learn_next:
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.actions_memory[index] = action
            self.terminal_memory[index] = True if (terminal == 0) else False
            self.mem_cntr += 1

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.Q_eval.device)
            with T.no_grad():
                actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype = np.int32)
            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
            action_batch = self.actions_memory[batch]
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

class AgentNBND(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, layers=4, fc1_dim = 64, fc2_dim = 64, max_mem_size=1000000, eps_min=0.01, eps_dec=0.999996):
        self.mem_cntr = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dim, fc2_dim, n_actions, layers)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.actions_memory[index] = action
        self.terminal_memory[index] = True if (terminal == 0) else False
        self.mem_cntr += 1

    def choose_action(self, observation):
        state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.Q_eval.device)
        with T.no_grad():
            actions = F.softmax(self.Q_eval.forward(state), dim=1)
            rand = np.random.random()
            action = 0
            ind = 0
            while ind < self.n_actions and rand > 0:
                if rand < actions[0][ind]:
                    action = ind
                rand = rand - actions[0][ind]
                ind = ind + 1
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype = np.int32)
            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
            action_batch = self.actions_memory[batch]
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * T.sum(q_next * F.softmax(q_next, dim=1), dim=1)
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)