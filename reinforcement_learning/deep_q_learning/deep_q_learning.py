'''
Kiến thức cơ bản về Reinformance learning và q-learning xem trong tabular q-learning

- Deep Q-Learning (DQN): dùng neural network để học trực tiếp Q-value
- Q(state, action, θ) với θ trọng số neural network
    => action chọn Q lớn nhất

- Approximate Q-Learning tự học feature còn DQN tự học insight

- Training: Online Network (học liên tục): 
    input layer -> x hidden layer -> output: Q-values

- Random experiences từ replay memory

- Inference: Target NetWork (update(copy) chậm từ online network)

- Bellman trong DQN:
    y = r + γmax_a′(​Qtarget​(s′,a′))

    - Loss function: L = (y − Qonline​(s,a))2
'''


import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#input
N, K = map(int, input().split())

maintenance_time = list(map(int, input().split()))
maintenance_time = [0] + maintenance_time

travel_time = []
for _ in range(N + 1):
    row = list(map(int, input().split()))
    travel_time.append(row)

#q learning and neural network params
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EPISODES = 1000
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQUENCY = 20

#action space (staff_id, customer_id)
ALL_ACTIONS = []

for staff_id in range(K):

    for customer_id in range(1, N + 1):

        ALL_ACTIONS.append(
            (staff_id, customer_id)
        )

ACTION_SIZE = len(ALL_ACTIONS)

#replay memory
replay_memory = deque(maxlen=MEMORY_SIZE)

#deep q-learning network
class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        return self.network(x)
    
#state
STATE_SIZE = N + K + K
def build_state_vector(visited, staff_positions, staff_workloads):
    #chuyen state bai toan sang neural network input (vector)
    state_vector = []
    #visited customer
    for customer_id in range(1, N + 1):
        if visited[customer_id]:
            state_vector.append(1.0)
        else:
            state_vector.append(0.0)
    #staff position
    for position in staff_positions:
        state_vector.append(position / N)
    #staff workload
    for workload in staff_workloads:
        state_vector.append(workload / 10000.0)
    
    return np.array(state_vector, dtype=np.float32)

#networks
online_network = DQNNetwork(STATE_SIZE, ACTION_SIZE)
target_network = DQNNetwork(STATE_SIZE, ACTION_SIZE)

#initial weight
target_network.load_state_dict(online_network.state_dict())
optimizer = optim.Adam(
    online_network.parameters(),
    lr=LEARNING_RATE
)
loss_function = nn.MSELoss()



#helper fuctions
def all_customers_served(visited): #kiem tra tat ca khach hang da duoc phuc vu
    return all(visited[1:])

def get_valid_actions(visited):
    """
    Only customers not yet visited are valid
    """

    valid_actions = []

    for action_index, action in enumerate(ALL_ACTIONS):

        staff_id, customer_id = action

        if not visited[customer_id]:

            valid_actions.append(action_index)

    return valid_actions

def choose_action(state_vector, valid_actions): #epsilon greedy
    #exploration - random
    if random.random() < EPSILON:
        return random.choice(valid_actions)
    
    #exploitation - greedy
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

    with torch.no_grad():
        q_values = online_network(state_tensor)

    best_action = None
    best_q = float('-inf')

    for action_index in valid_actions:
        q_value = q_values[0][action_index].item()
        if q_value > best_q:
            best_q = q_value
            best_action = action_index

    return best_action

def cal_reward(workloads): # max workload cang nho tho reward tot hon
    imbalance = max(workloads) - min(workloads)

    return -(max(workloads) + imbalance)


#train from replay memory
def train_dqn():
    if len(replay_memory) < BATCH_SIZE:
        return
    mini_batch = random.sample(
        replay_memory,
        BATCH_SIZE
    )
    states = []
    targets = []
    for (state, action, reward, next_state, done) in mini_batch:
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        current_q_values = online_network(state_tensor).detach().numpy()
        
        #compute target
        if done:
            target_q = reward
        else:
            with torch.no_grad():
                future_q_values = target_network(next_state_tensor)
                target_q = (reward + DISCOUNT_FACTOR * torch.max(future_q_values).item())

        current_q_values[action] = target_q
        states.append(state)
        targets.append(current_q_values)

    state_tensor = torch.FloatTensor(states)
    targets_tensor = torch.FloatTensor(np.array(targets))

    #forward + backpropagation
    predictions = online_network(state_tensor)
    loss = loss_function(predictions, targets_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# training ========================================================================================================
for episode in range(EPISODES):
    #initial state
    visited = [False] * (N+1)
    #vi tri cua tung nhan vien (tat ca nhan vien bat dau tu depot 0)
    staff_positions = [0] * K
    #workload cua tung nhan vien (ban dau la 0)
    staff_workloads = [0] * K

    done = False

    while not done:
        current_state = build_state_vector(
            visited,
            staff_positions,
            staff_workloads
        )

        valid_actions = get_valid_actions(visited)
        
        action_index = choose_action(current_state, valid_actions) #choose action
        staff_id, customer_id = ALL_ACTIONS[action_index]
        
        #transaction
        current_position = staff_positions[staff_id]
        # travel time
        moving_time = travel_time[current_position][customer_id]
        # service time
        service_time = maintenance_time[customer_id]
        # update workload
        staff_workloads[staff_id] += (moving_time + service_time)
        # update position
        staff_positions[staff_id] = customer_id
        # mark customer served
        visited[customer_id] = True

        done = all_customers_served(visited)

        next_state = build_state_vector(
            visited,
            staff_positions,
            staff_workloads
        )

        reward = cal_reward(staff_workloads)

        replay_memory.append(
            (current_state, action_index, reward, next_state, done)
        )

        #train network
        train_dqn()

    #update target network
    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_network.load_state_dict(online_network.state_dict())
    
    #decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY


# inference build fianl solution=======================================================================================
visited = [False] * (N + 1)

staff_positions = [0] * K

staff_workloads = [0] * K
staff_routes = [[0] for _ in range(K)]
done = False

while not done:

    current_state = build_state_vector(
        visited,
        staff_positions,
        staff_workloads
    )

    valid_actions = get_valid_actions(
        visited
    )

    state_tensor = torch.FloatTensor(
        current_state
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = online_network(state_tensor)

    best_action = None
    best_q = float('-inf')
    for action_index in valid_actions:

        q_value = q_values[0][action_index].item()

        if q_value > best_q:

            best_q = q_value
            best_action = action_index

    staff_id, customer_id = (
        ALL_ACTIONS[best_action]
    )

    current_position = staff_positions[staff_id]

    moving_time = travel_time[current_position][customer_id]

    service_time = maintenance_time[customer_id]

    staff_workloads[staff_id] += (
        moving_time + service_time
    )

    staff_positions[staff_id] = customer_id
    visited[customer_id] = True

    staff_routes[staff_id].append(customer_id)

    done = all_customers_served(visited)

#return to depot

for staff_id in range(K):

    last_position = staff_positions[staff_id]

    staff_workloads[staff_id] += (
        travel_time[last_position][0]
    )

    staff_routes[staff_id].append(0)

#output
print(K)

for route in staff_routes:

    print(len(route))

    print(*route)