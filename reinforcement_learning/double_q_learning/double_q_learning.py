'''
Mapping bài toán RF và Q-learning xem trong file tabular và 
- Vấn đề của DQN: cùng một network (online network) làm 2 nhiệm vụ chọn action tốt nhất và đánh giá action đó
    Toán tử max sẽ chọn các action có estimation noise cao => Q-value bị overestimate

- Double Q-Learning:
    Online Network: chọn action tốt nhất (train liên tục)
    Target Network: đánh giá action (update chậm)
- y=r+γQtarget​(s′,argmax_a′ (​Qonline​(s′,a′)))
- Replay memory: (state, action, reward, next_state, done)
'''

import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# READ INPUT
N, K = map(int, input().split())
maintenance_time = list(map(int, input().split()))
maintenance_time = [0] + maintenance_time
travel_time = []

for _ in range(N + 1):
    row = list(map(int, input().split()))
    travel_time.append(row)

# HYPERPARAMETERS
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
EPISODES = 1000
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQUENCY = 20

# ACTION SPACE
ALL_ACTIONS = []

for staff_id in range(K):
    for customer_id in range(1, N + 1):
        ALL_ACTIONS.append((staff_id, customer_id))

ACTION_SIZE = len(ALL_ACTIONS)

# REPLAY MEMORY
replay_memory = deque(
    maxlen=MEMORY_SIZE
)

# DQN NETWORK
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

# STATE REPRESENTATION
STATE_SIZE = N + K + K

def build_state_vector(visited, staff_positions, staff_workloads):
    """
    Convert environment state
    to neural network vector
    """
    state_vector = []

    # VISITED CUSTOMERS
    for customer_id in range(1, N + 1):
        if visited[customer_id]:
            state_vector.append(1.0)
        else:
            state_vector.append(0.0)

    # STAFF POSITIONS
    for position in staff_positions:
        state_vector.append(position / N)

    # STAFF WORKLOADS
    for workload in staff_workloads:
        state_vector.append(workload / 10000.0)
    
    return np.array(state_vector, dtype=np.float32)

# CREATE NETWORKS
online_network = DQNNetwork(STATE_SIZE, ACTION_SIZE)
target_network = DQNNetwork(STATE_SIZE, ACTION_SIZE)

# Copy initial weights
target_network.load_state_dict(online_network.state_dict())
optimizer = optim.Adam(online_network.parameters(), lr=LEARNING_RATE)

# Huber Loss
loss_function = nn.SmoothL1Loss()

# HELPER FUNCTIONS
def all_customers_served(visited):
    return all(visited[1:])

def get_valid_actions(visited):
    """
    Only unvisited customers are valid
    """

    valid_actions = []

    for action_index, action in enumerate(ALL_ACTIONS):

        staff_id, customer_id = action

        if not visited[customer_id]:

            valid_actions.append(action_index)

    return valid_actions

def choose_action(state_vector, valid_actions):
    """
    Epsilon-greedy action selection
    """
    global EPSILON
    # EXPLORATION
    if random.random() < EPSILON:
        return random.choice(valid_actions)

    # EXPLOITATION
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

def calculate_reward(workloads):
    """
    Minimize:
        max workload
        workload imbalance
    """
    imbalance = (max(workloads) - min(workloads))
    reward = -(max(workloads) + imbalance)

    return reward / 1000.0

# TRAINING FUNCTION
def train_double_dqn():
    if len(replay_memory) < BATCH_SIZE:
        return
    
    mini_batch = random.sample(replay_memory,BATCH_SIZE)
    states = []
    targets = []

    for (state, action, reward, next_state, done) in mini_batch:
        # CONVERT TO TENSORS
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # CURRENT Q VALUES
        current_q_values = (online_network(state_tensor).detach().cpu().numpy().flatten())
        if done:
            target_q = reward

        else:
            # DOUBLE DQN CORE
            with torch.no_grad():
                #khác biệt lớn nhất giữa deep và double q learning nằm ở đoạn train này
                # ONLINE NETWORK CHOOSES ACTION
                online_next_q = online_network(next_state_tensor)
                best_next_action = torch.argmax(online_next_q[0]).item()

                # TARGET NETWORK EVALUATES ACTION
                target_next_q = target_network(next_state_tensor)

                target_q = (reward + DISCOUNT_FACTOR * target_next_q[0][best_next_action].item())

        # Update only selected action
        current_q_values[action] = target_q
        states.append(state)
        targets.append(current_q_values)

    # BATCH TENSORS
    states_tensor = torch.FloatTensor(np.array(states))
    targets_tensor = torch.FloatTensor(np.array(targets))

    # FORWARD PASS
    predictions = online_network(states_tensor)
    loss = loss_function(predictions, targets_tensor)

    # BACKPROPAGATION
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# TRAINING LOOP
for episode in range(EPISODES):
    # print(f"episode: {episode}")
    # RESET ENVIRONMENT
    visited = [False] * (N + 1)

    staff_positions = [0] * K

    staff_workloads = [0] * K

    done = False

    while not done:
        # BUILD CURRENT STATE

        current_state = build_state_vector(visited, staff_positions, staff_workloads)

        # GET VALID ACTIONS
        valid_actions = get_valid_actions(visited)

        # CHOOSE ACTION
        action_index = choose_action(current_state, valid_actions)

        staff_id, customer_id = (ALL_ACTIONS[action_index])

        # EXECUTE ACTION
        current_position = staff_positions[staff_id]

        moving_time = travel_time[current_position][customer_id]

        service_time = maintenance_time[customer_id]

        # Update workload
        staff_workloads[staff_id] += (moving_time + service_time)

        # Update position
        staff_positions[staff_id] = (customer_id)

        # Mark served
        visited[customer_id] = True

        # TERMINATION CHECK
        done = all_customers_served(visited)

        # BUILD NEXT STATE
        next_state = build_state_vector(visited, staff_positions, staff_workloads)

        # COMPUTE REWARD
        reward = calculate_reward(
            staff_workloads
        )

        # STORE EXPERIENCE
        replay_memory.append(
            (current_state, action_index, reward, next_state, done)
        )

        # TRAIN DOUBLE DQN
        train_double_dqn()

    # UPDATE TARGET NETWORK

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_network.load_state_dict(online_network.state_dict())

    # DECAY EPSILON
    if EPSILON > EPSILON_MIN:
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

# ============================================================
# BUILD FINAL SOLUTION
# ============================================================

visited = [False] * (N + 1)

staff_positions = [0] * K

staff_workloads = [0] * K

staff_routes = [[0] for _ in range(K)]

done = False

while not done:

    current_state = build_state_vector(visited, staff_positions, staff_workloads)

    valid_actions = get_valid_actions(visited)

    state_tensor = torch.FloatTensor(current_state).unsqueeze(0)

    with torch.no_grad():
        q_values = online_network(state_tensor)

    # GREEDY ACTION SELECTION
    best_action = None

    best_q = float('-inf')

    for action_index in valid_actions:

        q_value = q_values[0][action_index].item()

        if q_value > best_q:

            best_q = q_value

            best_action = action_index

    staff_id, customer_id = (ALL_ACTIONS[best_action])

    # EXECUTE ACTION
    current_position = staff_positions[
        staff_id
    ]

    moving_time = travel_time[
        current_position
    ][customer_id]

    service_time = maintenance_time[
        customer_id
    ]

    staff_workloads[staff_id] += (
        moving_time + service_time
    )

    staff_positions[staff_id] = customer_id

    visited[customer_id] = True

    staff_routes[staff_id].append(
        customer_id
    )

    done = all_customers_served(
        visited
    )

# RETURN TO DEPOT
for staff_id in range(K):

    last_position = staff_positions[
        staff_id
    ]

    staff_workloads[staff_id] += (
        travel_time[last_position][0]
    )

    staff_routes[staff_id].append(0)

# OUTPUT
print(K)

for route in staff_routes:

    print(len(route))

    print(*route)