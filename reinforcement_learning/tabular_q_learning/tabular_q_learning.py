'''
Mapping bài toán Reinforcement learning:

- State mỗi thời điểm:
    Những khách nào đã được phục vụ
    Vị trí hiện tại của mỗi nhân viên
    Tổng thời gian hiện tại của mỗi nhân viên

- Action: chọn nhân viên, với nhân viên đó chọn khách tiếp theo (epsilon-greedy)
- Transition (chuyển trạng thái):
    Nhân viên di chuyển
    Bảo trì
    Cập nhật workload mỗi nhân viên  
    Đánh dấu khách đã phục vụ

- Reward: min max(workload) --> reward = -max(staff_workloads) reward nên phạt workload lớn

Q-table: Q(statem,action)
Q(s,a)←Q(s,a)+α[r+γmax_a′Q(s′,a′)−Q(s,a)] (bellman)
'''


import random
from collections import defaultdict

#input
N, K = map(int, input().split())

maintenance_time = list(map(int, input().split()))
maintenance_time = [0] + maintenance_time

travel_time = []
for _ in range(N + 1):
    row = list(map(int, input().split()))
    travel_time.append(row)

#q learning params
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.2
EPISODES = 5000

Q_table = defaultdict(float)

#helper fuctions
def all_customers_served(visited): #kiem tra tat ca khach hang da duoc phuc vu
    return all(visited[1:])

def get_possible_actions(visited): # tim tat ca cac actions kha thi 
    actions = []

    for customer in range(1, N+1):
        if not visited[customer]:
            for staff in range(K):
                actions.append((staff, customer))

    return actions

def choose_action(state, possible_actions): #epsilon greedy
    #exploration - random
    if random.random() < EPSILON:
        return random.choice(possible_actions)
    
    #exploitation - greedy
    best_action = None
    best_q = float('-inf')

    for action in possible_actions:
        q_value = Q_table[(state, action)]
        if q_value > best_q:
            best_q = q_value
            best_action = action

    return best_action

def cal_reward(workloads): # max workload cang nho tho reward tot hon
    return -max(workloads)


# training ========================================================================================================
for episode in range(EPISODES):
    #initial state
    visited = [False] * (N+1)
    #vi tri cua tung nhan vien (tat ca nhan vien bat dau tu depot 0)
    staff_positions = [0] * K
    #workload cua tung nhan vien (ban dau la 0)
    staff_workloads = [0] * K
    #route cua tung nhan vien
    staff_routes = [[0] for _ in range(K)]

    while not all_customers_served(visited):
        state = (
            tuple(visited),
            tuple(staff_positions)
        )

        possible_actions = get_possible_actions(visited)
        
        action = choose_action(state, possible_actions) #choose action
        staff_id, customer_id = action
        
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
        # save route
        staff_routes[staff_id].append(customer_id)

        next_state = (
            tuple(visited),
            tuple(staff_positions)
        )

        reward = cal_reward(staff_workloads)

        next_possible_actions = get_possible_actions(visited)

        if next_possible_actions:
            best_future_q = max(
                Q_table[(next_state, next_action)]
                for next_action in next_possible_actions
            )
        else: best_future_q = 0

        #q leanig update
        current_q = Q_table[(state, action)]

        new_q = current_q + LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR * best_future_q
            - current_q
        )

        Q_table[(state, action)] = new_q

# inference build fianl solution=======================================================================================
visited = [False] * (N + 1)

staff_positions = [0] * K

staff_workloads = [0] * K

staff_routes = [[0] for _ in range(K)]

while not all_customers_served(visited):

    state = (
        tuple(visited),
        tuple(staff_positions)
    )

    possible_actions = get_possible_actions(visited)

    best_action = None
    best_q = float('-inf')
    for action in possible_actions:

        q_value = Q_table[(state, action)]

        if q_value > best_q:
            best_q = q_value
            best_action = action

    staff_id, customer_id = best_action

    current_position = staff_positions[staff_id]

    moving_time = travel_time[current_position][customer_id]

    service_time = maintenance_time[customer_id]

    staff_workloads[staff_id] += (
        moving_time + service_time
    )

    staff_positions[staff_id] = customer_id
    visited[customer_id] = True

    staff_routes[staff_id].append(customer_id)

#returen to depot

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