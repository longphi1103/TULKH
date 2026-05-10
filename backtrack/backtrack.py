# This backtracking solution scores 664 in hustack.
import sys
sys.setrecursionlimit(10**7)


# input
N, K = map(int, input().split())
d = [0] + list(map(int, input().split()))
t = [list(map(int, input().split())) for _ in range(N + 1)]


# sort customers by difficulty
customers = list(range(1, N + 1))
customers.sort(
    key=lambda x: d[x] + t[0][x] + t[x][0],
    reverse=True
)


# init variables
routes = [[] for _ in range(K)]
cost = [0] * K

best = float('inf')
best_routes = None

remaining_work = [0] * (N + 1)


# preprocessing remaining lower bound
for i in range(N - 1, -1, -1):
    c = customers[i]
    remaining_work[i] = (
        remaining_work[i + 1]
        + d[c]
        + t[0][c]
        + t[c][0]
    )


def compute_increment(k, customer):
    if not routes[k]:
        return t[0][customer] + d[customer] + t[customer][0]

    last = routes[k][-1]

    return (
        -t[last][0]
        + t[last][customer]
        + d[customer]
        + t[customer][0]
    )


# backtrack
def backtrack(idx):
    global best, best_routes

    if idx == N:
        cur = max(cost)

        if cur < best:
            best = cur
            best_routes = [r[:] for r in routes]

        return

    # lower bound
    current_max = max(cost)

    lb = max(
        current_max,
        min(cost) + remaining_work[idx] / K
    )

    if lb >= best:
        return

    customer = customers[idx]

    seen = set()

    for k in range(K):

        # symmetry pruning
        if cost[k] in seen:
            continue

        seen.add(cost[k])

        inc = compute_increment(k, customer)

        cost[k] += inc
        routes[k].append(customer)

        if max(cost) < best:
            backtrack(idx + 1)

        routes[k].pop()
        cost[k] -= inc

        if len(routes[k]) == 0:
            break


backtrack(0)


# output
print(K)
for r in best_routes:
    route = [0] + r + [0]
    print(len(route))
    print(*route)
