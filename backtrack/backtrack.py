# This backtracking solution scores 664 in hustack.
import sys
sys.setrecursionlimit(10**7)


def input_():
    """
    Read input data.

    Returns:
        N (int): Number of customers.
        K (int): Number of routes/vehicles.
        d (list[int]): Service time of each customer.
        t (list[list[int]]): Travel time matrix.
    """
    N, K = map(int, input().split())
    d = [0] + list(map(int, input().split()))
    t = [list(map(int, input().split())) for _ in range(N + 1)]

    return N, K, d, t


def compute_increment(routes, t, d, k, customer):
    """
    Compute the additional cost when assigning a customer to route k.

    Args:
        routes (list[list[int]]): Current routes.
        t (list[list[int]]): Travel time matrix.
        d (list[int]): Service times.
        k (int): Route index.
        customer (int): Customer to insert.

    Returns:
        int: Incremental route cost.
    """
    if not routes[k]:
        return t[0][customer] + d[customer] + t[customer][0]

    last = routes[k][-1]

    return (
        -t[last][0]
        + t[last][customer]
        + d[customer]
        + t[customer][0]
    )


def solve(N, K, d, t):
    """
    Solve the routing problem using backtracking with branch-and-bound pruning.

    Args:
        N (int): Number of customers.
        K (int): Number of routes.
        d (list[int]): Service times.
        t (list[list[int]]): Travel time matrix.

    Returns:
        list[list[int]]: Best route assignment found.
    """

    # Sort customers by difficulty
    customers = list(range(1, N + 1))
    customers.sort(
        key=lambda x: d[x] + t[0][x] + t[x][0],
        reverse=True
    )

    # Initialize variables
    routes = [[] for _ in range(K)]
    cost = [0] * K

    best = float("inf")
    best_routes = None

    remaining_work = [0] * (N + 1)

    # Preprocessing remaining lower bound
    for i in range(N - 1, -1, -1):
        c = customers[i]
        remaining_work[i] = (
            remaining_work[i + 1]
            + d[c]
            + t[0][c]
            + t[c][0]
        )

    def backtrack(idx):
        """
        Recursive backtracking search.

        Args:
            idx (int): Current customer index
                in the sorted customer list.
        """
        nonlocal best, best_routes

        # All customers assigned
        if idx == N:
            cur = max(cost)

            if cur < best:
                best = cur
                best_routes = [r[:] for r in routes]

            return

        # Lower bound
        current_max = max(cost)

        lb = max(
            current_max,
            min(cost) + remaining_work[idx] / K
        )

        # Branch-and-bound pruning
        if lb >= best:
            return

        customer = customers[idx]

        # Symmetry pruning
        seen = set()

        for k in range(K):

            if cost[k] in seen:
                continue

            seen.add(cost[k])

            inc = compute_increment(
                routes,
                t,
                d,
                k,
                customer
            )

            # Assign customer
            cost[k] += inc
            routes[k].append(customer)

            if max(cost) < best:
                backtrack(idx + 1)

            # Undo assignment
            routes[k].pop()
            cost[k] -= inc

            # Avoid equivalent empty-route states
            if len(routes[k]) == 0:
                break

    backtrack(0)

    return best_routes


def output_(K, best_routes):
    """
    Print solution in required format.

    Args:
        K (int): Number of routes.
        best_routes (list[list[int]]): Best routes found.
    """
    print(K)
    for r in best_routes:
        route = [0] + r + [0]

        print(len(route))
        print(*route)


def main():
    N, K, d, t = input_()
    best_routes = solve(N, K, d, t)
    output_(K, best_routes)


if __name__ == "__main__":
    main()
