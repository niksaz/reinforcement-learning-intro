# Author: Mikita Sazanovich

from time import time
import numpy as np
from collections import defaultdict

eps = 1e-3
gamma = 0.8
final_reward = 1


def compute_neighbours(N):
    def is_valid_state(x, y):
        return 0 <= x < N and 0 <= y < N
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neigh = [[[] for _ in range(N)] for _ in range(N)]
    for x in range(N):
        for y in range(N):
            if x == N-1 and y == N-1:
                neigh[x][y].append((x, y))
                continue
            for drt in dirs:
                xx = x + drt[0]
                yy = y + drt[1]
                if is_valid_state(xx, yy):
                    neigh[x][y].append((xx, yy))
    return neigh


def evaluate_policy(pi, r):
    N = len(pi)
    v = np.zeros((N, N), dtype=float)
    delta = 2*eps
    iters = 0
    while delta >= eps:
        iters += 1
        delta = 0.0
        for x in range(N):
            for y in range(N):
                old_v = v[x, y]
                xx, yy = pi[x][y]
                v[x, y] = r[(x, y, xx, yy)] + gamma*v[xx, yy]
                delta = max(delta, abs(v[x, y] - old_v))
    return v, iters


def find_best_neighbour(x, y, neigh, r, v):
    best_ngh = None
    best_v = None
    for ngh in neigh[x][y]:
        xx, yy = ngh
        cur_v = r[(x, y, xx, yy)] + gamma * v[xx, yy]
        if best_ngh is None or cur_v > best_v:
            best_ngh = ngh
            best_v = cur_v
    return best_ngh, best_v


def do_policy_improvement(neigh, r):
    start_time = time()
    N = len(neigh)
    pi = [[None for _ in range(N)] for _ in range(N)]
    for x in range(N):
        for y in range(N):
            neighs = len(neigh[x][y])
            pi[x][y] = neigh[x][y][np.random.randint(0, neighs)]
    v = np.zeros((N, N), dtype=float)
    iters_sum = 0
    while True:
        iters_sum += 1
        v, iters = evaluate_policy(pi, r)
        iters_sum += iters
        policy_stable = True
        for x in range(N):
            for y in range(N):
                best_ngh, best_v = find_best_neighbour(x, y, neigh, r, v)
                if best_ngh != pi[x][y]:
                    policy_stable = False
                pi[x][y] = best_ngh
        if policy_stable:
            break
    print(f'Iterations taken: {iters_sum}')
    print(f'Learning took {time() - start_time}')
    return pi, v


def do_value_iteration(neigh, r):
    start_time = time()
    N = len(neigh)
    v = np.zeros((N, N), dtype=float)
    delta = 2*eps
    iters = 0
    while delta >= eps:
        iters += 1
        delta = 0.0
        for x in range(N):
            for y in range(N):
                old_v = v[x, y]
                best_ngh, best_v = find_best_neighbour(x, y, neigh, r, v)
                v[x, y] = best_v
                delta = max(delta, abs(v[x, y] - old_v))
    pi = [[None for _ in range(N)] for _ in range(N)]
    for x in range(N):
        for y in range(N):
            best_ngh, _ = find_best_neighbour(x, y, neigh, r, v)
            pi[x][y] = best_ngh
    print(f'Iterations taken: {iters}')
    print(f'Learning took {time() - start_time}')
    return pi, v


def main():
    np.random.seed(0)
    N = 10

    neigh = compute_neighbours(N)
    r = defaultdict(int)
    r[(N-2, N-1, N-1, N-1)] = final_reward
    r[(N-1, N-2, N-1, N-1)] = final_reward

    # pi, v = do_policy_improvement(neigh, r)
    pi, v = do_value_iteration(neigh, r)

    print('v:')
    for x in range(N):
        for y in range(N):
            print(f'{v[x, y]:4.3}', end=' ')
        print()
    print('pi:')
    for x in range(N):
        for y in range(N):
            dx = pi[x][y][0] - x
            dy = pi[x][y][1] - y
            print(f'{(dx, dy)}', end=' ')
        print()


if __name__ == '__main__':
    main()
