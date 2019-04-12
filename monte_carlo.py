# Author: Mikita Sazanovich

import random

import matplotlib.pyplot as plt
import numpy as np

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
          'T': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}
actions = ['H', 'S']
eps = 1e-2
epoch_iterations = 1000
epochs = 100


def next_card():
    ps = random.randrange(len(RANKS))
    rank = RANKS[ps]
    return VALUES[rank]


def resolve_hand(hand):
    value = sum(hand)
    i = 0
    while value > 21 and i < len(hand):
        if hand[i] == 11:
            value -= 10
        i += 1
    return value


def have_free_ace(hand):
    value = sum(hand)
    i = 0
    while value > 21 and i < len(hand):
        if hand[i] == 11:
            value -= 10
        i += 1
    while value <= 21 and i < len(hand):
        if hand[i] == 11:
            return True
        i += 1
    return False


def resolve_game(agent_hand, dealer_card):
    agent_score = resolve_hand(agent_hand)
    dealer_hand = [dealer_card, next_card()]
    while resolve_hand(dealer_hand) < 17:
        dealer_hand.append(next_card())
    dealer_score = resolve_hand(dealer_hand)
    if dealer_score > 21 or agent_score > dealer_score:
        return 1
    elif agent_score < dealer_score:
        return -1
    else:
        return 0


def get_state(agent_hand, dealer_card):
    agent_score = resolve_hand(agent_hand)
    free_ace = have_free_ace(agent_hand)
    return agent_score, free_ace, dealer_card


def draw_init_hand():
    hand = []
    while resolve_hand(hand) < 12:
        hand.append(next_card())
    return hand


def play_game(pi):
    dealer_card = next_card()
    agent_hand = draw_init_hand()
    game_log = []
    while True:
        state = get_state(agent_hand, dealer_card)
        game_log.append(state)
        p = []
        for action in actions:
            p.append(pi[(state, action)])
        choice = np.random.choice(2, 1, p=p)[0]
        action = actions[choice]
        game_log.append(action)
        if action == 'H':
            agent_hand.append(next_card())
        elif action == 'S':
            reward = resolve_game(agent_hand, dealer_card)
            game_log.append(reward)
            break
        score = resolve_hand(agent_hand)
        if score > 21:
            game_log.append(-1)
            break
    return game_log


def train_agent():
    winrates = []
    states = []
    for score in range(12, 21+1):
        for has_ace in [False, True]:
            for dealer_card in range(2, 11+1):
                state = (score, has_ace, dealer_card)
                states.append(state)
    Q, cumulative, total, pi = {}, {}, {}, {}
    for state in states:
        for action in actions:
            Q[(state, action)] = 0.0
            cumulative[(state, action)] = 0
            total[(state, action)] = 0
            pi[(state, action)] = 1.0 / len(actions)
    won_games = 0
    total_games = 0
    for it in range(epochs * epoch_iterations):
        game_log = play_game(pi)
        reward = game_log[-1]
        won_games += 1 if reward == 1 else 0
        total_games += 1
        if total_games == epoch_iterations:
            winrates.append(won_games / total_games)
            won_games = 0
            total_games = 0

        seen_pairs = set()
        seen_states = set()
        for i in range(0, len(game_log) - 1, 2):
            state = game_log[i]
            action = game_log[i + 1]
            pair = (state, action)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            seen_states.add(state)
            cumulative[pair] += reward
            total[pair] += 1
            Q[pair] = cumulative[pair] / total[pair]
        for state in seen_states:
            best_action = None
            best_quality = None
            for action in actions:
                quality = Q[(state, action)]
                if best_quality is None or best_quality < quality:
                    best_action = action
                    best_quality = quality
            for action in actions:
                if action == best_action:
                    pi[(state, action)] = 1 - eps + eps / len(actions)
                else:
                    pi[(state, action)] = eps / len(actions)
    return winrates


def winrates_with_seed(seed):
    random.seed(seed)
    winrates = train_agent()
    return winrates


def main():
    epoch_winrates = [[] for _ in range(epochs)]
    seeds = 10
    for seed in range(seeds):
        winrates = winrates_with_seed(seed)
        for epoch, winrate in enumerate(winrates):
            epoch_winrates[epoch].append(winrate)
    winrate_ups = []
    winrate_means = []
    winrate_downs = []
    for epoch in range(epochs):
        winrates = epoch_winrates[epoch]
        mean = np.mean(winrates)
        se = np.std(winrates) / np.sqrt(seeds)
        winrate_ups.append(mean + se)
        winrate_means.append(mean)
        winrate_downs.append(mean - se)

    xs = range(epochs)
    lines = []
    line, = plt.plot(xs, winrate_ups, label='mean+SE')
    lines.append(line)
    line, = plt.plot(xs, winrate_means, label='mean')
    lines.append(line)
    line, = plt.plot(xs, winrate_downs, label='mean-SE')
    lines.append(line)
    plt.xlabel(f'epochs (with {epoch_iterations} iterations each)')
    plt.ylabel('winrates')
    plt.legend(handles=lines)
    plt.show()


if __name__ == '__main__':
    main()
