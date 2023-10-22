import random


class SimpleBandit:
    def __init__(self):
        self.outcomes = {
            'A': {1: 0.7, 4: 0.1, 0: 0.2},
            'B': {6: 0.25, 3: 0.25, 0: 0.5},
            'C': {9: 0.3, 3: 0.1, 1: 0.15, 0: 0.65},
            'D': {11: 0.04, 2: 0.16, 1: 0.15, 0: 0.65}
        }

    def run(self, agent, n=100):
        total_score = 0
        choices = list(self.outcomes.keys())
        expectations = [sum([k * self.outcomes[choice][k] for k in self.outcomes[choice]]) for choice in choices]
        best_expectation = max(expectations)

        for _ in range(n):
            choice = agent.present(0, choices)  # context is not used
            possibleScores = list(self.outcomes[choice].keys())
            probs = [self.outcomes[choice][k] for k in possibleScores]
            score = random.choices(possibleScores, weights=probs)[0]
            agent.feedback(score)
            total_score += score

        total_regret = n * best_expectation - total_score
        return n, total_score, total_regret


class EpsilonDecreasingAgent:
    def __init__(self, choices, initial_epsilon=1.0, decrease_factor=0.99):
        self.epsilon = initial_epsilon
        self.decrease_factor = decrease_factor
        self.choices = choices
        self.estimated_values = {choice: 0 for choice in choices}
        self.number_of_selections = {choice: 0 for choice in choices}

    def present(self, context, choices):

        if random.random() < self.epsilon:
            chosen_arm = random.choice(choices)
        else:
            chosen_arm = max(self.estimated_values, key=self.estimated_values.get)

        self.last_choice = chosen_arm


        return chosen_arm


    def feedback(self, score):

        chosen_arm = self.last_choice
        self.number_of_selections[chosen_arm] += 1
        n = self.number_of_selections[chosen_arm]

        # Mise à jour de la valeur estimée du bras choisi
        self.estimated_values[chosen_arm] += (score - self.estimated_values[chosen_arm]) / n

        # Diminuer la valeur de epsilon
        self.epsilon *= self.decrease_factor


