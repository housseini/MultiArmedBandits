import math
import random
import csv

class UCBAgent:
    def __init__(self, choices):
        self.choices = choices
        self.estimated_values = {choice: 0 for choice in choices}
        self.number_of_selections = {choice: 0 for choice in choices}
        self.total_rounds = 0

    def present(self, context, choices):
        self.total_rounds += 1
        if 0 in self.number_of_selections.values():
            # Si certains bras n'ont pas encore été choisis, choisissez-en un au hasard
            unselected_arms = [arm for arm, count in self.number_of_selections.items() if count == 0]
            chosen_arm = random.choice(unselected_arms)
        else:
            upper_bounds = {choice: self.estimated_values[choice] + math.sqrt(2 * math.log(self.total_rounds) / self.number_of_selections[choice]) for choice in choices}
            chosen_arm = max(upper_bounds, key=upper_bounds.get)
        self.last_choice = chosen_arm
        return chosen_arm

    def feedback(self, reward):
        chosen_arm = self.last_choice
        self.number_of_selections[chosen_arm] += 1
        n = self.number_of_selections[chosen_arm]
        self.estimated_values[chosen_arm] += (reward - self.estimated_values[chosen_arm]) / n
# Charger les données des films
def load_data():
    movies = {}
    with open('movies.csv', newline='', encoding='utf-8') as csvfile:
        moviereader = csv.reader(csvfile, delimiter=',')
        for row in moviereader:
            movies[row[0]] = row[1:]
    return movies
def calculate_reward(movie_id, user_ratings):
    # Dans une application réelle, vous auriez besoin d'une fonction plus sophistiquée
    # Pour cet exemple, nous allons juste simuler une évaluation utilisateur basée sur des évaluations stockées
    return user_ratings.get(movie_id, 0)

user_ratings = {
    '62': 5,
    '86': 3,
    # ...
}
movies = load_data()

# Créer un agent UCB
agent = UCBAgent(choices=list(movies.keys()))

# Boucle pour effectuer des recommandations
for _ in range(10):  # Faisons 10 recommandations pour cet exemple
    recommended_movie = agent.present(None, list(movies.keys()))
    reward = calculate_reward(recommended_movie, user_ratings)
    print(f"Recommandation: {movies[recommended_movie][0]}, Récompense: {reward}")
    agent.feedback(reward)
