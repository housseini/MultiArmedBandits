import csv
import random
import math

class SoftmaxAgent:
    def __init__(self, choices, temperature=1.0):
        self.temperature = temperature
        self.choices = choices
        self.estimated_values = {choice: 0 for choice in choices}
        self.number_of_selections = {choice: 0 for choice in choices}

    def present(self, context, choices):
        probabilities = self.softmax_probabilities()
        chosen_arm = random.choices(self.choices, weights=probabilities)[0]
        self.last_choice = chosen_arm
        return chosen_arm

    def feedback(self, score):
        chosen_arm = self.last_choice
        self.number_of_selections[chosen_arm] += 1
        n = self.number_of_selections[chosen_arm]
        self.estimated_values[chosen_arm] += (score - self.estimated_values[chosen_arm]) / n

    def softmax_probabilities(self):
        total = sum([math.exp(v / self.temperature) for v in self.estimated_values.values()])
        probabilities = [math.exp(self.estimated_values[choice] / self.temperature) / total for choice in self.choices]
        return probabilities

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
# Supposons que user_ratings est un dictionnaire où les clés sont les ID de films et les valeurs sont les évaluations
user_ratings = {
    '1': 5,  # L'utilisateur a donné la note 5 au film avec l'ID 1
    '2': 3,  # L'utilisateur a donné la note 3 au film avec l'ID 2
    # ...
}

# Charger les données des films
movies = load_data()

# Créer un agent Softmax
agent = SoftmaxAgent(choices=list(movies.keys()), temperature=0.1)

# Boucle pour effectuer des recommandations
for _ in range(10):  # Faisons 10 recommandations pour cet exemple
    recommended_movie = agent.present(None, list(movies.keys()))
    reward = calculate_reward(recommended_movie, user_ratings)
    print(f"Recommandation: {movies[recommended_movie][0]}, Récompense: {reward}")
    agent.feedback(reward)
