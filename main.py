import random
import matplotlib.pyplot as plt
import numpy as np
from EpsilonDecreasingAgent import EpsilonDecreasingAgent as EDA, SimpleBandit as SB
from  UCBAgent import  UCBAgent as ucb
from  SoftmaxAgent import  SoftmaxAgent as soft


def exécuter_simulation(agent_class, bandit, choices, n=1000, nb_exécutions=100):
  scores = []
  regrets = []

  for _ in range(nb_exécutions):
    agent = agent_class(choices)
    _, score, regret = bandit.run(agent, n=n)
    scores.append(score)
    regrets.append(regret)

  return scores, regrets

  def plot_line(data, labels):
    for d, label in zip(data, labels):
      plt.plot(np.mean(d), label=label)
    plt.legend()
    plt.title('Tendance des Scores Moyens')
    plt.xlabel('Numéro de Simulation')
    plt.ylabel('Score Moyen')
    plt.show()


def plot_histogram(data, labels):
  for d, label in zip(data, labels):
    plt.hist(d, alpha=0.5, label=label)
  plt.legend(loc='upper right')
  plt.title('Distribution des Scores')
  plt.xlabel('Score')
  plt.ylabel('Fréquence')
  plt.show()


def plot_violin(data, labels):
  plt.violinplot(data)
  plt.xticks(range(1, len(labels) + 1), labels)
  plt.title('Distribution des Scores')
  plt.ylabel('Score')
  plt.show()

def plot_line(data, labels):
    for d, label in zip(data, labels):
      plt.plot(np.mean(d), label=label)
    plt.legend()
    plt.title('Tendance des Scores Moyens')
    plt.xlabel('Numéro de Simulation')
    plt.ylabel('Score Moyen')
    plt.show()
if __name__ == '__main__':
  bandit = SB()
  choices = list(bandit.outcomes.keys())
  eda_scores, eda_regrets = exécuter_simulation(EDA, bandit, choices)
  ucb_scores, ucb_regrets = exécuter_simulation(ucb, bandit, choices)
  softmax_scores, softmax_regrets = exécuter_simulation(soft, bandit, choices)

  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.boxplot([eda_scores, ucb_scores, softmax_scores], labels=['EDA', 'UCB', 'Softmax'])
  plt.title('Distribution des Scores')
  plt.ylabel('Score')

  plt.subplot(1, 2, 2)
  plt.boxplot([eda_regrets, ucb_regrets, softmax_regrets], labels=['EDA', 'UCB', 'Softmax'])
  plt.title('Distribution des Regrets')
  plt.ylabel('Regret')

  plt.tight_layout()
  plt.show()
  plot_line([eda_scores, ucb_scores, softmax_scores], ['EDA', 'UCB', 'Softmax'])
  plot_histogram([eda_scores, ucb_scores, softmax_scores], ['EDA', 'UCB', 'Softmax'])
  plot_violin([eda_scores, ucb_scores, softmax_scores], ['EDA', 'UCB', 'Softmax'])


  bandit = SB()
  agent = EDA(choices=list(bandit.outcomes.keys()))

  cnt, total_score, total_regret = bandit.run(agent, n=1000)
  print(f"Nombre de tours : {cnt}, Score total : {total_score}, Regret total : {total_regret}")

  ucb=ucb(choices=list(bandit.outcomes.keys()))
  cnt, total_score, total_regret = bandit.run(ucb, n=1000)
  print(f"Nombre de tours : {cnt}, Score total : {total_score}, Regret total : {total_regret}")
  soft=soft(choices=list(bandit.outcomes.keys()))
  cnt, total_score, total_regret = bandit.run(soft, n=1000)
  print(f"Nombre de tours : {cnt}, Score total : {total_score}, Regret total : {total_regret}")



