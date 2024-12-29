import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Pour éxecuter le programme veuillez mettre le bon chemin de fichier
data = pd.read_csv('', sep=';')

# Calcul du prix moyen et des rendements log
data['Prix_moy'] = (data['High'] + data['Low']) / 2
data['rt'] = np.log(data['Prix_moy'] / data['Prix_moy'].shift(1))

# Conversion des dates
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Calcul de la volatilité annuelle
vol = data['rt'].std() * np.sqrt(252)

# Sélection des données de 2023-2024
data_2023 = data[(data['Date'] >= '2023-01-04') & (data['Date'] <= '2024-11-19')].copy()
vol_2023 = data_2023['rt'].std() * np.sqrt(252)

############ Simulation Black-Scholes
SO = data_2023['Prix_moy'].iloc[-1]
sigma = vol_2023
r = 2.9 / 100
T = 0.5  # En années
n_steps = 180  # Nombre de pas de temps
dt = T / n_steps
t = np.linspace(0, T, n_steps)

np.random.seed(42)
dW = np.random.normal(0, np.sqrt(dt), n_steps)
W = np.cumsum(dW)
prediction_black = SO * np.exp((r - 0.5 * sigma**2) * t + sigma * W)

############ Simulation Monte Carlo
t_pred = np.linspace(len(data_2023), len(data_2023) + n_steps, n_steps)
prediction_monte = [SO]
for i in range(1, len(t_pred)):
    prediction_monte.append(
        prediction_monte[-1] * np.exp((r - (sigma**2) / 2) * (1 / 365) + sigma * np.sqrt(1 / 365) * np.random.normal(0, 1))
    )

############ Simulation avec Chaînes de Markov
# Construction de la matrice de transition
n_bins = 10  # Nombre de groupes (discrétisation)
bins = np.linspace(data_2023['Prix_moy'].min(), data_2023['Prix_moy'].max(), n_bins + 1)
labels = range(n_bins)
data_2023['State'] = pd.cut(data_2023['Prix_moy'], bins=bins, labels=labels, include_lowest=True)

transition_matrix = np.zeros((n_bins, n_bins))

# Remplissage de la matrice de transition
for i in range(len(data_2023) - 1):
    current_state = data_2023['State'].iloc[i]
    next_state = data_2023['State'].iloc[i + 1]
    transition_matrix[current_state, next_state] += 1

# Normalisation de la matrice
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# Simulation des états futurs avec la chaîne de Markov
price_future = [SO]
current_state = data_2023['State'].iloc[-1]

for _ in range(n_steps):
    next_state = np.random.choice(labels, p=transition_matrix[current_state])
    next_price = np.mean(bins[next_state:next_state + 2])  # Centre de l'intervalle
    price_future.append(next_price)
    current_state = next_state

# Ajustement de l'échelle temporelle pour la simulation Markov
tm = np.linspace(len(data_2023), len(data_2023) + n_steps, n_steps + 1)

# Tracé des résultats
plt.figure(figsize=(12, 6))
t_hist = np.linspace(0, len(data_2023), len(data_2023))
plt.plot(t_hist, data_2023['Prix_moy'], label='Historique (2023-01-04 à 2024-11-19')
plt.plot(t_pred, prediction_monte, label='Prédiction (Monte Carlo)')
plt.plot(t_pred, prediction_black, label='Prédiction (Black-Scholes)')
plt.plot(tm, price_future, label='Prédiction (Chaînes de Markov)')
plt.xlabel('Temps (jours)')
plt.ylabel('Prix (USD)')
plt.grid()
plt.title('BP - Historique et Prédiction')
plt.legend()
plt.ylim(0, max(data_2023['Prix_moy']) * 1.1) 
plt.show()
