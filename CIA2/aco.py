import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam

df = pd.read_csv("/home/ghoul/Documents/projects/python/DAA-CIA1/CIA2/dataset/data.csv")
df = df = df = df[
    [
        "Age",
        "Experience",
        "Income",
        "ZIP Code",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
        "Personal Loan",
    ]
]
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
import numpy as np


class AntColonyOptimizer:
    def __init__(
        self,
        cost_func,
        dimensions,
        colony_size,
        min_values,
        max_values,
        iterations,
        alpha=1,
        beta=1,
        rho=0.1,
    ):
        self.cost_func = cost_func
        self.dimensions = dimensions
        self.colony_size = colony_size
        self.min_values = min_values
        self.max_values = max_values
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.best_params = None
        self.best_score = float("-inf")

    def initialize_pheromone_trails(self):
        self.pheromone_trails = (
            np.ones((self.dimensions, self.colony_size)) / self.dimensions
        )

    def run(self):
        self.initialize_pheromone_trails()
        for i in range(self.iterations):
            solutions = self.generate_solutions()
            scores = np.array([self.cost_func(sol) for sol in solutions])
            best_index = np.argmax(scores)
            if scores[best_index] > self.best_score:
                self.best_score = scores[best_index]
                self.best_params = solutions[best_index]
            self.update_pheromone_trails(solutions, scores)
        return self.best_params, self.best_score

    def generate_solutions(self):
        solutions = []
        for ant in range(self.colony_size):
            solution = []
            for dim in range(self.dimensions):
                prob = (
                    self.pheromone_trails[dim] ** self.alpha
                    * (1.0 / (self.max_values[dim] - self.min_values[dim])) ** self.beta
                )
                prob /= np.sum(prob)
                value = np.random.choice(np.arange(self.colony_size), p=prob)
                solution.append(value)
            solutions.append(solution)
        return solutions

    def update_pheromone_trails(self, solutions, scores):
        delta_pheromone_trails = np.zeros((self.dimensions, self.colony_size))
        for i, solution in enumerate(solutions):
            for j, value in enumerate(solution):
                delta_pheromone_trails[j][value] += scores[i]
        self.pheromone_trails = (
            self.rho * self.pheromone_trails + delta_pheromone_trails
        )


def cost_function(params):
    neurons = int(params[0])
    learning_rate = params[1]
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(1, activation="softmax"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=10, batch_size=10)
    _, val_acc = model.evaluate(X_val, y_val)
    return val_acc


aco = AntColonyOptimizer(
    cost_function,
    dimensions=2,
    colony_size=10,
    min_values=[1, 0.001],
    max_values=[100, 0.1],
    iterations=20,
    alpha=1,
    beta=1,
    rho=0.1,
)
best_params, best_score = aco.run()

neurons = int(best_params[0])
learning_rate = best_params[1]
model = Sequential()
model.add(Dense(neurons, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(1, activation="softmax"))
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)
model.fit(X_train, y_train, epochs=10, batch_size=10)
test_acc = model.evaluate(X_test, y_test)

print("Best accuracy: {test_acc}")
