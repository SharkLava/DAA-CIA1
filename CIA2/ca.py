import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from modules.nn import NN

df = pd.read_csv("dataset/data.csv")
df.drop(["ID", "ZIP Code"], axis=1, inplace=True)
X = df.drop(columns="Personal Loan")
y = df["Personal Loan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

torch_X = torch.from_numpy(np.array(X_train)).to(torch.float32)
torch_y = torch.from_numpy(np.array(y_train)).to(torch.float32)
data = TensorDataset(torch_X, torch_y)
data = DataLoader(data, batch_size=4, shuffle=True)
model = NN()


def fitness_function(model):
    y_pred = model(torch_X)
    y_pred = torch.where(y_pred >= 0.5, 1, 0).flatten()
    accuracy = (y_pred == torch_y).sum().float().item() / len(data.dataset)

    return accuracy


def crossover_mutation(parent1, parent2):
    shape = [i.numpy().shape for i in parent1.parameters()]
    size = [i[0] * i[1] if len(i) == 2 else i[0] for i in shape]
    param_1 = np.concatenate([i.numpy().flatten() for i in parent1.parameters()])
    param_2 = np.concatenate([i.numpy().flatten() for i in parent2.parameters()])

    # Crossover
    start = len(param_1) // 2 - 10
    end = len(param_1) // 2 + 10
    mid_value = random.randrange(start, end)

    original_child_1 = np.concatenate([param_1[:mid_value], param_2[mid_value:]])
    original_child_2 = np.concatenate([param_2[:mid_value], param_1[mid_value:]])

    # Child 1 Mutation
    random_start = random.randrange(0, len(param_1) // 2)
    random_end = random.randrange(random_start, len(param_1))

    mutated_child_1 = original_child_1.copy()
    mutated_child_1[random_start:random_end] = mutated_child_1[random_start:random_end][
        ::-1
    ]

    # Child 2 Mutation
    random_start = random.randrange(0, len(param_1) // 2)
    random_end = random.randrange(random_start, len(param_1))

    mutated_child_2 = original_child_2.copy()
    mutated_child_2[random_start:random_end] = mutated_child_2[random_start:random_end][
        ::-1
    ]

    # Converting the array to parameters
    children = [original_child_1, original_child_2, mutated_child_1, mutated_child_2]
    output = []

    for child in children:
        param = []
        sum = 0
        for i in range(len(size)):
            array = child[sum : sum + size[i]]
            array = array.reshape(shape[i])
            sum += size[i]
            param.append(array)
        param = np.array(param, dtype="object")
        output.append(param)

    output = np.array(output, dtype="object")
    return output


# Define the knowledge sources
def domain_knowledge(net):
    # Example of biasing the search towards positive weights
    net.fc1.weight.data.clamp_(min=0)
    net.fc2.weight.data.clamp_(min=0)


def experience_knowledge(net, best_net):
    # Example of using the best individual from previous generation
    net.fc1.weight.data += 0.1 * best_net.fc1.weight.data
    net.fc2.weight.data += 0.1 * best_net.fc2.weight.data


# Define the fitness function
def evaluate_fitness(net, dataloader, objective_function):
    fitness = 0.0
    for inputs, targets in dataloader:
        outputs = net(inputs)
        fitness += objective_function(outputs, targets)
    return fitness


# Define the evolutionary process
def evolutionary_process(
    population,
    dataloader,
    objective_function,
    domain_knowledge,
    experience_knowledge,
    evaluate_fitness,
    selection_rate=0.5,
    mutation_rate=0.1,
):
    # Evaluate the fitness of each individual
    fitnesses = []
    for net in population:
        fitness = evaluate_fitness(net, dataloader, objective_function)
        fitnesses.append(fitness)
    best_net_idx = torch.tensor(fitnesses).argmin()
    best_net = population[best_net_idx]

    # Select individuals for reproduction based on their fitness
    sorted_idx = torch.tensor(fitnesses).argsort()
    selected_idx = sorted_idx[: int(len(population) * selection_rate)]

    # Generate new individuals through genetic operators
    new_population = []
    for idx in selected_idx:
        new_net = Net()
        new_net.fc1.weight.data = population[idx].fc1.weight.data.clone()
        new_net.fc2.weight.data = population[idx].fc2.weight.data.clone()
        new_net.fc1.bias.data = population[idx].fc1.bias.data.clone()
        new_net.fc2.bias.data = population[idx].fc2.bias.data.clone()

        # Incorporate domain knowledge into mutation
        for param in new_net.parameters():
            param.data += torch.randn_like(param.data) * mutation_rate
            domain_knowledge(new_net)
            experience_knowledge(new_net, best_net)

        new_population.append(new_net)

    return new_population, best_net


def train(epoch):
    population = np.array([NN() for i in range(population_size)])
    best_model = None
    for i in range(epoch):
        population = population[
            np.argsort([fitness_function(model) for model in population])
        ]

        # Printing Max Accuracy
        best_model = population[-1]
        print("Epoch", i, " :", fitness_function(population[-1]))

        first = population[-1]
        second = population[-2]
        last = population[-3]

        output_1 = crossover_mutation(first, second)
        output_2 = crossover_mutation(first, last)
        output = np.concatenate([output_1, output_2])
        new_population = np.array([NN() for i in range(len(output))])
        for count, model in enumerate(new_population, 0):
            for index, param in enumerate(model.parameters(), 0):
                param.data = torch.tensor(output[count][index])

        new_population = np.concatenate([new_population, [first, second]])
        population = new_population.copy()

    return best_model


population_size = 30
torch.set_grad_enabled(False)
best_model = train(200)

test_x = torch.from_numpy(np.array(X_test)).to(torch.float32)
test_y = torch.from_numpy(np.array(y_test)).to(torch.float32)
test = TensorDataset(test_x, test_y)
test = DataLoader(test, batch_size=1)
y_pred = best_model(test_x)
y_pred = torch.where(y_pred >= 0.5, 1, 0).flatten()
print(classification_report(y_pred, test_y))
