# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from modules.nn import NN
from pyswarms.single import GlobalBestPSO

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


# PSO
torch.set_grad_enabled(False)

model = NN()
param = np.concatenate([i.numpy().flatten() for i in model.parameters()])
shape = [i.numpy().shape for i in model.parameters()]
size = [i[0] * i[1] if len(i) == 2 else i[0] for i in shape]


def objective_function(particles, shape=shape, size=size):
    accuracy = []
    output = []

    for particle in particles:
        param = list()
        cum_sum = 0
        for i in range(len(size)):
            array = particle[cum_sum : cum_sum + size[i]]
            array = array.reshape(shape[i])
            cum_sum += size[i]
            param.append(array)
        param = np.array(param, dtype="object")
        output.append(param)

    for i in range(len(output)):
        # Copy Weights and Biases
        model = NN()
        for idx, wei in enumerate(model.parameters()):
            wei.data = (torch.tensor(output[i][idx])).to(torch.float32)

        y_pred = model(torch_X)
        y_pred = torch.where(y_pred >= 0.5, 1, 0).flatten()
        acc = (y_pred == torch_y).sum().float().item() / len(data.dataset)
        accuracy.append(1 - acc)
    return accuracy


# Tunable Parameters

options = {"c1": 0.8, "c2": 0.3, "w": 0.3}
dim = len(param)
x_max = 1.0 * np.ones(dim)
x_min = -1.0 * x_max
bounds = (x_min, x_max)
pso = GlobalBestPSO(n_particles=900, dimensions=209, options=options, bounds=bounds)
best_cost, best_parameters = pso.optimize(objective_function, iters=50)
print("Accuracy : ", 1 - best_cost)

result = []
for par in [best_parameters]:
    param = list()
    cum_sum = 0
    for i in range(len(size)):
        array = par[cum_sum : cum_sum + size[i]]
        array = array.reshape(shape[i])
        cum_sum += size[i]
        param.append(array)
    param = np.array(param, dtype="object")
    result.append(param)

best_model = NN()
for idx, wei in enumerate(best_model.parameters()):
    wei.data = (torch.tensor(result[0][idx])).to(torch.float32)

y_pred = best_model(torch_X)
y_pred = torch.where(y_pred >= 0.5, 1, 0).flatten()
acc = (y_pred == torch_y).sum().float().item() / len(data.dataset)
print("Accuracy : ", acc)


# ## Testing
test_x = torch.from_numpy(X_test).to(torch.float32)
test_y = torch.from_numpy(y_test).to(torch.float32)
test = TensorDataset(test_x, test_y)
test = DataLoader(test, batch_size=1)
y_pred = best_model(test_x)
y_pred = torch.where(y_pred >= 0.5, 1, 0).flatten()
print(classification_report(y_pred, test_y))
