import torch 
class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.layer1 = torch.nn.Linear(11, 16)
        self.sigmoid = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(16, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

