import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm


class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_var = nn.Parameter(torch.zeros(out_features))
        self.relu = nn.ReLU()

    def forward(self, x):
        device = x.device
        weight_mu = self.weight_mu.to(device)
        weight_log_var = self.weight_log_var.to(device)
        bias_mu = self.bias_mu.to(device)
        bias_log_var = self.bias_log_var.to(device)

        weight_std = torch.exp(0.5 * weight_log_var)
        bias_std = torch.exp(0.5 * bias_log_var)
        weight = weight_mu + weight_std * torch.randn_like(weight_std, device=device)
        bias = bias_mu + bias_std * torch.randn_like(bias_std, device=device)

        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, weight.T) + bias.unsqueeze(0))

    def kl_loss(self):
        return 0.5 * (torch.sum(self.weight_mu ** 2 + torch.exp(self.weight_log_var) - self.weight_log_var - 1) +
                      torch.sum(self.bias_mu ** 2 + torch.exp(self.bias_log_var) - self.bias_log_var - 1))


def MNIST_loaders(train_batch_size=50000*4, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class BayesianFFNN(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([BayesianLayer(dims[d], dims[d + 1]) for d in range(len(dims) - 1)])

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(h.pow(2).mean(1))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_layer(self, x_pos, x_neg):
        for layer in self.layers:
            optimizer = Adam(layer.parameters(), lr=0.03)
            threshold = 1.0
            num_epochs = 10000
            for _ in tqdm(range(num_epochs)):
                g_pos = layer(x_pos).pow(2).mean(1)
                g_neg = layer(x_neg).pow(2).mean(1)
                loss = torch.log(1 + torch.exp(torch.cat([-g_pos + threshold, g_neg - threshold]))).mean()
                kl = layer.kl_loss()
                total_loss = loss + (1.0 / x_pos.size(0)) * kl

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            x_pos, x_neg = layer(x_pos).detach(), layer(x_neg).detach()


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = MNIST_loaders()

    net = BayesianFFNN([784, 500, 500]).to(device)
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)

    net.train_layer(x_pos, x_neg)

    # Calculate and print train accuracy
    train_accuracy = net.predict(x).eq(y).float().mean().item()
    print('Train accuracy:', train_accuracy)

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

    # Calculate and print test accuracy
    test_accuracy = net.predict(x_te).eq(y_te).float().mean().item()
    print('Test accuracy:', test_accuracy)
