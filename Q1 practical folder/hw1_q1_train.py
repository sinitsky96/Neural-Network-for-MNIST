import torch
import torchvision
from matplotlib import pyplot as plt
import math

from torchvision.transforms import transforms


##### Utils functions
def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def sigmoidPrime(s):
    # derivative of sigmoid
    # s: sigmoid output
    return s * (1 - s)


def softmax(s):
    # s: torch tensor
    # return: torch tensor with softmax applied
    return torch.exp(s) / torch.sum(torch.exp(s), dim=1).view(-1, 1)


def cross_entropy_loss(y_hat, y):
    # y_hat: torch tensor, output of the network
    # y: torch tensor, labels
    # return: torch tensor, cross entropy loss
    return -torch.sum(y * torch.log(y_hat)) / y.size(0)


class Neural_Network:
    def __init__(self, input_size=2, output_size=1, hidden_size=6):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights and biases for two layers
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)

    def forward(self, X):
        """
        using a sigmoid activation function then a softmax activation function
        :param X: input data
        :return: output of the network
        """
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax(self.z2)

    def backward(self, X, y, y_hat, lr=.1):
        """
        backpropagation algorithm to update the weights
        :param X: input data
        :param y:  labels
        :param y_hat:  output of the network
        :param lr:  learning rate
        """
        batch_size = y.size(0)
        dl_dz2 = (1 / batch_size) * (y_hat - y)

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * sigmoidPrime(self.h)
        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))
        #weighted learning rate



    def train(self, data, cur_epoch, tot_epochs, lr=.1):
        """
        train the network for one epoch
        :param softmax: input data
        :param cur_epoch: current epoch
        :param tot_epochs:  total number of epochs
        :param lr:  learning rate
        """
        for i, (images, labels) in enumerate(data):
            # transform the labels to one hot encoding
            labels = torch.zeros(labels.size(0), 10).scatter_(1, labels.view(-1, 1), 1)
            # reshape the images to a vector
            images = images.view(-1, 28 * 28)
            # forward pass
            y_hat = self.forward(images)
            loss = cross_entropy_loss(y_hat, labels)
            # backward pass
            self.backward(images, labels, y_hat, lr)
            # print the loss every 1000 iterations
            if i % 1000 == 0:
                print(f"Epoch {cur_epoch}/{tot_epochs} - Loss: {loss.item()}")
        return loss.item()

    def test(self, data):
        """
        test the network
        :param data: test data
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data:
                # reshape the images to a vector
                images = images.view(-1, 28 * 28)
                # forward pass
                y_hat = self.forward(images)
                # get the predicted class
                _, predicted = torch.max(y_hat, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {round(correct / total, 3)}")
        return correct / total

    def save(self, path):
        torch.save({"W1": self.W1, "W2": self.W2, "b1": self.b1, "b2": self.b2}, path)

    def load(self, path):
        return torch.load(path)


def Q1():
    input_size = 784
    num_classes = 10
    num_epochs = 50
    batch_size = 128
    learning_rate = 0.22
    # load the data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # create the network
    model = Neural_Network(input_size, num_classes, 100)
    # train the network
    train_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        model.train(train_loader, epoch, num_epochs, learning_rate)
       # model.test(train_loader)
        #model.test(test_loader)
        train_acc.append(model.test(train_loader))
        test_acc.append(model.test(test_loader))
    print(f"Train accuracy: {train_acc[-1]}")
    print(f"Test accuracy: {test_acc[-1]}")
    # save the model
    model.save("model_q1.pkl")
    # plot the accuracy
    plt.plot(test_acc, label="Test")
    plt.plot(train_acc, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    Q1()
