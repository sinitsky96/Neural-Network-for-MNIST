import torch
import torchvision
import torchvision.transforms as transforms
from hw1_train_q2 import Neural_Network


def evaluate_hw1():
    input_size = 784
    num_classes = 10
    num_epochs = 50
    batch_size = 128
    learning_rate = 0.2
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_dataset.data = test_dataset.data[:128]
    # apply labels 0 and 1 to test_dataset randomly from a bernoulli distribution with p=0.5
    test_dataset.targets = torch.bernoulli(torch.ones(128) * 0.5)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = Neural_Network(input_size, num_classes)
    weights = model.load('model_q2.pkl')
    model.W1, model.b1 = weights['W1'], weights['b1']
    model.W2, model.b2 = weights['W2'], weights['b2']
    return sum([model.test(test_loader) for _ in range(num_epochs)]) / num_epochs


if __name__ == "__main__":
    print(evaluate_hw1())
