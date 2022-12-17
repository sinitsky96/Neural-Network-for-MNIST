import torch
import torchvision
import torchvision.transforms as transforms
from hw1_q1_train import Neural_Network


def evaluate_hw1():
    # parameters
    input_size = 784
    num_classes = 10
    num_epochs = 5
    batch_size = 128
    learning_rate = 0.15
    # load MNIST data
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # load model
    model = Neural_Network(input_size, num_classes)
    trained_weights = model.load('model_q1.pkl')
    model.W1, model.b1 = trained_weights['W1'], trained_weights['b1']
    model.W2, model.b2 = trained_weights['W2'], trained_weights['b2']
    # evaluate model
    return sum([model.test(test_loader) for _ in range(num_epochs)]) / num_epochs


if __name__ == '__main__':
    print(round(evaluate_hw1(), 2), '%')

