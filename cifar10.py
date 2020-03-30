import os
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device: ', device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = os.path.join(os.environ['INPUT_PATH'], 'cifar10')
trainset = torchvision.datasets.CIFAR10(data_path, train=True,  download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(data_path, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
n_epochs = 16

for epoch in range(n_epochs):

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print('[epoch %d / %d] loss: %.3f' % (epoch + 1, n_epochs, running_loss / i))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

print('Saving model to TMP_OUTPUT_PATH')
checkpoint_path = os.path.join(os.environ['TMP_OUTPUT_PATH'], './cifar_model.pth')
torch.save(model.state_dict(), checkpoint_path)
