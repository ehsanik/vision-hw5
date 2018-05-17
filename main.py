from utils import argParser
from dataloader import CifarLoader
import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import pdb


def train(net, dataloader, optimizer, criterion, epoch):

    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(dataloader.trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 2000 == 0:    # print every 2000 mini-batches
            net.log('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    net.log('Final Summary:   loss: %.3f' %
          (total_loss / i))


def test(net, dataloader, tag=''):
    correct = 0
    total = 0
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.testloader
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.log('%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        net.log('%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))

def main():

    args = argParser()

    cifarLoader = CifarLoader(args)
    net = args.model()
    print('The log is recorded in ')
    print(net.logFile.name)

    criterion = net.criterion()
    optimizer = net.optimizer()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, cifarLoader, optimizer, criterion, epoch)
        if epoch % 1 == 0: # Comment out this part if you want a faster training
            test(net, cifarLoader, 'Train')
            test(net, cifarLoader, 'Test')


    print('The log is recorded in ')
    print(net.logFile.name)

if __name__ == '__main__':
    main()
