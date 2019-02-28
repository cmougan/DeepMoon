import torch
from torch.utils import data
import numpy as np 
np.random.seed(999)
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True

import os.path
import random
import json
import sys
import torch.optim as optim


np.seterr(divide='ignore', invalid='ignore')

grade_names = ['5+',
               '6A', '6A+',
               '6B', '6B+',
               '6C', '6C+',
               '7A', '7A+',
               '7B', '7B+',
               '7C', '7C+',
               '8A', '8A+',
               '8B', '8B+']

n_grades = len(grade_names)

save_directory = os.path.join('..', 'net')

# Create Dataset

from dataset import MoonboardProblemDataset
json_data = []
print(sys.argv[3:])
for arg in sys.argv[3:]:
    with open(arg) as input_file:
        json_data += json.load(input_file)

random.shuffle(json_data)

train_set_size = int(float(sys.argv[2]) * len(json_data))
train_set = MoonboardProblemDataset(json_data[:train_set_size])
valid_set = MoonboardProblemDataset(json_data[train_set_size:])

params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 4}

train_gen = data.DataLoader(train_set, **params)
valid_gen = data.DataLoader(valid_set, **params)

# Train the Model

from torch.optim import Adam
from classifier import DeepMoonClassifier
from torch.autograd import Variable
import torch.nn as nn

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = DeepMoonClassifier(num_classes=n_grades)
tensor_float_type = torch.FloatTensor

#if cuda is available, move the model to the GPU
if cuda_avail:
    model.cuda()
    tensor_float_type = torch.cuda.FloatTensor
    # Uncomment for multi-GPU parallelism
    #model = nn.DataParallel(model)

#Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_models(epoch):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    torch.save(model.state_dict(), os.path.join(save_directory, 'classifier_{}.model'.format(epoch)))

def test():
    model.eval()
    test_acc = 0.0
    grade_hist = np.zeros((n_grades,), dtype=np.int)
    acc_hist = tensor_float_type(n_grades).zero_()

    for i, (problems, grades) in enumerate(valid_gen):
        if cuda_avail:
            problems = Variable(problems.cuda())
            grades = Variable(grades.cuda())

        #Predict classes using problems from the test set
        outputs = model(problems)
        _, prediction = torch.max(outputs.data, 1)
        #print('pr', prediction)
        #print('gt', grades.data)

        #Compute accuracy
        test_acc_float = 1 - (prediction - grades.data).abs_().type(tensor_float_type) / n_grades
        acc_hist.index_add_(0, grades, test_acc_float)
        local_hist, _ = np.histogram(grades.cpu(), bins=range(0, n_grades + 1))
        grade_hist += local_hist
        test_acc += test_acc_float.sum().item()

    #Compute the average acc and loss over all test problems
    test_acc = test_acc / len(valid_set)
    acc_hist = acc_hist.cpu().numpy() / grade_hist
    #print(grade_hist)
    #print(acc_hist)

    return test_acc, acc_hist

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0

        for i, (problems, grades) in enumerate(train_gen):
            #Move problems and grades to gpu if available
            if cuda_avail:
                problems = Variable(problems.cuda())
                grades = Variable(grades.cuda())

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using problems from the test set
            outputs = model(problems)
            #Compute the loss based on the predictions and actual grades
            loss = loss_fn(outputs,grades)
            #Backpropagate the loss
            loss.backward()
            #Adjust parameters according to the computed gradients
            optimizer.step()

            #Compute accuracy
            train_loss += loss.cpu().item() * problems.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc_float = 1 - (prediction - grades.data).abs_().type(tensor_float_type) / n_grades
            train_acc += train_acc_float.sum().item()

        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all training problems
        train_acc = train_acc / len(train_set)
        train_loss = train_loss / len(train_set)

        #Evaluate on the test set
        test_acc, acc_hist = test()

        # Print the metrics
        print('│ %05d │ %.16f │ %.16f │ %.16f ' % (epoch, train_loss, train_acc, test_acc), end='')

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            print('│   *   │')
            save_models(epoch)
            best_acc = test_acc
            best_hist = acc_hist
        else:
            print('│       │')

    return best_acc, best_hist

if __name__ == "__main__":
    print('┌───────┬────────────────────┬────────────────────┬────────────────────┬───────┐')
    print('│ Epoch │     Train Loss     │   Train Accuracy   │   Valid Accuracy   │ Saved │')
    print('├───────┼────────────────────┼────────────────────┼────────────────────┼───────┤')
    _, best_hist = train(int(sys.argv[1]))
    print('└───────┴────────────────────┴────────────────────┴────────────────────┴───────┘')
    print('')
    print('┌───────┬────────────────────┐')
    print('│ Grade │      Accuracy      │')
    print('├───────┼────────────────────┤')
    result=[]

    for i in range(0, n_grades):
        print('│ % 5s │ %.16f │' % (grade_names[i], best_hist[i]))
        result.append(best_hist[i])
    print('└───────┴────────────────────┘')
    print(np.mean(result))
    
