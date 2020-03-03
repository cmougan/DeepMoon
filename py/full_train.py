import torch
from torch.utils import data
import numpy as np 
import os.path
import random
import json
import sys
import torch.optim as optim

#Implementing seeds to be able to reproduce the experiments
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.deterministic = True

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()


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

path_data=['../data/2017.json']
for arg in path_data:
    with open(arg) as input_file:
        json_data += json.load(input_file)


random.shuffle(json_data)
#parametro2deentrada=float(sys.argv[2])
train_set_size = int(0.7 * len(json_data))
train_set = MoonboardProblemDataset(json_data[:train_set_size])
valid_set = MoonboardProblemDataset(json_data[train_set_size:])

params = {'batch_size': 128,
          'shuffle': False,
          'num_workers': 4}

train_gen = data.DataLoader(train_set, **params)
valid_gen = data.DataLoader(valid_set, **params)

# Train the Model
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class DeepMoonClassifier(nn.Module):
    def __init__(self, num_classes=n_grades):
        super(DeepMoonClassifier, self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=1, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=64)
        self.unit9 = Unit(in_channels=64, out_channels=64)
        self.unit10 = Unit(in_channels=64, out_channels=64)
        self.unit11 = Unit(in_channels=64, out_channels=64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=64, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)
        self.unit15 = Unit(in_channels=128, out_channels=128)
        self.unit16 = Unit(in_channels=128, out_channels=128)
        

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4,self.unit5,  self.unit6,
                                 self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11,self.pool3,
                                 self.unit12, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=1)
        self.sig = nn.Sigmoid()


    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        output = self.sig(output)
        output = output*17
       
        return output
regression=True

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


if regression:
    loss_fn=nn.MSELoss()
else:
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
        outputs=model(problems)     #The prediction
        
        if regression:
            prediction=torch.round(torch.squeeze(outputs.data))
        else:
            _, prediction = torch.max(outputs.data, 1)
        print(prediction)
        

        #Compute accuracy
        test_acc_float = 1 - (prediction.type(tensor_float_type) - grades.data.type(tensor_float_type)).abs_().type(tensor_float_type) / n_grades
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
            #print('Outputs')
            #print(outputs)
            #print('Grades')
            #print(grades)
            
            
            
            #Compute the loss based on the predictions and actual grades
            #Taking into account wether it runs in the CPU or GPU
            #For Regresion
            if regression:
                if cuda_avail:
                    loss = loss_fn(outputs,grades.type(torch.cuda.FloatTensor))
                else:
                    loss = loss_fn(outputs,grades.type(torch.FloatTensor))
            else:
                loss = loss_fn(outputs,grades)
            #Backpropagate the loss
            loss.backward()
            #Adjust parameters according to the computed gradients
            optimizer.step()

            #Compute accuracy
            train_loss += loss.cpu().item() * problems.size(0)
            
            _, prediction = torch.max(outputs.data, 1)
            
            train_acc_float = 1 - (prediction.type(tensor_float_type) - grades.data.type(tensor_float_type)).abs_().type(tensor_float_type) / n_grades
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

#Trainning
if __name__ == "__main__":
    print('┌───────┬────────────────────┬────────────────────┬────────────────────┬───────┐')
    print('│ Epoch │     Train Loss     │   Train Accuracy   │   Valid Accuracy   │ Saved │')
    print('├───────┼────────────────────┼────────────────────┼────────────────────┼───────┤')
    _, best_hist = train(int(20))
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


    