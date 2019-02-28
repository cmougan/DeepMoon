from random import randint
import numpy as np
import json
import sys

# PyTorch stuff

import torch
from torch.autograd import Variable
import classifier

np.set_printoptions(suppress=True,
                    linewidth=np.nan,
                    threshold=np.nan,
                    precision=1)

torch.set_printoptions(threshold=10000,
                       linewidth=10000,
                       precision=1)

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

def get_problem_tensor(problem):
    #pprint(problem['Moves'])
    tensor = torch.zeros(1, 32, 32)
    for move in problem['Moves']:
        v = (1/3) if move['IsStart'] else (1 if move['IsEnd'] else (2/3))
        x = ord(move['Description'][0].lower()) - 96 # 1-index
        y = int(move['Description'][1:]) # 1-index
        #print(move['Description'], x, y, v)
        tensor[0, y - 1, x - 1] = v
    return tensor

def get_problem_grade(problem):
    return grade_names.index(problem['Grade'])

modo='fixed'
if modo=='fixed':
  s = 1 / 3
  h = 2 / 3
  t = 1

             # A  B  C  D  E  F  G  H  I  J  K

  problem = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], # 18
             [ 0, 0, 0, t, 0, 0, 0, 0, 0, 0, 0 ], # 17
             [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], # 16
             [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], # 15
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 14
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 13
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 12
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 11
             [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], # 10
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 9
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 8
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 7
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 6
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 5
             [ 0, 0, 0, h, 0, 0, 0, h, 0, 0, 0 ], # 4
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 3
             [ 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0 ], # 2
             [ 0, 0, 0, s, s, 0, 0, 0, 0, 0, 0 ]] # 1

  ground_truth_grade = '8B'
  ground_truth_index=15


  array = np.array(np.flip(problem, axis=0))
  array_1x32x32 = np.zeros((1, 32, 32))
  array_1x32x32[0, 0:18, 0:11] = array
  tensor = torch.FloatTensor(array_1x32x32)
  tensor.unsqueeze_(0)
if modo=='random':
# Read data
  json_data = []

  for arg in sys.argv[2:]:
      with open(arg) as input_file:
          json_data += json.load(input_file)
  index = randint(0, len(json_data))
  print("Index =", index)
  problem = json_data[index]
  tensor = get_problem_tensor(problem)
  tensor.unsqueeze_(0)
  ground_truth_index = get_problem_grade(problem)
  
# Load model from file

checkpoint = torch.load(sys.argv[1])
model = classifier.DeepMoonClassifier(num_classes=n_grades)
model.load_state_dict(checkpoint)

# Test model

model.eval()
inputs = Variable(tensor)
output = model(inputs)
index = output.data.numpy().argmax()
if modo=='random':
  print('Ground truth grade =', grade_names[ground_truth_index])
  print('Predicted grade =', grade_names[index])
  print('Accuracy =', 1.0 - abs(ground_truth_index - index) / n_grades)
if modo=='fixed':
  print(index)
  print('Ground truth grade =', ground_truth_grade)
  print('Predicted grade =', grade_names[index])
  print('Accuracy =', 1.0 - abs(ground_truth_index - index) / n_grades)

