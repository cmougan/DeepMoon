import torch
from torch.utils import data

grade_names = ['5+',
               '6A', '6A+',
               '6B', '6B+',
               '6C', '6C+',
               '7A', '7A+',
               '7B', '7B+',
               '7C', '7C+',
               '8A', '8A+',
               '8B', '8B+']

class MoonboardProblemDataset(data.Dataset):

    start = 1 / 3
    hold = 2 / 3
    top = 1

    def __init__(self, problem_list, tensor_h=32, tensor_w=32):
        'Initialization'
        self.problem_list = problem_list
        self.tensor_h = tensor_h
        self.tensor_w = tensor_w

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.problem_list)

    def get_problem_tensor(self, problem):

        tensor = torch.zeros(1, self.tensor_h, self.tensor_w)
        for move in problem['Moves']:
            v = self.start if move['IsStart'] else (self.top if move['IsEnd'] else self.hold)
            x = ord(move['Description'][0].lower()) - 96 # 1-index
            y = int(move['Description'][1:]) # 1-index
            tensor[0, y - 1, x - 1] = v
        return tensor

    def get_problem_grade(self, problem):
        return grade_names.index(problem['Grade'])

    def __getitem__(self, index):
        'Generates one sample of data'
        problem = self.problem_list[index]
        tensor = self.get_problem_tensor(problem)
        grade = self.get_problem_grade(problem)
        return tensor, grade
