import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable


class generateur(nn.Module):
    
    def __init__(self,input_length:int,middle_length:int, output_length:int):
        super(generateur,self).__init__()
        self.layer1 = nn.Linear(int(input_length), int(middle_length),bias=True)
        self.layer2 = nn.Linear(int(middle_length),int(middle_length),bias=True)
        self.layer3 = nn.Linear(int(middle_length),int(middle_length),bias=True)
        self.layer4= nn.Linear(int(middle_length),int(middle_length),bias = True)
        self.layer5= nn.Linear(int(middle_length),int(middle_length),bias = True)
        self.layer6= nn.Linear(int(middle_length),int(middle_length),bias = True)
        self.layer7= nn.Linear(int(middle_length),int(middle_length),bias = True)
        self.layer8= nn.Linear(int(middle_length),int(output_length),bias = True)
        
    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer7(x)
        x = F.relu(x)
        x = self.layer8(x)
        return x

class discriminateur(nn.Module):
    def __init__(self, input_length: int, middle_length:int):
        super(discriminateur, self).__init__()
        self.layer1 = nn.Linear(int(input_length), int(middle_length), bias=True)
        self.layer2 = nn.Linear(int(middle_length), int(middle_length),bias = True)
        self.layer3 = nn.Linear(int(middle_length), int(middle_length),bias = True)
        self.layer4 = nn.Linear(int(middle_length), int(middle_length),bias = True)
        self.layer5 = nn.Linear(int(middle_length), int(middle_length),bias = True)
        self.layer6 = nn.Linear(int(middle_length), int(middle_length),bias = True)
        self.layer7 = nn.Linear(int(middle_length), int(middle_length),bias = True)
        self.layer8 = nn.Linear(int(middle_length),1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer8(x)
        return x


class generateur_small(nn.Module):
    
    def __init__(self,input_length:int,middle_length:int, output_length:int):
        super(generateur_small,self).__init__()
        self.layer1 = nn.Linear(int(input_length), int(middle_length),bias=True)
        self.layer2 = nn.Linear(int(middle_length),int(middle_length),bias=True)
        self.layer3 = nn.Linear(int(middle_length),int(middle_length),bias=True)
        self.layer4= nn.Linear(int(middle_length),int(output_length),bias = True)
        
    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        return x

class discriminateur_small(nn.Module):
    
    def __init__(self,input_length:int,middle_length:int):
        super(discriminateur_small,self).__init__()
        self.layer1 = nn.Linear(int(input_length), int(middle_length),bias=True)
        self.layer2 = nn.Linear(int(middle_length),int(middle_length),bias=True)
        self.layer3 = nn.Linear(int(middle_length),int(middle_length),bias=True)
        self.layer4 = nn.Linear(int(middle_length),1)
        
    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        return x

