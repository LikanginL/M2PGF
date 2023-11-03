import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
class MultiTask(nn.Module):
    def __init__(self, input_size, num_tasks,model_path):
        super(MultiTask, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.load_state_dict(torch.load(model_path))
        print("load_resnet50")
        self.resnet.fc = nn.Identity()
        self.batch_norm=nn.BatchNorm1d(input_size)
        self.linear=nn.Linear(input_size,num_tasks,bias=False)
        self.dropout1=nn.Dropout(0.1)
    def forward(self, x):
        x = self.resnet(x)
       # x=self.dropout1(x)
        x=self.batch_norm(x)
        out=self.linear(x)
        return torch.sigmoid(out)
        