import torch

#Model definition
class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(path='digit_model.pt'):
    model = DigitClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
