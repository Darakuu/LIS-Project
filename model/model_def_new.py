import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # input (3, 64, 64) output (16, 64, 64)
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # output (16, 32, 32)

            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # output (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output (32, 16, 16)
        )

        # 22 letters in the alphabet, 22 classes in output in one fc layer

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20), # dropout layer
            nn.Linear(32 * 16 * 16, 128), # input (16 * 32 * 32) output (22)
            nn.ReLU(),
            nn.Linear(128, 22)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x

    def test(self, predictions,labels):
        self.eval()
        correct = 0
        for p,l in zip(predictions,labels):
            if p==l:
                correct+=1
        acc = correct/len(predictions)
        return(acc, correct, len(predictions))

    def evaluate(self, predictions,labels):
        correct = 0
        for p,l in zip(predictions,labels):
            if p==l:
                correct+=1
        acc = correct/len(predictions)
        return(acc)