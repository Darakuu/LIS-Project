import torch.nn as nn
import torch.nn.functional as F


class SignLanguage_DeepCNN(nn.Module):
    def __init__(self):
        super(SignLanguage_DeepCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #input: 1x64x64, output: 16x64x64
            nn.ReLU(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # input: 16x64x64, output: 32x64x64
            nn.MaxPool2d(2, 2), #input: 32x64x64, output: 32x32x32
            nn.ReLU(),

            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #input: 32x32x32, output: 64x32x32
            nn.MaxPool2d(2, 2),  # input: 64x32x32, output: 64x16x16
            nn.ReLU(),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #input: 64x16x16, output: 64x16x16
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #input: 64x16x16, output: 128x16x16
            nn.MaxPool2d(2, 2), #input: 128x16x16, output: 128x8x8
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(128 * 8 * 8, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 22, bias=True),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x.view(x.shape[0],-1))
        F.softmax(x, dim=1)
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
