# Neural Network within limited Training time that maximize the accuracy on the CIFAR-10 test-set
The goal of this project is to improve a neural network's performance on the CIFAR10 dataset under a strict 10-minute training constraint, focusing on architectural optimization and efficient training strategies without using pretrained models.\
## 1. Summary:
### Approach: 
Developed a custom CNN (MyNet) with **residual connections**, **batch normalization**, and **data augmentation**. Trained using **SGD** with **OneCycleLR** scheduler, **mixed-precision training**, and GPU acceleration in PyTorch.\
### Key Results:
 - Achieved **83.84%** test accuracy (**vs. ResNet-18's 73.6%**) 
 - **35%** **fewer parameters** than ResNet-18 (1.8M vs. 2.8M) 
 - Training time: **9.2 minutes for 20 epochs** (T4 GPU) 
 - Stability and training curve also show good performance
## 2.	Baseline vs. Final Version:
### Baseline definition:
The ResNet-18 model provided by the PyTorch official website is used as a comparison baseline without additional data augmentation and preprocessing.
### Final Version (Our Model):
```python
import torch
import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # depthwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),  # pointwise
            nn.BatchNorm2d(64)
        )
        self.residual_act = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.conv1(x)
        identity = x
        res_out = self.residual(x)
        x = self.residual_act(res_out + identity)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
net = MyNet().cuda()
```
o	Add three layers of convolution modules and introduce Batch Normalization and ReLU\
o	Introduce Depthwise Separable Convolution residual branch after the first layer of convolution\
o	Use OneCycleLR to dynamically adjust the learning rate to accelerate model convergence\
o	Use AMP (Automatic Mixed Precision) to improve training efficiency
### Quantitative Comparison:
|Metirc|ResNet-18|Our Model|
|------|---------|----------|
|Final Accuracy|~73.6%|~83.8%|
|Training Time|~360s|~540s|
|Loss at End|~0.05|~0.35|
|Parameters|~2.8M|~1.8M|
|Stability (Fluctuations)|low|High|
### Figures:
ResNet-18:\
<img width="864" height="261" alt="image" src="https://github.com/user-attachments/assets/988df754-6e46-4d8b-be84-fb7feda829e4" />

Our Model (My_Net):\
<img width="864" height="248" alt="image" src="https://github.com/user-attachments/assets/d9a5c5ff-4c7c-4514-b8e0-97b8f876b5b5" />

## 3.	Ablation Study:
| Change Test	|   Hypothesis	|Result|	Analysis|
|-------------|---------------|------|----------|
|With Normalization	|Accelerate model convergence and make the contribution of different features to the loss function more balanced	|Normalize pixel values to a fixed range (such as [0,1] or mean 0 variance 1) to make the optimization process more stable|	Standardization makes gradient updates smoother and avoids local optimality caused by too little noise in large batches|
|With RandomCrop and RandomHorizontalFlip(Data augmentation)|	Random cropping simulates slight translations and helps the model generalize better by exposing it to varied perspectives of the same image.|	Slightly enhance in test accuracy, especially on unseen data. Avoiding model overfitting to the center position of objects.	|Introduces spatial variability, improves the generalization of models. Under a 10-minutes constraint, it leads to low-cost, high-benefit strategy. |
|Increasing the batch size (256→512)| 	Speed up training but may reduce generalization, too large a batch size may cause the model to converge to a sharp minimum and have poor generalization ability.|	After appropriately increasing the learning rate, large batches can achieve similar accuracy, but stronger regularization is required	|Large batches converge quickly but have small gradient noise and slightly worse generalization, while small batches have large noise but better implicit regularization effects|
|Using OneCycleLR	|expected to reduce training efficiency|	Convergence is faster, requiring less epochs|	Dynamic learning rate helps to search quickly and stabilize convergence|
|With AMP	|the training time is expected to be shorter|	The training time decreases by about 20%	|Mixed precision effectively improves training efficiency without affecting so many accuracy|
