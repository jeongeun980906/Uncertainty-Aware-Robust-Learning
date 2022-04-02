import torchvision
import torch
from torch import optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from core.backbones.resnet import *
from dataloader.clothing1M import *

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,14)

transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ])    
transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
    ])

BATCH_SIZE = 64
train = clothing1M(
    root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
    transform   = transform_train,
    mode        = 'train',
    num_samples = 1000*BATCH_SIZE
)
val = clothing1M(

    root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
    transform   = transform_val,
    mode        = 'test'
)

EPOCH = 50
DEVICE = 'cuda:6' 
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-3)
#optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
model.to(DEVICE)

train_loader = torch.utils.data.DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
valid_loader = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)


for epoch in range(EPOCH):
    model.train()
    print(f'Train Epoch :: {epoch}')
    train_correct = list()
    for i,(x,y) in enumerate(train_loader):
        if i%100==0:
            print(i)
        out = model(x.to(DEVICE))
        #print( list((torch.argmax(out, dim=1)==y.to(DEVICE)).detach().cpu().numpy()) )
        loss = criterion(out,y.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_correct += list((torch.argmax(out, dim=1)==y.to(DEVICE)).detach().cpu().numpy())

    correct  = list()
    print('eval')
    model.eval()
    with torch.no_grad():
        for x,y in valid_loader:
            out = model(x.to(DEVICE))
            correct += list((torch.argmax(out, dim=1)==y.to(DEVICE)).detach().cpu().numpy())

    print(f'Epoch {epoch+1}  \
            train acc : {sum(train_correct)/len(train_correct)} \
            test acc : {sum(correct)/len(correct)}')

