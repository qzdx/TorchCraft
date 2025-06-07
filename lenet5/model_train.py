import time
import pandas as pd
from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet
import torch
import torch.nn as nn
from torchinfo import summary
import copy
def train_val_data_process():
    train_data = FashionMNIST(root="./data",
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                          download=True)
    train_data , val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=4)
    val_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=4)
    return train_dataloader , val_dataloader
def train_model_process(model ,train_dataloader , val_dataloader , num_epoches):
    device = torch.device("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    model = model.to(device) #模型放入设备中

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epoches):
        print("Epoch {}/{}".format(epoch , num_epoches-1))
        print("-"*10)

        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0

        for step, (b_x , b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train() # model有forward， 没有train目前
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)

            loss = criterion(output , b_y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()*b_x.size(0)

            train_corrects += torch.sum(pre_lab == b_y.data)
            
            train_num += b_x.size(0)

        for step, (b_x , b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval() # model有forward， 没有train目前 , eval更没有
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)

            loss = criterion(output , b_y)

            val_loss += loss.item()*b_x.size(0)

            val_corrects += torch.sum(pre_lab == b_y.data)
            
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num) #一轮训练的loss
        train_acc_all.append(train_corrects.float().item() / train_num)
        val_loss_all.append(val_loss / val_num) #一轮验证的loss
        val_acc_all.append(val_corrects.float().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch , train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch , val_loss_all[-1], val_acc_all[-1]))
        
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("训练耗时{:.0f}m{:.0f}".format(time_use//60,time_use%60))

    torch.save(best_model_wts,'./best.pth')
    train_process = pd.DataFrame(data={"epoch":range(num_epoches),
                                           "train_loss_all":train_loss_all,
                                           "val_loss_all":val_loss_all,
                                           "train_acc_all":train_acc_all,
                                           "val_acc_all":val_acc_all
                        })
    return train_process
    

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4)) 
    plt.subplot(1, 2, 1) 
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss") 
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss") 
    plt.legend() 
    plt.xlabel("epoch") 
    plt.ylabel("loss") 
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc") 
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.xlabel("epoch") 
    plt.ylabel("acc") 
    plt.legend() 
    plt.show() 

if __name__ == "__main__":
    LeNet = LeNet()
    train , val =  train_val_data_process()

    train_process = train_model_process(LeNet,train,val,20)
    matplot_acc_loss(train_process)
