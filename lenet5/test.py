import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_data_process():
    test = FashionMNIST(root="./data",
                          train=False,
                          transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                          download=True)

    test_load = Data.DataLoader(dataset=test,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_load
def test_model_process(model,test_load):
    device = "mps"
    model = model.to(device)

    test_correct = 0.0
    test_num = 0

    with torch.no_grad():
        for data_x , data_y in test_load:
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            output = model(data_x)
            pre_lab = torch.argmax(output,dim=1)

            test_correct += torch.sum(pre_lab ==data_y.data)

            test_num += data_x.size(0)

    test_acc = test_correct.float().item() / test_num

    print("准确率",test_acc)

if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load("./best.pth"))

    test = test_data_process()

    test_model_process(model,test)