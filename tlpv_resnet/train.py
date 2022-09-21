import os
import json
import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm

from Googlenet import resnet18, resnet34, resnet50


# from rbnresnet import resnet18,resnet34,resnet50
# from matplotlib import pyplot as plt
# from googlenet import GoogLeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # image_path="D:\\git\\resnet-study\\datasets\\spv_data"
    image_path = "D:\\git\\datasets\\spv_data\\spvdata5_5"

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # batch_size = 16
    batch_size = 32
    # 多线程
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net =resnet34()
    net = resnet50(num_classes=6)

    # net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    # model = models.googlenet()
    # model.aux_logits = False
    # net=model
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "checkpoint_pre//resnet18-5c106cde.pth"
    # model_weight_path = "checkpoint_pre//resnet34-333f7ec4.pth"
    # # model_weight_path = "checkpoint_pre//resnet50-19c8e357.pth"
    #
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # # for param in net.parameters():
    # #     param.requires_grad = False
    #
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 6)
    # print(in_channel)
    #
    # print(net)
    #
    # net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)

    name_file = '30spvdata2-5-5resnet50'
    epochs = 30

    best_acc = 0.0
    # save_path = './rbnresnet18.pth'
    save_path = './' + name_file + '.pth'
    print(save_path)
    train_steps = len(train_loader)
    loss_list = []
    accuracy_list = []
    epoch_list = []
    for epoch in range(epochs):
        t1 = time.perf_counter()
        epoch_list.append(epoch + 1)
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # logits, aux_logits2, aux_logits1 = net(images.to(device))
            logits = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            # loss1 = loss_function(aux_logits1, labels.to(device))
            # loss2 = loss_function(aux_logits2, labels.to(device))
            # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss = loss0
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # scheduler.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        t2 = time.perf_counter()
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f train_time:%.2f s' %
              (epoch + 1, running_loss / train_steps, val_accurate, (t2 - t1)))
        file_w = open(name_file + '.txt', mode='a')
        # file_w.write(' \n'+'epoch:'+str(epoch+1)+'  val_acc:'+str(running_loss/train_steps)+'  train_time:'+str(t2-t1) + ' \n')
        file_w.write('\n' + '[epoch %d]  train_loss: %.3f  val_accuracy: %.3f  train_time: %.2f s' %
                     (epoch + 1, running_loss / train_steps, val_accurate, (t2 - t1)))
        file_w.close()

        loss_list.append(running_loss / train_steps)
        accuracy_list.append(val_accurate * 100)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    plt.rcParams["font.sans-serif"] = "SimHei"  # 修改字体的样式可以解决标题中文显示乱码的问题

    plt.title(name_file + ':准确率曲线')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    plt.plot(epoch_list, accuracy_list)
    plt.grid()
    plt.show()

    plt.title(name_file + ':损失函数曲线')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_list, loss_list)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
