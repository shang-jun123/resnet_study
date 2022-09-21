import os
import json
import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms, datasets,models
import torch.optim as optim
from tqdm import tqdm

#from vgg_model import vgg
#from model import GoogLeNet
from resnet import resnet18,resnet34,resnet50
#from rbn_resnet import resnet18
#from i_conv1_resnet18 import ResNet18

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #image_path="D:\\git\\resnet-study\\datasets\\spv_data"
    image_path = "D:\\git\\datasets\\pvdata11"
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=5)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
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

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
    #             "aux2.fc2.weight", "aux2.fc2.bias",
    #             "fc.weight", "fc.bias"]
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)
    #net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net = resnet18(num_classes=11)
    #net = resnet34(num_classes=6)
    #net = resnet50(num_classes=6)
    #model_name = "vgg19"
    #net = vgg(model_name=model_name, num_classes=6, init_weights=True)
    #net=ResNet18()
    print(net)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    #optimizer = optim.Adam(net.parameters(), lr=0.05)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    name_file='200-pvdata11-bzresnet18'
    epochs = 200
    best_acc = 0.0
    #save_path = './rbnresnet18.pth'
    save_path = './'+name_file+'.pth'
    print (save_path)
    train_steps = len(train_loader)
    loss_list = []
    accuracy_list = []
    epoch_list = []
    for epoch in range(epochs):
        t1=time.perf_counter()
        epoch_list.append(epoch+1)
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            #logits, aux_logits2, aux_logits1 = net(images.to(device))
            logits= net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            #loss1 = loss_function(aux_logits1, labels.to(device))
            #loss2 = loss_function(aux_logits2, labels.to(device))
            #loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss=loss0
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        #scheduler.step()
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
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
        t2=time.perf_counter()
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f train_time:%.2f s' %
              (epoch + 1, running_loss / train_steps, val_accurate,(t2-t1)))
        file_w = open(name_file+'.txt', mode='a')
        #file_w.write(' \n'+'epoch:'+str(epoch+1)+'  val_acc:'+str(running_loss/train_steps)+'  train_time:'+str(t2-t1) + ' \n')
        file_w.write('\n'+'[epoch %d]  train_loss: %.3f  val_accuracy: %.3f  train_time: %.2f s' %
              (epoch + 1, running_loss / train_steps, val_accurate, (t2 - t1)))
        file_w.close()

        loss_list.append( running_loss / train_steps)
        accuracy_list.append(val_accurate*100)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    plt.rcParams["font.sans-serif"] = "SimHei"  # 修改字体的样式可以解决标题中文显示乱码的问题

    plt.title(name_file+':准确率曲线')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    plt.plot(epoch_list, accuracy_list)
    plt.grid()
    plt.show()

    plt.title(name_file+':损失函数曲线')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_list, loss_list)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
