import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

#   自定义数据集所需要的头文件
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import csv
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import random

#   引入resnet18
from resnet import ResNet18

#   定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   超参数设置
EPOCH = 200         #训练的幕数
BATCH_SIZE = 128    #批处理尺寸
Learning_Rate = 0.1            #学习率

#   图像标签路径
train_path = 'G:/Users/mask/data/mask_data/data/train'
test_path = 'G:/Users/mask/data/mask_data/data/test'
label_path = 'G:/Users/mask/data/mask_data/data/sample_submit.csv'
mask_label = ['mask_weared_incorrect', 'with_mask', 'without_mask']
mini_batch_size = 128
img_size = 32
n_cpu = 1

class ListDataset(Dataset):
    def __init__(self, img_path, label_path, img_size, transform):
        #   读取图像路径
        self.mask_wear_incorrect = os.listdir(img_path+'/mask_weared_incorrect')
        self.with_mask = os.listdir(img_path + '/with_mask')
        self.without_mask = os.listdir(img_path + '/without_mask')
        self.mask_wear_incorrects = [img_path+'/mask_weared_incorrect/'+mask_wear_incorrect_name for mask_wear_incorrect_name in self.mask_wear_incorrect
                                     if mask_wear_incorrect_name!='.DS_Store']
        self.with_masks = [img_path+'/with_mask/'+with_mask for with_mask in self.with_mask
                                    if with_mask!='.DS_Store']
        self.without_masks = [img_path+'/without_mask/'+without_mask for without_mask in self.without_mask
                                    if without_mask!='.DS_Store']
        self.img_paths = self.mask_wear_incorrects + self.with_masks + self.without_masks

        #   制作labels
        self.labels = [[1,0,0] for i in self.mask_wear_incorrects] + [[0,1,0] for i in self.with_masks] + [[0,0,1] for i in self.without_masks]

        #   其他参数
        self.img_nums = len(self.img_paths)
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        #   获取图像
        try:
            img_path = self.img_paths[index % self.img_nums]
            # img =np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            # img = transforms.ToTensor()(img)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.uint8)
            img = transforms.ToTensor()(img)

        except Exception:
            print(f"不能找到图像'{img_path}'.")
            return

        #   获取标签
        try:
            label = torch.tensor(self.labels[index % self.img_nums]).unsqueeze(0).float()
        except Exception:
            print(f"不能找到标签'{img_path}'.")
            return
        return img, label

    def collate_fn(self, batch):
        batch = [data for data in batch if data is not None]
        #   实现类似转置的操作
        imgs, labels = list(zip(*batch))
        #   还需要把imgs转化成所需要的input shape，tensor类型 （batchsize，channals，weight，height）
        imgs = torch.stack([F.interpolate(img.unsqueeze(0), size=32).squeeze(0) for img in imgs])
        labels = torch.cat(labels, dim=0)
        return imgs, labels

    def __len__(self):
        return self.img_nums

class TestListDataset(Dataset):
    def __init__(self, img_path, label_path, img_size, transform):

        self.img_paths = os.listdir(img_path)
        self.img_paths = [img_path +'/'+ path for path in self.img_paths
                                    if path != '.DS_Store']
        self.names = []
        self.types = []
        with open(label_path, encoding='utf-8') as f:
            readers = csv.reader(f)
            headers = next(readers)
            for reader in readers:
                self.names.append(reader[0])
                self.types.append(reader[1])
        self.labels = []
        for path in self.img_paths:
            name = path.rsplit('/')[-1]
            num = self.names.index(name)
            if self.types[num] == mask_label[0]:
                self.labels.append([1, 0, 0])
            elif self.types[num] == mask_label[1]:
                self.labels.append([0, 1, 0])
            else:
                self.labels.append([0, 0, 1])

                #   其他参数
        self.img_nums = len(self.img_paths)
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        #   获取图像
        try:
            img_path = self.img_paths[index % self.img_nums]
            # img =np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            # img = img.resize((32,32,3))
            img = Image.open(img_path).convert('RGB')
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.uint8)
            img = transforms.ToTensor()(img)

        except Exception:
            print(f"不能找到图像'{img_path}'.")
            return

        #   获取标签
        try:
            label = torch.tensor(self.labels[index % self.img_nums]).unsqueeze(0).float()
        except Exception:
            print(f"不能找到标签'{img_path}'.")
            return
        return img, label

    def collate_fn(self, batch):
        batch = [data for data in batch if data is not None]
        #   实现类似转置的操作
        imgs, labels = list(zip(*batch))
        #   还需要把imgs转化成所需要的input shape，tensor类型 （batchsize，channals，weight，height）
        imgs = torch.stack([F.interpolate(img.unsqueeze(0), size=32).squeeze(0) for img in imgs])
        labels = torch.cat(labels, dim=0)
        return imgs, labels

    def __len__(self):
        return self.img_nums



def worker_seed_set(worker_id):
    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


if __name__ == '__main__':
    dataset = ListDataset(train_path, label_path, img_size, None)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=dataset.collate_fn,
                            pin_memory=True,
                            worker_init_fn=worker_seed_set)

    testdataset = TestListDataset(test_path, label_path, img_size, None)
    testdataloader = DataLoader(testdataset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=testdataset.collate_fn,
                                pin_memory=True)

    # for imgs, labels in testdataloader:
    #     pass

    model = ResNet18(3).to(device)

    #   多分类问题用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimzer = optim.SGD(model.parameters(), lr=Learning_Rate, momentum=0.9, weight_decay=5e-4)

    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            for epoch in range(0, EPOCH):
                print('\n第{}幕'.format(epoch+1))
                model.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for batch_i, (imgs, labels) in enumerate(dataloader, 0):
                    length = len(dataloader)
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimzer.zero_grad()
                    outputs = model(imgs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimzer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    _, label = torch.max(labels, 1)
                    correct += predicted.eq(label).cpu().sum()
                    print('[幕:%d, 迭代次数:%d] 平均损失:%.03f | 准确率:%.03f'
                          %(epoch+1, (batch_i+1 + epoch*length), sum_loss/(batch_i+1), 100. *correct/total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (batch_i + 1 + epoch * length), sum_loss / (batch_i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                print("等待测试！")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for imgs2, labels2 in testdataloader:
                        model.eval()
                        imgs2, labels2 = imgs2.to(device), labels2.to(device)
                        outputs = model(imgs2)

                        _, predicted = torch.max(outputs, 1)
                        _, label = torch.max(labels2, 1)

                        total += labels2.size(0)
                        correct += predicted.eq(label).cpu().sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    if acc > 85:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

                    torch.save(model.state_dict(), 'mask.pth')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


