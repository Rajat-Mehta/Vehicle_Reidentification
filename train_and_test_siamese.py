import random
import scipy
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from model import SiameseNetwork, ContrastiveLoss
from torch import optim
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
from augmentation import RandomErasing


try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################

DATA_TRANSFORMS = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])


class Config():
    training_dir = "../Datasets/VeRi_with_plate/pytorch/train"
    testing_dir = "../Datasets/VeRi_with_plate/pytorch/gallery"
    data_dir = '../Datasets/VeRi_with_plate/pytorch'
    train_batch_size = 32
    train_number_epochs = 60


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def imshow(img, i, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("./model/siamese/test_results/test_" + str(i) +".png")


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig("./model/siamese/loss.png")
    plt.show()


def get_dataloader(data_transforms, batch_size):
    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]),
                                            should_invert=False)

    # vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
    # dataiter = iter(vis_dataloader)
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())

    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=batch_size)

    return train_dataloader


def save_model(epoch, model, loss, optimizer):
    save_filename = 'net_%s.pth' % epoch
    save_path = os.path.join('./model/siamese', save_filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)


def train_siamese_network(nclasses, fp16, transform, batch_size, num_epochs):
    since = time.time()
    net = SiameseNetwork().cuda()
    # net.classifier.classifier = nn.Sequential()

    print(net)
    print("Start time: ", since)

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    if fp16:
        # model = network_to_half(model)
        # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
        print("Memory saving is on using fp16")
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    counter = []
    loss_history = []
    iteration_number = 0
    train_dataloader = get_dataloader(transform, batch_size)
    print("Started training siamese network")

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            optimizer.zero_grad()

            output1, output2 = net(img0, img1)

            loss_contrastive = criterion(output1, output2, label)
            # loss_contrastive.backward()
            # optimizer.step()
            if fp16:  # we use optimier to backward loss
                with amp.scale_loss(loss_contrastive, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_contrastive.backward()
            optimizer.step()

            if i % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Epoch number {} finished, Current loss {}\n".format(epoch, loss_contrastive.item()))

        if epoch % 10 == 9:
            save_model(epoch, net, loss_contrastive, optimizer)
    show_plot(counter, loss_history)


def test_siamese(net):

    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]),
                                            should_invert=False)

    test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=32, shuffle=True)
    dataiter = iter(test_dataloader)
    batch = next(dataiter)
    concatenated = torch.cat((batch[0], batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated),100)

    for i in range(len(batch[0])):
        x0, x1, label = batch[0][i], batch[1][i], batch[2][i]
        x0.unsqueeze_(0)
        x1.unsqueeze_(0)

        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)

        imshow(torchvision.utils.make_grid(concatenated), i,
               'Dissimilarity: {:.2f}, Same: {}'.format(euclidean_distance.item(),
                                                        "Yes" if label.item() == 0.0 else "No"))


def extract_features(net, dataloader):
    features = torch.FloatTensor()
    count = 0
    print("Extracting siamese features:")
    for data in dataloader:
        img, _ = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 100).zero_()
        output, _ = net(Variable(img).cuda(), None)
        f = output.data.cpu().float()
        features = torch.cat((features, f), 0)

    return features


def get_siamese_features(gallery_cam, gallery_label, query_cam, query_label, nclasses):
    model = get_siamese_model(nclasses)

    image_datasets = {x: datasets.ImageFolder(os.path.join(Config.data_dir, x), DATA_TRANSFORMS) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query']}
    gallery_feature = extract_features(model, dataloaders['gallery'])
    query_feature = extract_features(model, dataloaders['query'])
    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('./saved_features/result_VeRi_siamese.mat', result)
    print("Features saved in saved_feature directory.")
    exit()


def get_siamese_model(nclasses):
    device = torch.device("cuda")
    model = SiameseNetwork()
    # model.classifier.classifier = nn.Sequential()
    checkpoint = torch.load('./model/siamese/net_59.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    print('Testing the trained Siamese network on VeRi dataset:')
    model = get_siamese_model(575)
    test_siamese(model)
    print('Testing finished. Check result of the test in model/siamese/test_results folder')


if __name__ == "__main__":
    main()
