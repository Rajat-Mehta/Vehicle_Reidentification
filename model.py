import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import torch.nn.functional as F
from sklearn.cluster import KMeans
from kmeans_pytorch.kmeans import lloyd
import numpy as np
import pickle


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock_Siamese(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=False, num_bottleneck=512):
        super(ClassBlock_Siamese, self).__init__()
        add_block = []
        # add_block += [nn.Linear(input_dim, num_bottleneck)]
        num_bottleneck = input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(f)
        return x, f


# Define the ResNet50-based Model
class ft_net_siamese(nn.Module):

    def __init__(self, class_num):
        super(ft_net_siamese, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock_Siamese(2048, class_num, dropout=False, relu=False)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x, f = self.classifier(x)
        return x, f


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=2048, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
            f = f.div(f_norm)
            # x = self.classifier(f)
            # return x, f
            return f
        else:
            x = self.classifier(x)
            return x


# Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        if input2 is not None:
            output2 = self.forward_once(input2)
        else:
            output2 = None
        return output1, output2


class SiameseNetworkResnet(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(SiameseNetwork, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward_once(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

    """
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    """

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        if input2 is not None:
            output2 = self.forward_once(input2)
        else:
            output2 = None
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# Define the ResNet50-based Model
# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg',
                 num_bottleneck=512, return_f=False, linear=True):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        self.return_f = return_f
        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            self.model = model_ft
            self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate, linear=linear,
                                         return_f=return_f, num_bottleneck=num_bottleneck)

        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
        # avg pooling to global pooling

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_name = 'nasnetalarge'
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, return_f=False, num_bottleneck=256, num_parts=6, parts_ver=1, checkerboard=False, rpp=False, share_conv=False):
        super(PCB, self).__init__()
        self.share_conv=share_conv
        self.in_features = 2048
        self.out_features = 2048
        self.return_f = return_f
        relu_bool = False
        linear_bool = True
        dropout_bool = 0.5

        if self.share_conv:
            relu_bool = True
            linear_bool = False
            self.out_features = 5000

        self.part = num_parts  # We cut the pool5 to 6 parts
        self.parts_ver = parts_ver
        self.checkerboard = checkerboard
        self.rpp=rpp

        if self.parts_ver == 1:
            pool_size = (self.part, 1)
        elif self.parts_ver == 0:
            pool_size = (1, self.part)

        if self.checkerboard:
            pool_size = (int(num_parts / 2), 2)

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        self.dropout = nn.Dropout(p=0.5)

        if self.share_conv:
            self.conv11 = nn.Conv2d(self.in_features, self.out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        #self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        
        # define classifiers
        for i in range(num_parts):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(self.out_features, class_num, droprate=0.5, relu=relu_bool, bnorm=True, linear=linear_bool,
                                           num_bottleneck=num_bottleneck, return_f=return_f))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        if self.share_conv:
            x = self.conv11(x)
        
        part = {}
        predict = {}

        if self.checkerboard and not self.rpp:
            # get checkerboard feature maps
            k = 0
            for i in range(int(self.part / 2)):
                for j in range(2):
                    part[k] = torch.squeeze(x[:, :, i, j])
                    name = 'classifier' + str(k)
                    c = getattr(self, name)
                    predict[k] = c(part[k])
                    k += 1
        else:
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                if self.parts_ver == 1:
                    part[i] = torch.squeeze(x[:, :, i])
                elif self.parts_ver == 0:
                    part[i] = torch.squeeze(x[:, :, :, i])

                name = 'classifier' + str(i)
                c = getattr(self, name)
                predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]

        if not self.return_f:
            y = []
            for i in range(0, self.part):        
                y.append(predict[i])
            return y
        else:
            conc = predict[0]
            for i in range(1, self.part):
                conc = torch.cat((conc, predict[i]), dim=1)
            return conc, conc

    def convert_to_rpp(self):
        self.avgpool = RPP(self.part)
        return self

    def convert_to_rpp_cluster(self):
        self.avgpool = Cluster(self.part)
        return self


# Define the RPP layers
class RPP(nn.Module):
    def __init__(self, num_parts):
        super(RPP, self).__init__()
        self.part = num_parts
        add_block = []
        # add_block += [nn.Linear(2048, self.part)]

        add_block += [nn.Conv2d(2048, self.part, kernel_size=1, bias=False)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        norm_block = []
        norm_block += [nn.BatchNorm2d(2048)]
        #norm_block += [nn.ReLU(inplace=True)]
        norm_block += [nn.LeakyReLU(0.1, inplace=True)]
        norm_block += [nn.Dropout(p=0.5)]
                

        norm_block = nn.Sequential(*norm_block)
        norm_block.apply(weights_init_kaiming)

        self.add_block = add_block
        self.norm_block = norm_block
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        w = self.add_block(x)
        p = self.softmax(w)

        y = []
        for i in range(self.part):
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.norm_block(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)

        f = torch.cat(y, 2)

        return f


class Cluster(nn.Module):
    def __init__(self, num_parts):
        super(Cluster, self).__init__()
        self.part = num_parts
        self.kmeans = KMeans(n_clusters=self.part)
        self.centers_path=None
        self.clusters=None 

    def pairwise(self, x, y=None):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        dist[dist != dist] = 0
        return dist
    
    def remove_nan(self, x):
        x[x != x] = 0
        return x

    def isnan(self, x):
        x = x != x
        t = np.unique(x.cpu().float().numpy())
        #print(t)
        return 1 in t

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        with open(self.centers_path, "rb") as c:
            self.clusters = pickle.load(c).cuda()

        output = torch.FloatTensor().cuda()
        for i in range(x.shape[0]):
            clus = torch.FloatTensor().cuda()
            dist = self.pairwise(x[i],self.clusters)
            choice_cluster = torch.argmin(dist, dim=1)
            initial_state = []
            for index in range(self.part):
                selected = torch.nonzero(choice_cluster==index).squeeze()   
                selected = torch.index_select(x[i], 0, selected)                
                m = selected.mean(dim=0)
                if self.isnan(m):
                    m = self.remove_nan(m)
                m=m.unsqueeze_(0)
                clus=torch.cat((clus, m), dim=0)
            clus.unsqueeze_(0)
            output = torch.cat((output,clus),dim=0)
        output=output.permute(0,2,1)
        # print(x.shape)
        #clusters_index, centers = KMeans(x, self.part)#load precomputed 6 clusters from pickle file
        #for each of the 288 vectors in each image, find the nearest cluster.
        #vectors belonging to same clusters will form clusters
        #take their average, which will result in 32*2048*6
        #return it

        #for i in range(x.shape[0]):
            # lloyd(X, n_clusters, device=0, tol=1e-4)
            #clusters_index, centers = lloyd(x[i].cpu().detach().numpy(), self.part, device=0, tol=1e-4)
            #cent = centers

            #self.kmeans.fit(x[i].cpu().detach().numpy())
            #cent = self.kmeans.cluster_centers_
            #cent = torch.from_numpy(cent).float()
            #cent = cent.permute(1, 0)
            #cent.unsqueeze_(0)
            #clus = torch.cat((clus, cent), 0)
        #clus.unsqueeze_(3)
        return output


class PCB_test(nn.Module):
    def __init__(self, model, num_parts):
        super(PCB_test, self).__init__()
        self.part = num_parts
        self.model = model.model
        self.avgpool = model.avgpool
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), self.part)

        return y


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12288, 6144),
            nn.ReLU(True), 
            nn.Linear(6144, 2048), 
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(2048, 6144),
            nn.ReLU(True),
            nn.Linear(6144, 12288),
            nn.Sigmoid())

    def min_max_normalization(self, tensor, min_value, max_value):
        min_tensor = tensor.min()
        tensor = (tensor - min_tensor)
        max_tensor = tensor.max()
        tensor = tensor / max_tensor
        tensor = tensor * (max_value - min_value) + min_value
        return tensor
        
    def forward(self, x):
        x = self.min_max_normalization(x, 0, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)
