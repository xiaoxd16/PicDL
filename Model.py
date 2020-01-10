import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.utils.data import  DataLoader
from DataLoader import *
import numpy as np
import torch.nn.functional as F
#原图像640*480.输入图像160*120
import math
import torch
from torch.optim.optimizer import Optimizer
from PublicValues import *
SIZE = WIDTH * HEIGHT

class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Acceleration of stochastic approximation by averaging:
        http://dl.acm.org/citation.cfm?id=131098
    """
    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay)
        super(ASGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # decay term
                p.data.mul_(1 - group['lambd'] * state['eta'])

                # update parameter
                p.data.add_(-state['eta'], grad)

                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)

                # update eta and mu
                state['eta'] = (group['lr'] /
                                math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
                state['mu'] = 1 / max(1, state['step'] - group['t0'])

        return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class SPPNet(nn.Module):
    sum0 = 0
    def __init__(self,num_levels = 3,pool_type = 'max_pool'):
        super(SPPNet, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self,x):
        num,c,h,w = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (
            math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new = 2 * pooling[0] + h
            w_new = 2 * pooling[1] + w

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class Discriminator(nn.Module):
    def __init__(self,num_levels = 3):
        super().__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),

            nn.Conv2d(16,midChannels,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),
            nn.BatchNorm2d(midChannels),
        )

        self.spp = SPPNet(num_levels=num_levels)
        dim=0
        for i in range(num_levels+1):
            dim += i*i * midChannels

        self.fc = nn.Sequential( nn.Linear(16*120*160,dim))

        self.mid = nn.Sequential(nn.Linear(dim,32),nn.ELU(),nn.Linear(32,16))

        if usingInterpolate:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.3),

                nn.Conv2d(16, midChannels, 3, 1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.3)
            )
            self.spp1 = SPPNet(num_levels=num_levels)
            dim = 0
            for i in range(num_levels + 1):
                dim += i * i * midChannels
            self.mid1 = nn.Sequential(nn.Linear(dim, 32), nn.LeakyReLU(), nn.Linear(32, 16))

    def forwardMain(self, x):
        x = x.cuda()
        pc = self.conv(x)
        s = self.spp(pc)
        m = self.mid(s)
        return m.cuda()

    def forwardInterpolate(self,x):
        x = x.cuda()
        pc = self.conv1(x)
        s = self.spp1(pc)
        m = self.mid1(s)
        return m.cuda()

    def forward(self,x1,x2):
        out1 = self.forwardMain(x1)
        out2 = self.forwardMain(x2)
        if usingInterpolate:
            y1 = F.interpolate(x1, scale_factor=0.5)
            y2 = F.interpolate(x2, scale_factor=0.5)
            int1 = self.forwardInterpolate(y1)
            int2 = self.forwardInterpolate(y2)
            out1 = torch.cat((out1,int1),1)
            out2 = torch.cat((out2,int2),1)
        #out = torch.cat((out1,out2),1)
        #out = self.decision(out)
        #return out.cuda()
        return out1.cuda(),out2.cuda()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100,512),
            nn.ReLU(True),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Linear(512,SIZE),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        x  = x.view(x.size(0),100)
        out = self.model(x)
        return out

if __name__ == '__main__':
    discriminator = Discriminator()
    d_optimizer = ASGD(discriminator.parameters(),lr=LR, lambd=LR, alpha=0.9, t0=1000000.0, weight_decay=0.0005)
    criterion = ContrastiveLoss()
    using_cuda = False
    if torch.cuda.is_available() :
        print("Using Cuda")
        discriminator = discriminator.cuda()

        criterion.cuda()
        using_cuda = True

    s = TrainSet()
    s.AddFiles('Data/DB2')
    s.AddFiles('Data/DB3')
    loader = DataLoader(dataset=s, batch_size=1, shuffle=True)
    acc = 0
    tol = 0
    for epoch in range(1):
        discriminator.train()
        for i, data in enumerate(loader):
            (input1,input2), labels = data
            output1,output2 = discriminator(input1, input2)
            labels = labels.cuda()
            '''
            output = output.reshape((-1,))
            loss = criterion(output ,labels)
            predict = np.array(output.tolist()).reshape((-1,))[0]
            right = np.array(labels.tolist()).reshape(((-1,)))[0]
            '''
            loss = criterion(output1,output2, labels)
            d_optimizer.zero_grad()
            loss.backward()
            d_optimizer.step()
            tol+=1
            if ((loss.data < 1).tolist() and labels.tolist()[0] == 0) or ((loss.data >= 1).tolist() and labels.tolist()[0] == 1):
                acc += 1
            if (tol % 50 == 0):
                print(acc/tol)
                acc = 0
                tol = 0

    s1 = TrainSet(dirPath='Data/DB4')
    loader = DataLoader(dataset=s1, batch_size=1, shuffle=True)
    discriminator.eval()
    print('start eval')
    acc = 0
    tol = 0
    for i, data in enumerate(loader):
        (input1, input2), labels = data
        output1, output2 = discriminator(input1, input2)
        labels = labels.cuda()
        loss = criterion(output1, output2, labels)
        ed = F.pairwise_distance(output1,output2)
        tol += 1
        print(loss.data,labels)
        if ((loss.data < 1).tolist() and labels.tolist()[0] == 0) or (
                (loss.data >= 1).tolist() and labels.tolist()[0] == 1):
            acc += 1
    print(acc/tol,acc,tol)