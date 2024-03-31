import torch
from data.dataloader import data_loader_train
from models.network import Networks
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import functional as F


def data_load(data_name):
    data_dir = "/Volumes/My Passport/数据/data/BIC/"
    data_path = data_dir + data_name
    f = open(data_path)  # 读取文件
    lines = f.readlines()
    # print(len(lines))

    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("\t")[1:]
        # print(lines[i])

    gene_expression = np.zeros([len(lines) - 1, len(lines[1])])

    for i in range(len(lines)):
        if i == 0:
            continue
        for j in range(len(lines[1])):
            gene_expression[i - 1][j] = float(lines[i][j])

    gene_expression = np.matrix(gene_expression)
    # print(gene_expression.shape)
    gene_expression = np.transpose(gene_expression)
    gene_expression = np.array(gene_expression)

    return gene_expression


def load_data(data_name):
    gene_expression = data_load(data_name)
    Label = gene_expression
    # Img = np.reshape(Img, (Img.shape[0], 32, 32, 1))
    Img = np.reshape(gene_expression, [gene_expression.shape[0], 1, gene_expression.shape[1], 1])
    n_input = [1, gene_expression.shape[1]]

    return gene_expression, Img, Label, n_input


def get_kNNgraph2(data, K_num):
    # each row of data is a sample

    x_norm = np.reshape(np.sum(np.square(data), 1), [-1, 1])  # column vector
    x_norm2 = np.reshape(np.sum(np.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * np.matmul(data, np.transpose(data)) + x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample, num_sample), dtype=np.int_)
    for i in range(num_sample):
        distance = dists[i, :]
        small_index = np.argsort(distance)
        graph[i, small_index[0:K_num]] = 1
    graph = graph - np.diag(np.diag(graph))
    resultgraph = np.maximum(graph, np.transpose(graph))
    return resultgraph


def comp(g):
    g = g + np.identity(g.shape[0])
    g = torch.tensor(g)
    d = np.diag(g.sum(axis=1))
    d = torch.tensor(d)
    s = pow(d, -0.5)
    where_are_inf = torch.isinf(s)
    s[where_are_inf] = 0
    s = torch.matmul(torch.matmul(s, g), s).to(torch.float32)
    return s


data_dir = "/Volumes/My Passport/数据/data/BIC/"
data_name1 = "BREAST_Gene_Expression.txt"
#data_name1 = "partial/BIC_Gene_0.5.txt"
data_name2 = "BREAST_Methy_Expression.txt"
data_name3 = "BREAST_Mirna_Expression.txt"
gene_expression1, Img1, Label1, n_input1 = load_data(data_name1)
gene_expression2, Img2, Label2, n_input2 = load_data(data_name2)
gene_expression3, Img3, Label3, n_input3 = load_data(data_name3)
batch_size = gene_expression1.shape[0]

Img1 = np.squeeze(Img1, axis=None)
Img2 = np.squeeze(Img2, axis=None)
Img3 = np.squeeze(Img3, axis=None)
from bunch import *

s = Bunch()
s.data = [[Img1], [Img2], [Img3]]

ind1 = np.any(Img1, axis=1).astype(int)
ind2 = np.any(Img2, axis=1).astype(int)
ind3 = np.any(Img3, axis=1).astype(int)
we = np.array(list(zip(ind1, ind2, ind3)), dtype='float32')
we = torch.tensor(we)

g1 = get_kNNgraph2(Img1, 10)
s1 = comp(g1)
g2 = get_kNNgraph2(Img2, 10)
s2 = comp(g2)
g3 = get_kNNgraph2(Img3, 10)
s3 = comp(g3)

regg2 = [1]
regg3 = [1]
model = Networks(Img1, Img2, Img3).to(device)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
n_epochs = 100
l = 0

n_epochs2 = 100

y = 0  # 记录reg组合
for r in range(0, len(regg2)):
    reg2 = regg2[r]
    t = 0
    for h in range(0, len(regg3)):

        print("---------------------------------------")
        reg3 = regg3[h]
        l = 0
        model = Networks(Img1, Img2, Img3).to(device)
        optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
        for epoch in range(n_epochs2):
            for data in data_loader_train:
                train_imga, train_imgb, train_imgc = data

                input1 = train_imga.clone().detach()
                input2 = train_imgb.clone().detach()
                input3 = train_imgc.clone().detach()
            
                x_1, x_2, x_3, a_1, a_2, a_3, z, p, q, cluster_layer = model.forward(input1, input2, input3, we, s1, s2,
                                                                                     s3)

                loss_x = criterion(x_1, input1) + criterion(x_2, input2) + criterion(x_3, input3)
                loss_a = criterion(a_1, torch.Tensor(g1)) + criterion(a_2, torch.Tensor(g2)) + criterion(a_3,
                                                                                                         torch.Tensor(
                                                                                                             g3))
                loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
                loss = loss_x + reg2 * loss_a + reg3 * loss_kl

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
            if epoch % 1 == 0:
                print("Epoch {}/{}".format(epoch, n_epochs2))
                print("Loss is:{:.4f}".format(loss.item()))
                path=("/Volumes/My Passport/gcn/test/BIC/0/_param_comb/param2_" + str(regg2[y]) + "/param3_" + str(
                    regg3[t]) + "/epoch_" + str(l)+"/")
                if os.path.exists(path):
                    pass
                else:
                    os.mkdir(path)

                np.savetxt("/Volumes/My Passport/gcn/test/BIC/0/_param_comb/param2_" + str(regg2[y]) + "/param3_" + str(
                    regg3[t]) + "/epoch_" + str(l) + "/matrix" + ".txt", z.cpu().detach().numpy(),
                           fmt='%lf', delimiter='\t')
                torch.save(model.state_dict(),
                           '/Volumes/My Passport/gcn/test/BIC/0/models/param2_' + str(regg2[y]) + '_param3_' + str(
                               regg3[t]) + '_epoch_' + str(l) + '.pth')
                l = l + 1
        t = t + 1
    y = y + 1


