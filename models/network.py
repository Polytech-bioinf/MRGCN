import torch.nn as nn
import torch
import numpy as np
from torch.nn import Linear
from sklearn.cluster import KMeans
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_kNNgraph2(data, K_num):
    # each row of data is a sample

    x_norm = np.reshape(np.sum(np.square(data), 1), [-1, 1])  # column vector
    x_norm2 = np.reshape(np.sum(np.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * np.matmul(data, np.transpose(data)) + x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample, num_sample), dtype=np.int)
    for i in range(num_sample):
        distance = dists[i, :]
        small_index = np.argsort(distance)
        graph[i, small_index[0:K_num]] = 1
    graph = graph - np.diag(np.diag(graph))
    resultgraph = np.maximum(graph, np.transpose(graph))
    return resultgraph
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
class Networks(nn.Module):
    def __init__(self, input1, input2,input3):
        super(Networks, self).__init__()
        dims1 = []
        for idim in range(1):
            linshidim = round(input1.shape[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = input3.shape[1] * 0.8
        linshidim = int(linshidim)
        dims1.append(linshidim)

        dims2 = []
        for idim in range(1):
            linshidim = round(input2.shape[1] * 0.8)
            linshidim = int(linshidim)
            dims2.append(linshidim)
        linshidim = input3.shape[1] * 0.8
        linshidim = int(linshidim)
        dims2.append(linshidim)

        dims3 = []
        for idim in range(1):
            linshidim = round(input3.shape[1] * 0.8)
            linshidim = int(linshidim)
            dims3.append(linshidim)
        linshidim = input3.shape[1] * 0.8
        linshidim = int(linshidim)
        dims3.append(linshidim)
        # encoder0
        self.enc1_1 = Linear(input1.shape[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        #self.enc1_3 = Linear(dims1[1], dims1[2])

        # encoder1
        self.enc2_1 = Linear(input2.shape[1], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        #self.enc2_3 = Linear(dims2[1], dims2[2])

        # encoder2
        self.enc3_1 = Linear(input3.shape[1], dims3[0])
        self.enc3_2 = Linear(dims3[0], dims3[1])
        #self.enc3_3 = Linear(dims3[1], dims3[2])

        # decoder0
        self.dec1_1 = Linear(dims1[1], dims1[0])
        self.dec1_2 = Linear(dims1[0], input1.shape[1])
        # self.enc1_3 = Linear(dims1[1], dims1[2])

        # decoder1
        self.dec2_1 = Linear(dims2[1], dims2[0])
        self.dec2_2 = Linear(dims2[0], input2.shape[1])
        # self.enc2_3 = Linear(dims2[1], dims2[2])

        # decoder2
        self.dec3_1 = Linear(dims3[1], dims3[0])
        self.dec3_2 = Linear(dims3[0], input3.shape[1])
        # self.enc3_3 = Linear(dims3[1], dims3[2])

        self.weight1 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims1[1], dims1[1])))
        self.weight2 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims2[1], dims2[1])))
        self.weight3 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims3[1], dims3[1])))
        self.weight_1 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims1[1], dims1[1])))
        self.weight_2 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims2[1], dims2[1])))
        self.weight_3 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims3[1], dims3[1])))

        #self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, dims1[2]))
    def forward(self, input1, input2,input3,we,s1,s2,s3):

        output1_1 = torch.tanh(self.enc1_1(torch.matmul(s1,input1)))
        output1_2 = torch.tanh(self.enc1_2(torch.matmul(s1,output1_1)))

        output2_1 = torch.tanh(self.enc2_1(torch.matmul(s2,input2)))
        output2_2 = torch.tanh(self.enc2_2(torch.matmul(s2,output2_1)))

        output3_1 = torch.tanh(self.enc3_1(torch.matmul(s3,input3)))
        output3_2 = torch.tanh(self.enc3_2(torch.matmul(s3,output3_1)))

        summ = torch.diag(we[:, 0]).mm(output1_2) + torch.diag(we[:, 1]).mm(output2_2) + torch.diag(we[:, 2]).mm(output3_2)
        wei = 1 / torch.sum(we, 1)
        z = torch.diag(wei).mm(summ)
        a_1 = torch.sigmoid(torch.matmul(torch.matmul(z, self.weight1), z.T))
        a_2 = torch.sigmoid(torch.matmul(torch.matmul(z, self.weight2), z.T))
        a_3 = torch.sigmoid(torch.matmul(torch.matmul(z, self.weight3), z.T))
        h1 = torch.tanh(torch.matmul(z, self.weight_1))
        h2 = torch.tanh(torch.matmul(z, self.weight_2))
        h3 = torch.tanh(torch.matmul(z, self.weight_3))
        h1_1 = torch.tanh(self.dec1_1(torch.matmul(s1, h1)))
        h1_2 = torch.tanh(self.dec1_2(torch.matmul(s1, h1_1)))
        h2_1 = torch.tanh(self.dec2_1(torch.matmul(s2, h2)))
        h2_2 = torch.tanh(self.dec2_2(torch.matmul(s2, h2_1)))
        h3_1 = torch.tanh(self.dec3_1(torch.matmul(s3, h3)))
        h3_2 = torch.tanh(self.dec3_2(torch.matmul(s3, h3_1)))
        x_1= h1_2
        x_2= h2_2
        x_3= h3_2
        kmeans = KMeans(n_clusters=5, n_init=20)
        y_pred_ = kmeans.fit_predict(z.detach().numpy())
        cluster_layer = torch.tensor(kmeans.cluster_centers_).to(device)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2))
        q = q.pow(1)
        q = (q.t() / torch.sum(q, 1)).t()
        p = target_distribution(q)
        return x_1, x_2, x_3, a_1, a_2, a_3,z,p,q,cluster_layer







