import os.path as osp
import os
import random

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv, GCNConv, BatchNorm, PNAConv
from torch.nn import Linear
from torch_geometric.utils import degree

from torch_geometric.data import InMemoryDataset#, download_url
from torch_geometric.data import Dataset#, download_url
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#to read http://pages.di.unipi.it/citraro/files/slides/Landolfi_tutorial.pdf
# variables needed to know as loaded:


class MyOwnDataset(InMemoryDataset):
    root = "/home/mgarcia/si_RNA_Network/test/"
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data  = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_input = "/home/mgarcia/si_RNA_Network/raw/"
        arr = os.listdir(raw_input)
        return ['RF00034.txt']

    @property
    def processed_file_names(self):
        processed_folder = "/home/mgarcia/si_RNA_Network/processed/"
        processed_files = os.listdir(processed_folder)
        if len(processed_files) > 10:
            file_names = processed_files
        else:
            file_names = ["data_0.pt"]
        return file_names

    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        pass

    def process(self):
        # Read data into huge `Data` list.
        #print("path_raw", self.raw_paths)
        #idx = 0
        for raw_path in self.raw_paths:
            # print(raw_path)
            # Read data from `raw_path`.
            raw_list_atrr = []
            raw_list_edge = []
            raw_list_feature = []
            raw_list_label = []
            raw_train_mask = []
            raw_test_mask = []
            raw_val_mask = []

            with open(raw_path, 'r') as f:
                for i in f:
                    if "feature" in i:
                        # i.strip().split('feature')[1]
                        feature = i.strip().split('feature')
                        feature = feature[1:]
                        # print("feature:", feature)
                        for k in feature:
                            d = k.replace("[", "").replace("]", "").split(",")
                            for k in d:
                                k = float(k)
                                # print("k", k)
                                raw_list_feature.append(k)
                    elif "edge_index" in i:
                        edge_index_raw = i.strip().split('edge_index')
                        edge_index_raw = edge_index_raw[1:]
                        # print("edge_index_raw", edge_index_raw)
                        # print(type(edge_index_raw))

                        for k in edge_index_raw:
                            # print(type(k))
                            q = k.split("], ")
                            # print("q", q)
                            # print(type(q))
                            for j in q:
                                u = j.replace("[", "").replace("]", "").split(",")
                                u = [int(u[0]), int(u[1])]
                                # print(u)
                                raw_list_edge.append(u)
                    elif "edge_attribute" in i:
                        edge_attribute = i.strip().split('edge_attribute')
                        edge_attribute = edge_attribute[1:]
                        # print("edge_attribute", edge_attribute)
                        # print(type(edge_attribute))

                        for k in edge_attribute:
                            # print(type(k))
                            q = k.split("], ")
                            # print("q", q)
                            # print(type(q))
                            for j in q:
                                u = j.replace("[", "").replace("]", "").split(",")
                                u = [float(u[0]), float(u[1])]
                                # print("atrr", u)
                                raw_list_atrr.append(u)
                    elif "label" in i:
                        label = i.strip().split('label')
                        label = label[1:]
                        # print("feature:", feature)
                        for k in label:
                            d = k.replace("[", "").replace("]", "").split(",")
                            for k in d:
                                k = int(k)
                                # print("k", k)
                                raw_list_label.append(k)  #
                attributes_tensor = torch.tensor(raw_list_atrr)
                edge_index_tensor = torch.tensor(raw_list_edge, dtype=torch.long).t().contiguous()
                node_feature_tensor = torch.tensor(raw_list_feature, dtype=torch.float)
                label_tensor = torch.tensor(raw_list_label)

        data = Data(x=node_feature_tensor, edge_index=edge_index_tensor, edge_attr=attributes_tensor, y=label_tensor)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        #data, slices = self.collate(data_list)
        torch.save((data), self.processed_paths[0])

dataset = MyOwnDataset("/home/mgarcia/si_RNA_Network/")  #, transform=transform
data = dataset[0]
#print(data)
#print(data.keys)

max_degree = -1
d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
max_degree = max(max_degree, int(d.max()))
deg = torch.zeros(max_degree + 1, dtype=torch.long)
deg += torch.bincount(d, minlength=deg.numel())

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_channels = 32

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.conv1 = PNAConv(in_channels=data.num_node_features, out_channels=hidden_channels,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=2, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm1 = BatchNorm(hidden_channels)

        self.conv2 = PNAConv(in_channels=hidden_channels, out_channels=dataset.num_classes,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=2, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #print(f"x: {x}")
        #print(x.size())
        #print(f"edge_index: {edge_index}")
        #print(edge_index.size())
        #print(f"edge_attr: {edge_attr}")
        #print(edge_attr.size())
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

#train_, test_ and val_mask:
count_target = 0
cluster_count = 0
print("count_target: ", count_target)
print("cluster_count: ", cluster_count)
print("num_labels", len(data.y))
for reader in data.y:
    if reader == 0:
        cluster_count = cluster_count + 1
    elif reader == 1:
        count_target = count_target + 1
    elif reader == 2:
        count_target = count_target + 1
print("80% target: ", round(count_target*0.8),"20% target: ", round(count_target*0.2))
train_targets = round(count_target*0.8)
train_mask_num = [1 for i in range(len(data.y))]
val_mask_num = [0 for i in range(len(data.y))]
test_mask_num = [0 for i in range(len(data.y))]
check = 0
#neuer ansatz mit nur einem loop
path_idx = "/home/mgarcia/si_RNA_Network/raw/RF00034.idx"
with open(path_idx, 'r') as f:
    for i in f:
        if 'pos_idx' in i:
            pos_idx = i.strip().split('pos_idx')
            pos_idx = pos_idx[1]
            pos_idx = pos_idx.replace('[','').replace(']','').split(',')
        elif 'targets' in i:
            targets = i.strip().split('targets')
            targets = targets[1]
            targets = targets.replace('[','').replace('[','').split(',')
"""
for index, i in enumerate(data.y):
    if i == 0:
        train_mask_num[index] = 1
        val_mask_num[index] = 0
        test_mask_num[index] = 0
    elif i == 1:
        if check == train_targets:
            test_mask_num[index] = 1
        else:
            train_mask_num[index] = 1
            check = check + 1
    elif i == 2:
        if check == train_targets:
            test_mask_num[index] = 1
        else:
            train_mask_num[index] = 1
            check = check + 1
print(len(data.y))
print(len(train_mask_num))
print(len(test_mask_num))
print(len(val_mask_num))
data.train_mask  = torch.tensor(train_mask_num, dtype=torch.bool)
data.test_mask = torch.tensor(test_mask_num, dtype=torch.bool)
data.val_mask = torch.tensor(val_mask_num, dtype=torch.bool)
print(data.train_mask)
print(data.test_mask)
print(data.val_mask)
"""
target_train = random.sample(targets, k=round(count_target*0.8))
test_and_val = []
for target in targets:
    if target not in target_train:
        test_and_val.append(target)
test_samples = test_and_val[:round(len(test_and_val)/2)]#<-- random as well?
val_samples = test_and_val[round(len(test_and_val)/2):]

for index, node in enumerate(pos_idx):
    for exclusion in test_and_val:
        if exclusion == node:
            train_mask_num[index] = 0
    for test_node in test_samples:
        if test_node == node:
            test_mask_num[index] = 1
    for val_node in val_samples:
        if val_node == node:
            val_mask_num[index] = 1
train_mask = torch.tensor(train_mask_num,dtype=torch.bool)
test_mask = torch.tensor(test_mask_num,dtype=torch.bool)
val_mask = torch.tensor(val_mask_num,dtype=torch.bool)
data.test_mask = test_mask
data.train_mask = train_mask
data.val_mask = val_mask


device = torch.device('cuda')

model = Net()
model.to(device)
data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    #error for train here tensor size is not equal see :
    #https://stackoverflow.com/questions/56783182/runtimeerror-the-size-of-tensor-a-133-must-match-the-size-of-tensor-b-10-at
    #print("shape",data.train_mask.size)
    #F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    out = model()
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    #loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
val_acc = 0
for epoch in range(1, 201):
    loss = train()
    val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Val: {:.4f}, Test:{:.4f} loss: {:.4f}'
    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc, loss))

