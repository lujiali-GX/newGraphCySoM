from rdkit import Chem
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric import loader
import numpy as np
import os
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.nn import MFConv
from torch.utils.data import random_split
import pickle
from torch_geometric.explain import CaptumExplainer, Explainer,PGExplainer, GNNExplainer
from torch_geometric.utils import from_networkx
from torch_geometric.explain import unfaithfulness
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, jaccard_score, balanced_accuracy_score

train_datapath = './datasets/train_datsets.pkl'
fr = open(train_datapath, 'rb')
train_data = pickle.load(fr)
test_datapath = './datasets/test_datsets.pkl'
fe = open(test_datapath, 'rb')
test_data = pickle.load(fe)

training_set, validation_set = random_split(train_data, [int(len(train_data) * 0.85), len(train_data) - int(len(train_data) * 0.85)], generator=torch.Generator().manual_seed(12345))
batch_size = 1
train_loader = loader.DataLoader(training_set, batch_size, shuffle=True)
val_loader = loader.DataLoader(validation_set, batch_size, shuffle=True)
test_loader = loader.DataLoader(test_data, batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        num_classses = 2

        conv_hidden = args['conv_hidden']
        cls_hidden = args['cls_hidden']
        self.n_layers = args['n_layers']

        self.conv1 = MFConv(29, conv_hidden, 5)
        self.conv2 = MFConv(conv_hidden, conv_hidden, 5)
        self.conv3 = MFConv(conv_hidden, conv_hidden, 5)
        self.conv4 = MFConv(conv_hidden, conv_hidden, 5)

        self.linear1 = nn.Linear(conv_hidden, cls_hidden)
        self.linear2 = nn.Linear(cls_hidden, num_classses)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)

    
    def forward(self, x, edge_index):
        
        res = self.conv1(x, edge_index) 
        res = self.conv2(res, edge_index)
        res_2 = res
        res = self.conv3(res, edge_index)
        res_3 = res
        res = self.conv4(res, edge_index)
        
        res = res+res_3+res_2
          
        res = self.linear1(res)
        res = self.relu(res)
        res = self.drop1(res)
        res = self.linear2(res)

        return res
    
args = {
#     'lr': 0.01,
#     'epoch': 50,
#     'seed': 12345,
#     'save_path': './model/train_sum',   # 对应修改模型
#     'pos_weight': 3,
    'conv_hidden': 1024, 
    'cls_hidden': 1024,
    'n_layers': 3,
#     'max_degree': 5
}

device = "cuda" if torch.cuda.is_available() else "cpu"
# main(args, train_loader, val_loader)

model = Model(args).to("cuda")
model.load_state_dict(torch.load(args['save_path']))

explainer = Explainer(
    model=model,
    # ExplainerAlgorithm 类：Explainer 用于生成给定训练实例的 Explanation（s） 的可解释性算法
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',  # Model returns log probabilities.
    ),
)

for data in train_loader:
    data = data.to(device)
    x = data.x.to(torch.float32)
    name = data.name
    explanation = explainer(x, data.edge_index, index=2)
    path = './visualize_graph_train/{}.png'.format(name[0])
    explanation.visualize_graph(path,'networkx')
    print(f"Subgraph visualization plot'{name}' has been saved to '{path}'")
    
# for data in val_loader:
#     data = data.to(device)
#     x = data.x.to(torch.float32)
#     name = data.name
#     explanation = explainer(x, data.edge_index, index=2)
#     path = './visualize_graph_val/{}.png'.format(name[0])
#     explanation.visualize_graph(path,'networkx')
#     print(f"Subgraph visualization plot'{name}' has been saved to '{path}'")

# for data in test_loader:
#     data = data.to(device)
#     x = data.x.to(torch.float32)
#     name = data.name
#     explanation = explainer(x, data.edge_index, index=2)
#     path = './visualize_graph_test/{}.png'.format(name[0])
#     explanation.visualize_graph(path,'networkx')
#     print(f"Subgraph visualization plot'{name}' has been saved to '{path}'")
    
#     print(explanation.edge_mask)
#     print(explanation.node_mask)

#     feature_importance_path = './feature_importance/feature_importance{}.png'.format(i)
#     explanation.visualize_feature_importance(feature_importance_path,top_k=5)
#     print(f"Subgraph visualization plot has been saved to '{feature_importance_path}'")
    
# import random
# import os
# import numpy as np
# # 可显示数组中的所有元素，不受默认的截断限制
# np.set_printoptions(threshold=np.inf)
# def seed_torch(seed=42):
#     # 设置种子可以确保每次运行代码时生成的随机数相同。
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed) 
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) 
    
# # evaluation
# def top2(output, label):
#     sf = nn.Softmax(dim=1)
#     preds = sf(output)
#     preds = preds[:, 1]
#     _, indices = torch.topk(preds, 2)
#     pos_index = []
#     for i in range(label.shape[0]):
#         if label[i] == 1:
#             pos_index.append(i)  
#     for li in pos_index:
#         if li in indices:
#             return True
#     return False

# def MCC(output, label):
#     tn,fp,fn,tp=confusion_matrix(label, output).ravel()
#     up = (tp * tn) - (fp * fn)
#     down = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
#     return up / down

# def metrics(output, label):
#     tn,fp,fn,tp=confusion_matrix(label, output).ravel()
#     up = (tp * tn) - (fp * fn)
#     down = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
#     mcc = up / down
#     selectivity = tn / (tn + fp)
#     recall = tp / (tp + fn)
#     g_mean = (selectivity * recall) ** 0.5
#     balancedAccuracy = (recall + selectivity) / 2
#     return mcc, selectivity, recall, g_mean, balancedAccuracy


# def train(args, model, device, training_set, optimizer, criterion, epoch):
#     model.train()
#     sf = nn.Softmax(dim=1)
#     total_loss = 0
#     all_pred = []
#     all_pred_raw = []
#     all_labels = []
#     top2n = 0
#     for mol in training_set:
#         mol = mol.to(device)
#         mol.x = mol.x.to(torch.float32)
#         target = mol.y
        
#         optimizer.zero_grad()
#         output = model(mol.x, mol.edge_index)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         # tracking
#         top2n += top2(output, target)
#         all_pred.append(np.argmax(output.cpu().detach().numpy(), axis=1))
#         all_pred_raw.append(sf(output)[:, 1].cpu().detach().numpy())
#         all_labels.append(target.cpu().detach().numpy())

#     all_pred = np.concatenate(all_pred).ravel()
#     all_pred_raw = np.concatenate(all_pred_raw).ravel()
#     all_labels = np.concatenate(all_labels).ravel()

#     mcc = MCC(all_pred, all_labels)
#     print(f'Train Epoch: {epoch}, Ave Loss: {total_loss / len(training_set)} ACC: {accuracy_score(all_labels, all_pred)} Top2: {top2n / len(training_set)} AUC: {roc_auc_score(all_labels, all_pred_raw)} MCC: {mcc}')
#     return top2n / len(training_set)


# def val(args, model, device, val_set, optimizer, criterion, epoch):
#     model.eval()
#     sf = nn.Softmax(dim=1)
#     total_loss = 0
#     all_pred = []
#     all_pred_raw = []
#     all_labels = []
#     top2n = 0
#     for mol in val_set:
#         mol = mol.to(device)
#         mol.x = mol.x.to(torch.float32)
#         target = mol.y
#         optimizer.zero_grad()
#         output = model(mol.x, mol.edge_index)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         # tracking
#         top2n += top2(output, target)
#         all_pred.append(np.argmax(output.cpu().detach().numpy(), axis=1))
#         all_pred_raw.append(sf(output)[:, 1].cpu().detach().numpy())
#         all_labels.append(target.cpu().detach().numpy())
#     all_pred = np.concatenate(all_pred).ravel()
#     all_pred_raw = np.concatenate(all_pred_raw).ravel()
#     all_labels = np.concatenate(all_labels).ravel()
#     mcc = MCC(all_pred, all_labels)

#     print(f'Val Epoch: {epoch}, Ave Loss: {total_loss / len(val_set)} ACC: {accuracy_score(all_labels, all_pred)} Top2: {top2n / len(val_set)} AUC: {roc_auc_score(all_labels, all_pred_raw)} MCC: {mcc}')
#     return top2n / len(val_set)




# def main(args, train_loader, val_loader):
#     seed_torch(args['seed'])
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     torch.manual_seed(args['seed'])

#     model = Model(args).to(device)
#     print(model)
#     weights = torch.tensor([1, args['pos_weight']], dtype=torch.float32).to(device)
#     loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
#     optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#     max_top2 = 0
#     for epoch in range(1, args['epoch'] + 1):
#         train(args, model, device, train_loader, optimizer, loss_fn, epoch)
#         top2acc = val(args, model, device, val_loader, optimizer, loss_fn, epoch)
#         scheduler.step()
#         if top2acc > max_top2:
#             max_top2 = top2acc
#             print('Saving model (epoch = {:4d}, top2acc = {:.4f})'
#                 .format(epoch, max_top2))
#             torch.save(model.state_dict(), args['save_path'])
  
    
    # metric package:使用Explanation输出和可能的GNN模型/真实值来评估 ExplainerAlgorithm 的评估指标
    # 计算unfaithfulness()解释的
#     metric = unfaithfulness(explainer, explanation)
#     print(metric)
#     break
    
#     print(explanation.edge_mask)

# for epoch in range(30):
#     print(epoch)
#     for batch in train_loader:
#         batch = batch.to(device)
#         x = batch.x.to(torch.float32)
#         edge_index = batch.edge_index
#         target = batch.y
#         loss = explainer.algorithm.train(epoch, model, x, edge_index, target)
#         print(loss)
# print('finish')



