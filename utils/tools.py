from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.special import expit
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import LinearSVC

from .logger import error, info


def cube(x):
    return x ** 3


def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        o = np.reciprocal
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '/':
        o = np.divide
    elif o == '*':
        o = np.multiply
    elif o == 'stand_scaler':
        o = StandardScaler()
    elif o == 'minmax_scaler':
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == 'quan_trans':
        o = QuantileTransformer(random_state=0)
    elif o == 'exp':
        o = np.exp
    elif o == 'cube':
        o = cube
    elif o == 'sigmoid':
        o = expit
    elif o == 'log':
        o = np.log
    else:
        print('Please check your operation!')
    return o


def mi_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(mutual_info_regression(features[:, i].reshape
                                                     (-1, 1), y) - mutual_info_regression(features[:, j].reshape
                                                                                          (-1, 1), y))[0] / (
                               mutual_info_regression(features[:, i].
                                                      reshape(-1, 1), features[:, j].reshape(-1, 1))[
                                   0] + 1e-05))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat


# 注意这个y的shape
def cos_distance(feature, y):
    return pairwise_distances(feature.reshape(1, -1), y.reshape(1, -1), metric='cosine')


# 注意这个y的shape
def eu_distance(feature, y):
    return pairwise_distances(feature.reshape(1, -1), y.reshape(1, -1), metric='euclidean')


def eu_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(eu_distance(features[:, i], features[:, j])))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat


def cos_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(cos_distance(features[:, i], features[:, j])))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat


SUPPORT_DISTANCE_MODE = {
    'mi': mi_feature_distance,
    'eu': eu_feature_distance,
    'cos': cos_feature_distance
}


def feature_distance(feature, y, method='mi'):
    return SUPPORT_DISTANCE_MODE[method](feature, y)

'''
for ablation study
if mode == c then don't do cluster
'''
def cluster_features(features, y, cluster_num=2, mode='', distance_method='mi'):
    if mode == 'c':
        return _wocluster_features(features, y, cluster_num)
    else:
        return _cluster_features(features, y, cluster_num, distance_method)

def _cluster_features(features, y, cluster_num=2, distance_method='mi'):
    k = int(np.sqrt(features.shape[1]))
    features = feature_distance(features, y, method=distance_method)
    features = features.reshape(features.shape[0], -1)
    clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='single').fit(features)
    labels = clustering.labels_
    clusters = defaultdict(list)
    for ind, item in enumerate(labels):
        clusters[item].append(ind)
    return clusters

'''
return single column as cluster
'''
def _wocluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        clusters[item].append(ind)
    return clusters


class LinearAutoEncoder(nn.Module):

    def __init__(self, input, hidden, act=torch.relu):
        self.encoder = nn.Linear(input, hidden)
        self.encoder_act = act
        self.decoder = nn.Linear(hidden, input)
        self.decoder_act = act
        super().__init__()

    def forward(self, X):
        return self.decoder_act(self.decoder(self.encoder_act(self.encoder(X)))
                                )

    def generate(self, X):
        return self.encoder_act(self.encoder(X))


def Feature_GCN(X):
    """
    group feature 可能有一个cluster内元素为1的情况，这样corr - eye后返回的是一个零矩阵，故在这里设置为0时返回一个1.
    """
    corr_matrix = X.corr().abs()
    if len(corr_matrix) == 1:
        W = corr_matrix
    else:
        corr_matrix[np.isnan(corr_matrix)] = 0
        corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
        sum_vec = corr_matrix_.sum()
        for i in range(len(corr_matrix_)):
            corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
            corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
        W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)
    return Feature


class AutoEncoder(nn.Module):

    def __init__(self, N_feature):
        self.N_feature = N_feature
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(self.N_feature, 16), nn.Tanh
        (), nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(
            8, 16), nn.Tanh(), nn.Linear(16, self.N_feature))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def Feature_AE(X, gpu=-1):
    N_feature = X.shape[1]
    autoencoder = AutoEncoder(N_feature)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    X_tensor = torch.Tensor(X.values)
    if gpu >= 0:
        device = torch.device('cuda:' + str(gpu))
        autoencoder.to(device)
    else:
        device = torch.device('cpu')
    train_loader = Data.DataLoader(dataset=X_tensor, batch_size=128,
                                   shuffle=True, drop_last=False, num_workers=8)
    for epoch in range(10):
        for x in train_loader:
            b_x = x.view(-1, N_feature).float().to(device)
            encoded, decoded = autoencoder.forward(b_x)
            optimizer.zero_grad()
            loss = loss_func(decoded, b_x)
            loss.backward()
            optimizer.step()
    X_tensor.to(device)
    X_encoded = np.mean(autoencoder.forward(X_tensor)[0].cpu().detach().
                        numpy(), axis=1)
    return X_encoded


SUPPORT_STATE_METHOD = {
    'ds', 'ae', 'cg', 'ds+cg', 'ds+ae', 'ae+cg', 'ds+ae+cg'
}


def feature_state_generation(X, method='ds', gpu=-1):
    assert method in SUPPORT_STATE_METHOD
    if method == 'ds':
        return _feature_state_generation_des(X)
    elif method == 'cg':
        return Feature_GCN(X)
    elif method == 'ae':
        return Feature_AE(X, gpu)
    elif method == 'ds+cg':
        state_mds = _feature_state_generation_des(X)
        state_gcn = Feature_GCN(X)
        return np.append(state_gcn, state_mds)
    elif method == 'ae+cg':
        return np.append(Feature_AE(X, gpu),
                         Feature_GCN(X))
    elif method == 'ds+ae':
        return np.append(Feature_AE(X, gpu),
                         _feature_state_generation_des(X))
    elif method == 'ds+ae+cg':
        state_mds = _feature_state_generation_des(X)
        state_gcn = Feature_GCN(X)
        state_ae = Feature_AE(X, gpu)
        return np.append(np.append(state_ae, state_gcn), state_mds)
    else:
        error('Wrong feature state method')
        raise Exception('Wrong feature state method')


def _feature_state_generation_des(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
                                               describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix


def select_meta_cluster1(clusters, X, feature_names, epsilon, dqn_cluster,
                         method='mds', gpu=-1):
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
                                          method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    if np.random.uniform() > epsilon:
        act_id = np.argmax(q_vals)
    else:
        act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    f_cluster = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    info('current select feature name : ' + str(f_names))
    return action_emb, state_emb, f_cluster, f_names


def select_operation(f_cluster, operation_set, dqn_operation, steps_done,
                     method='mds', gpu=-1):
    op_state = feature_state_generation(pd.DataFrame(f_cluster), method, gpu)
    op_index = dqn_operation.choose_action(op_state, steps_done)
    op = operation_set[op_index]
    info('current select op : ' + str(op))
    return op_state, op, op_index


def select_meta_cluster2(clusters, X, feature_names, f_cluster1, op_emb,
                         epsilon, dqn_cluster, method='mds', gpu=-1):
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster1), method,
                                           gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
                                          method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0
        ]
        q_vals.append(q_value)
        cluster_list.append(key)
    act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    f_cluster2 = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    return action_emb, state_emb, f_cluster2, f_names


def operate_two_features(f_cluster1, f_cluster2, op, op_func, f_names1,
                         f_names2):
    if f_cluster1.shape[1] < f_cluster2.shape[1]:
        inds = np.random.randint(0, f_cluster2.shape[1], f_cluster1.shape[1])
        rand_fs = f_cluster2[:, inds]
        rand_names = f_names2[inds]
        f_generate = op_func(f_cluster1, rand_fs)
        final_name = [(str(f1_item) + op + str(rand_names[ind])) for ind,
                                                                     f1_item in enumerate(f_names1)]
    elif f_cluster1.shape[1] > f_cluster2.shape[1]:
        inds = np.random.randint(0, f_cluster1.shape[1], f_cluster2.shape[1])
        rand_fs = f_cluster1[:, inds]
        rand_names = f_names1[inds]
        f_generate = op_func(rand_fs, f_cluster2)
        final_name = [(str(f1_item) + op + str(f_names2[ind])) for ind,
                                                                   f1_item in enumerate(rand_names)]
    else:
        f_generate = op_func(f_cluster1, f_cluster2)
        final_name = [(str(f1_item) + op + str(f_names2[ind])) for ind,
                                                                   f1_item in enumerate(f_names1)]
    return f_generate, final_name


def operate_two_features_new(f_cluster1, f_cluster2, op, op_func, f_names1,
                             f_names2):
    feas, feas_names = [], []
    for i in range(f_cluster1.shape[1]):
        for j in range(f_cluster2.shape[1]):
            feas.append(op_func(f_cluster1[:, i], f_cluster2[:, j]))
            feas_names.append(str(f_names1[i]) + op + str(f_names2[j]))
    feas = np.array(feas)
    feas_names = np.array(feas_names)
    return feas.T, feas_names


def insert_generated_feature_to_original_feas(feas, f):
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data


def generate_next_state_of_meta_cluster1(X, y, dqn_cluster, cluster_num=2,
                                         method='mds', gpu=-1):
    clusters = cluster_features(X, y, cluster_num)
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
                                          method, gpu)
        q_value = dqn_cluster.get_q_value_next_state(state_emb, action).detach(
        ).numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
        action_list.append(action)
    act_emb = action_list[np.argmax(q_vals)]
    act_ind = cluster_list[np.argmax(q_vals)]
    f_cluster = X[:, list(clusters[act_ind])]
    return act_emb, state_emb, f_cluster, clusters


def generate_next_state_of_meta_operation(f_cluster_, operation_set,
                                          dqn_operation, method='mds', gpu=-1):
    op_state = feature_state_generation(pd.DataFrame(f_cluster_), method, gpu)
    op_index = dqn_operation.choose_next_action(op_state)
    op = operation_set[op_index]
    return op_state, op


def generate_next_state_of_meta_cluster2(f_cluster_, op_emb_, clusters, X,
                                         dqn_cluster, method='mds', gpu=-1):
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster_), method,
                                           gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb_)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
                                          method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value_next_state(state_emb, action).detach(
        ).numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    action_emb = action_list[np.argmax(q_vals)]
    return action_emb, state_emb


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'rank':
        pass
    else:
        return -1


def test_task_new(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list = [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average=
            'weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted')
                            )
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='macro'))
            ras.append(roc_auc_score(y_test, y_predict))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras)
    elif task == 'rank':
        pass
    else:
        return -1