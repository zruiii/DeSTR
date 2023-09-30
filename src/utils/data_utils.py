import csv
import os
import math
import numpy as np
import pandas as pd
import pickle as pkl

import random
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pdb

import dgl
import torch

# function for STGODE
def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


# 同时加载多个城市数据
class Multi_Data_Container(object):
    def __init__(self,
                 data_path,
                 data_dict,
                 look_back,
                 pred_len,
                 train_ratio=0.6,
                 val_ratio=0.2,
                 graph="DIST",              # DTW & DIST & GAU
                 time_encode="raw"):
        super().__init__()
        """ merge multi-source datasets """
        flow_df, rel_df, dtw_li, nfeat_df = [], [], [], []
        min_time, max_time = None, None
        for i, (key, value) in enumerate(data_dict.items()):
            temp_df = pd.read_csv(os.path.join(data_path, key, value["flow"]),
                                  index_col="start_time",
                                  parse_dates=True)
            if i == 0 or min_time > temp_df.index.min():
                min_time = temp_df.index.min()

            if i == 0 or max_time < temp_df.index.max():
                max_time = temp_df.index.max()

            flow_df.append(temp_df)
            rel_df.append(pd.read_csv(os.path.join(data_path, key, value["rel"])))
            nfeat_df.append(pd.read_csv(os.path.join(data_path, key, value["nfeat"])))
            dtw_li.append(np.load(os.path.join(data_path, key, value["dtw"])))

        data_range = pd.date_range(min_time, max_time, freq=pd.Timedelta(minutes=5))
        data_range = pd.Series(data=range(len(data_range)), index=data_range)
        data_range = pd.DataFrame({'index': data_range})
        for temp_df in flow_df:
            temp_df = pd.merge(data_range, temp_df, left_index=True, right_index=True,
                               how='left').drop(['index'], axis=1)

        flow_df = pd.concat(flow_df, axis=1)
        rel_df = pd.concat(rel_df, axis=0)
        nfeat_df = pd.concat(nfeat_df, axis=0)

        flow_df = flow_df.where(flow_df >= 0, np.nan)
        arr_flow = flow_df.values.copy().astype(np.float32)
        flow_mask = np.isnan(arr_flow)

        """ global ID """
        id_dict = {x: i for i, x in enumerate(flow_df.columns)}
        with open(os.path.join(data_path, "full_id.pkl"), "wb") as f:
            pkl.dump(id_dict, f)
        nids = list(id_dict.values())

        """ data split: 60% training, 20% test, 20% eval """
        L = arr_flow.shape[0]
        train_steps = round(train_ratio * L)
        val_steps = round(val_ratio * L)
        start_points = [0, train_steps - look_back, train_steps + val_steps - look_back]
        end_points = [train_steps, train_steps + val_steps, L]

        """ Normalization """
        org_flow = arr_flow.copy()
        scaler = StandardScaler()
        scaler.fit(arr_flow[start_points[0]:end_points[0]])
        arr_flow = scaler.transform(arr_flow)

        """ Time Feature """
        time_index = flow_df.index
        time_feat = np.vstack([
            time_index.month.values, time_index.day.values, time_index.hour.values,
            time_index.minute.values, time_index.weekday.values,
            time_index.isocalendar().week.values.astype(np.int64)
        ]).T

        if time_encode == "cyclic":
            time_feat = self._cyclic_time_encode(time_feat)  # 周期特征
        elif time_encode == "scale":
            time_scaler = StandardScaler()
            time_scaler.fit(time_feat[start_points[0]:end_points[0]])
            time_feat = time_scaler.transform(time_feat)
        else:
            # 只保留 day hour miniute weekday
            time_feat = time_feat[:, [1, 2, 3, 4]]  # 原始特征(t2v or nn.Embedding)
        time_feat = time_feat.astype(np.float32)

        """ road network """
        if graph == "DTW":
            arr_adj = self.get_dtw_matrix(dtw_li, len(nids))
        elif graph == "DIST":
            arr_adj = self.get_adjacency_matrix(rel_df, id_dict)
        elif graph == "GAU":
            arr_adj = self.get_normalized_adjacency_matrix(rel_df, id_dict)

        src, dst = arr_adj.nonzero()
        weights = arr_adj[src, dst]

        g = dgl.graph((src, dst))
        g.add_nodes(arr_adj.shape[0] - g.number_of_nodes())
        g.edata['w'] = torch.FloatTensor(weights)

        """ node features """
        nfeat = self.get_node_feature(nfeat_df, id_dict)

        # Road Network
        self.g = g  # dgl.Graph
        self.adj = arr_adj  # Road Matrix
        self.nids = nids  # Flow IDs
        self.nfeat = nfeat  # Node Feature

        # Time
        self.time_feat = time_feat  # Temporal Feature

        # Signal
        self.start_points = start_points  # start time-point
        self.end_points = end_points  # end time-point
        self.flow_mask = ~flow_mask  # mask for missing records
        self.arr_flow = arr_flow  # normalized data
        self.org_flow = org_flow  # raw data
        self.scaler = scaler  # data scaler

        self.look_back = look_back  # window size
        self.pred_len = pred_len  # predict size

    def _cyclic_time_encode(self, time_feat):
        """ extract time features
        time_feat: [L, C] 年 月 日 小时 分钟 星期日 星期
        """
        month, day, hour, minute, weekday, week = np.split(time_feat, 6, 1)

        # year
        month_of_year_sin, month_of_year_cos = self._encode_cyclic_feature(month, 12)
        day_of_year_sin, day_of_year_cos = self._encode_cyclic_feature(day, 365)
        week_of_year_sin, week_of_year_cos = self._encode_cyclic_feature(week, 52)

        # month
        day_of_month_sin, day_of_month_cos = self._encode_cyclic_feature(day, 30)

        # week
        day_of_week_sin, day_of_week_cos = self._encode_cyclic_feature(weekday, 7)

        # day
        hour_of_day_sin, hour_of_day_cos = self._encode_cyclic_feature(hour, 24)
        min_of_day_sin, min_of_day_cos = self._encode_cyclic_feature(hour * 60 + minute, 60 * 24)

        time_feat = np.hstack((hour_of_day_sin, hour_of_day_cos, min_of_day_sin, min_of_day_cos,
                               day_of_week_sin, day_of_week_cos, day_of_month_sin, day_of_month_cos))

        return time_feat.astype(np.float32)

    def _encode_cyclic_feature(self, value, max_val):
        try:
            value = 2 * np.pi * value / max_val
        except:
            pdb.set_trace()
        return np.sin(value), np.cos(value)

    def get_adjacency_matrix(self, rel_df, id_dict):
        num_nodes = len(id_dict)
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for _, row in rel_df.iterrows():
            if row[0] in id_dict and row[1] in id_dict:
                A[id_dict[row[0]], id_dict[row[1]]] = row[2]
            else:
                continue
        
        return A
    
    def get_normalized_adjacency_matrix(self, rel_df, id_dict, thre=0.1):
        """ thresholded Gaussian kernel to construct adjacency matrix like GraphWaveNet """
        num_nodes = len(id_dict)
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        A[:] = np.inf
        for _, row in rel_df.iterrows():
            if row[0] in id_dict and row[1] in id_dict:
                A[id_dict[row[0]], id_dict[row[1]]] = row[2]
            else:
                continue
        
        distances = A[~np.isinf(A)].flatten()
        adj = np.exp(-np.square(A / distances.std()))
        adj[adj < thre] = 0
        return adj


    def get_dtw_matrix(self, dtw_li, num_nodes):
        """ binary DTW matrix """
        row_idx, col_idx = 0, 0
        dtw_array = np.zeros((num_nodes, num_nodes))
        for dtw in dtw_li:
            high, weight = dtw.shape
            dtw_array[row_idx:row_idx+high, col_idx:col_idx+weight] = dtw
            row_idx += high
            col_idx += weight

        return dtw_array


    def get_node_feature(self, nfeat_df, id_dict):
        tf = set(id_dict.keys())
        nfeat_df = nfeat_df[nfeat_df['id'].isin(tf)]
        assert nfeat_df.shape[0] == len(id_dict)

        flow = nfeat_df['flow'].values
        lane_feat = nfeat_df.iloc[:, 2:].values
        flow_feat = np.zeros((len(flow), 4))
        flow_feat[np.arange(len(flow)), flow - 1] = 1
        feature = np.hstack((flow_feat, lane_feat))
        return feature.astype(np.float32)


# 单独加载单个城市数据
class Single_Data_Container(object):
    def __init__(self,
                 data_path,
                 data_dict,
                 look_back,
                 pred_len,
                 train_ratio=0.6,
                 val_ratio=0.2,
                 graph="DIST",              # DTW & DIST & GAU
                 time_encode="raw"):
        super().__init__()
        """ single-source dataset """
        key = next(iter(data_dict))
        value = data_dict[key]
        flow_df = pd.read_csv(os.path.join(data_path, key, value["flow"]),
                              index_col="start_time",
                              parse_dates=True)
        rel_df = pd.read_csv(os.path.join(data_path, key, value["rel"]))
        nfeat_df = pd.read_csv(os.path.join(data_path, key, value["nfeat"]))
        dtw_arr = np.load(os.path.join(data_path, key, value["dtw"]))

        flow_df = flow_df.where(flow_df >= 0, np.nan)
        arr_flow = flow_df.values.copy().astype(np.float32)
        flow_mask = np.isnan(arr_flow)

        """ global ID """
        id_dict = {x: i for i, x in enumerate(flow_df.columns)}
        with open(os.path.join(data_path, "full_id.pkl"), "rb") as f:
            global_id_dict = pkl.load(f)
        nids = list([global_id_dict[x] for x in id_dict.keys()])

        """ data split: 60% training, 20% test, 20% eval """
        L = arr_flow.shape[0]
        train_steps = round(train_ratio * L)
        val_steps = round(val_ratio * L)
        start_points = [0, train_steps - look_back, train_steps + val_steps - look_back]
        end_points = [train_steps, train_steps + val_steps, L]

        """ Normalization """
        org_flow = arr_flow.copy()
        scaler = StandardScaler()
        scaler.fit(arr_flow[start_points[0]:end_points[0]])
        arr_flow = scaler.transform(arr_flow)

        """ Time Feature """
        time_index = flow_df.index
        time_feat = np.vstack([
            time_index.month.values, time_index.day.values, time_index.hour.values,
            time_index.minute.values, time_index.weekday.values,
            time_index.isocalendar().week.values.astype(np.int64)
        ]).T

        if time_encode == "cyclic":
            time_feat = self._cyclic_time_encode(time_feat)  # 周期特征
        elif time_encode == "scale":
            time_scaler = StandardScaler()
            time_scaler.fit(time_feat[start_points[0]:end_points[0]])
            time_feat = time_scaler.transform(time_feat)
        else:
            # 只保留 day hour miniute weekday
            time_feat = time_feat[:, [1, 2, 3, 4]]  # 原始特征(t2v or nn.Embedding)
        time_feat = time_feat.astype(np.float32)

        """ road network """
        if graph == "DTW":
            arr_adj = dtw_arr
        elif graph == "DIST":
            arr_adj = self.get_adjacency_matrix(rel_df, id_dict)
        elif graph == "GAU":
            arr_adj = self.get_normalized_adjacency_matrix(rel_df, id_dict)

        src, dst = arr_adj.nonzero()
        weights = arr_adj[src, dst]
        
        g = dgl.graph((src, dst))
        g.add_nodes(arr_adj.shape[0] - g.number_of_nodes())
        g.edata['w'] = torch.FloatTensor(weights)

        """ node features """
        nfeat = self.get_node_feature(nfeat_df, id_dict)

        # Road Network
        self.g = g  # dgl.Graph
        self.adj = arr_adj  # Road Matrix
        self.nids = nids  # Flow IDs
        self.nfeat = nfeat  # Node Feature

        # Time
        self.time_feat = time_feat  # Temporal Feature

        # Signal
        self.start_points = start_points  # start time-point
        self.end_points = end_points  # end time-point
        self.flow_mask = ~flow_mask  # mask for missing records
        self.arr_flow = arr_flow  # normalized data
        self.org_flow = org_flow  # raw data
        self.scaler = scaler  # data scaler

        self.look_back = look_back  # window size
        self.pred_len = pred_len  # predict size
    
    def _cyclic_time_encode(self, time_feat):
        """ extract time features
        time_feat: [L, C] 年 月 日 小时 分钟 星期日 星期
        """
        month, day, hour, minute, weekday, week = np.split(time_feat, 6, 1)

        # year
        month_of_year_sin, month_of_year_cos = self._encode_cyclic_feature(month, 12)
        day_of_year_sin, day_of_year_cos = self._encode_cyclic_feature(day, 365)
        week_of_year_sin, week_of_year_cos = self._encode_cyclic_feature(week, 52)

        # month
        day_of_month_sin, day_of_month_cos = self._encode_cyclic_feature(day, 30)

        # week
        day_of_week_sin, day_of_week_cos = self._encode_cyclic_feature(weekday, 7)

        # day
        hour_of_day_sin, hour_of_day_cos = self._encode_cyclic_feature(hour, 24)
        min_of_day_sin, min_of_day_cos = self._encode_cyclic_feature(hour * 60 + minute, 60 * 24)

        time_feat = np.hstack((hour_of_day_sin, hour_of_day_cos, min_of_day_sin, min_of_day_cos,
                               day_of_week_sin, day_of_week_cos, day_of_month_sin, day_of_month_cos))

        return time_feat.astype(np.float32)

    def _encode_cyclic_feature(self, value, max_val):
        try:
            value = 2 * np.pi * value / max_val
        except:
            pdb.set_trace()
        return np.sin(value), np.cos(value)

    def get_adjacency_matrix(self, rel_df, id_dict):
        """ thresholded Gaussian kernel to construct adjacency matrix like GraphWaveNet """
        num_nodes = len(id_dict)
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for _, row in rel_df.iterrows():
            if row[0] in id_dict and row[1] in id_dict:
                A[id_dict[row[0]], id_dict[row[1]]] = row[2]
            else:
                continue
        
        return A
    
    def get_normalized_adjacency_matrix(self, rel_df, id_dict, thre=0.1):
        """ thresholded Gaussian kernel to construct adjacency matrix like GraphWaveNet """
        num_nodes = len(id_dict)
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        A[:] = np.inf
        for _, row in rel_df.iterrows():
            if row[0] in id_dict and row[1] in id_dict:
                A[id_dict[row[0]], id_dict[row[1]]] = row[2]
            else:
                continue
        
        distances = A[~np.isinf(A)].flatten()
        adj = np.exp(-np.square(A / distances.std()))
        adj[adj < thre] = 0
        return adj

    def get_node_feature(self, nfeat_df, id_dict):
        tf = set(id_dict.keys())
        nfeat_df = nfeat_df[nfeat_df['id'].isin(tf)]
        assert nfeat_df.shape[0] == len(id_dict)

        flow = nfeat_df['flow'].values
        lane_feat = nfeat_df.iloc[:, 2:].values
        flow_feat = np.zeros((len(flow), 4))
        flow_feat[np.arange(len(flow)), flow - 1] = 1
        feature = np.hstack((flow_feat, lane_feat))
        return feature.astype(np.float32)


if __name__ == "__main__":
    # Multi Data
    data_dict = {
        "yizhuang": {
            "flow": "2023_01_03_flow.csv",
            "rel": "connect.csv",
            "dtw": "dtw_distance.npy",
            "nfeat": "node_feature.csv"
        },
        "guangzhou": {
            "flow": "2023_01_03_flow.csv",
            "rel": "connect.csv",
            "dtw": "dtw_distance.npy",
            "nfeat": "node_feature.csv"
        },
        "zhuzhou": {
            "flow": "2023_01_03_flow.csv",
            "rel": "connect.csv",
            "dtw": "dtw_distance.npy",
            "nfeat": "node_feature.csv"
        }
    }

    # data = Multi_Data_Container(data_path="/home/users/zharui/decoupledST/data",
    #                             data_dict=data_dict,
    #                             look_back=24,
    #                             pred_len=12,
    #                             graph="DTW")
    # pdb.set_trace()


    # Single Data
    data_dict = {
        "zhuzhou": {
            "flow": "2023_01_03_flow.csv",
            "rel": "connect.csv",
            "dtw": "dtw_distance.npy",
            "nfeat": "node_feature.csv"
        }
    }

    data = Single_Data_Container(data_path="/home/users/zharui/decoupledST/data",
                                data_dict=data_dict,
                                look_back=24,
                                pred_len=12,
                                time_encode="cyclic")
    pdb.set_trace()
