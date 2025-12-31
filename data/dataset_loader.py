"""Dataset Loaders"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph

class NSLKDDLoader:
    def load(self, filepath, k_neighbors=5):
        columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes'] + [f'f{i}' for i in range(35)] + ['label', 'difficulty']
        df = pd.read_csv(filepath, names=columns)
        
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        label_map = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
        df['label'] = df['label'].str.lower().str.split('.').str[0].map(label_map).fillna(0)
        
        X = df.drop(['label', 'difficulty'], axis=1).values
        y = df['label'].values.astype(int)
        X = StandardScaler().fit_transform(X)
        
        A = kneighbors_graph(X[:1000], k_neighbors, mode='connectivity')  # Sample for speed
        edge_index = torch.LongTensor(np.array(A.nonzero()))
        
        return torch.FloatTensor(X), edge_index, torch.LongTensor(y)
