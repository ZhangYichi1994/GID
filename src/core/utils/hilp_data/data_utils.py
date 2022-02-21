# The code below are borrowed from https://github.com/lucfra/LDS-GNN/blob/master/ and make some modifies.
import os
import pickle
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import kneighbors_graph

import torch
from ..generic_utils import *


class Config:
    """ Base class of a configuration instance; offers keyword initialization with easy defaults,
    pretty printing and grid search!
    """
    def __init__(self, **kwargs):
        self._version = 1
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))

    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items())) + ']\n'

    @classmethod
    def default_instance(cls):
        return cls()

    @classmethod
    def grid(cls, **kwargs):
        """Builds a mesh grid with given keyword arguments for this Config class.
        If the value is not a list, then it is considered fixed"""

        class MncDc:
            """This is because np.meshgrid does not always work properly..."""

            def __init__(self, a):
                self.a = a  # tuple!

            def __call__(self):
                return self.a

        sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
        for k, v in sin.items():
            copy_v = []
            for e in v:
                copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
            sin[k] = copy_v

        grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
        return [cls(**far.utils.merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
        )) for vv in grd]



class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        self.f1 = 'load_data_del_edges'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        if self.f2:
            res = eval(self.f2)(res, **self.kwargs_f2, seed=self.seed)
        return res



class HILP(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        self.one_class = False
        self.eliminate_small_part = False
        super().__init__(**kwargs)

    def calDis(self, t1, t2):
        dis = np.linalg.norm(t1- t2)
        return dis

    def load(self, data_dir=None, knn_size=None, epsilon=None, knn_metric='cosine', one_class=None):
        assert (knn_size is None) or (epsilon is None)
          
        if self.dataset_name == 'HardinLoopPlatform':
            scale_ = True
            import pandas as pd
            dataRead = pd.read_csv('../data/HardinLoopPlatform/DataPreprocessed.csv', header=None)
            dataArray = np.array(dataRead)
            np.random.shuffle(dataArray)
            dataArray = dataArray[:, 1:]

            trainData = dataArray[:, 1:]
            trainLabel = dataArray[:, 0]

        
        else:
            raise AttributeError('dataset not available')

        if (self.dataset_name == 'HardinLoopPlatform'):
            if scale_:
                from sklearn.preprocessing import scale
                features = scale(trainData)
            else:
                # normalization
                from sklearn.preprocessing import scale
                trainData /= trainData.sum(1).reshape(-1,1)
                features = scale(trainData)
                
            y = trainLabel.astype('int64')

        else:
            pass
        ys = LabelBinarizer().fit_transform(y)
        if ys.shape[1] == 1:
            ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]
        from sklearn.model_selection import train_test_split

        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val,
                                                        test_size=n - self.n_train - self.n_val,
                                                        stratify=y)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                    train_size=self.n_train, test_size=self.n_val,
                                                    stratify=y_train)

        features = torch.Tensor(features)
        labels = torch.LongTensor(np.argmax(ys, axis=1))
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)


        if not knn_size is None:
            adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
            adj_norm = normalize_sparse_adj(adj)
            adj_norm = torch.Tensor(adj_norm.todense())
        elif not epsilon is None:
            feature_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
            attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
            mask = (attention > epsilon).float()
            adj = attention * mask
            adj = (adj > 0).float()
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = None

        return adj_norm, features, labels, idx_train, idx_val, idx_test

