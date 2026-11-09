import os
import sys
import time
import numpy as np
import scipy.spatial as spatial

def load_data(filedir, filename, dtype=np.float32, wo=False):
    d = None
    filepath = os.path.join(filedir, filename + '.xyz')
    if os.path.exists(filepath):
        d = np.load(filepath)
    else:
        d = np.loadtxt(os.path.join(filedir, filename), dtype=dtype)
        np.save(filepath, d)
    return d

class BaseDataset():
    def __init__(self, root, data_set, data_list, evaluate):
        self.data_dir = os.path.join(root, data_set, data_list)
        self.evaluate = evaluate
        if len(data_list) > 0:
            cur_sets = []
            with open(os.path.join(root, data_set, data_list, 'list', data_list+'.txt')) as f:
                cur_sets = f.readlines()
            cur_sets = [x.strip() for x in cur_sets]
            cur_sets = list(filter(None, cur_sets))
        else:
            raise ValueError('Data list neet to be given.')
        for s in cur_sets:
            print("  ", s)
        self.cur_sets = cur_sets
    
    def get_data(self, shape_name):
        pcl = load_data(filedir=self.data_dir, filename=shape_name + '.xyz', dtype=np.float32)[:, :3]
        normal_gt = None
        if self.evaluate:
            normal_gt = load_data(filedir=self.data_dir, filename=shape_name+'_gt.xyz', dtype=np.float32)[:, -3:]
            assert pcl.shape == normal_gt.shape
        return pcl, normal_gt
