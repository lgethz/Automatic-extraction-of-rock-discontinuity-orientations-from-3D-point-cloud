import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.spatial as spatial


def load_data(filedir, filename, dtype=np.float32, wo=False, usecols=None):
    d = None
    filepath = os.path.join(filedir, filename + '.xyz')
    if os.path.exists(filepath):
        d = np.load(filepath)
    else:
        d = np.loadtxt(
            os.path.join(filedir, filename),
            dtype=dtype,
            usecols=usecols,
        )
        np.save(filepath, d)
    return d


class BaseDataset:
    def __init__(self, data_path, evaluate):
        path = Path(data_path).expanduser().resolve()
        self.evaluate = evaluate

        if path.is_file():
            if path.suffix.lower() != '.xyz':
                raise ValueError(
                    'Direct input file must have a .xyz extension: %s' % path
                )
            self.data_dir = str(path.parent)
            cur_sets = [path.stem]
        elif path.is_dir():
            self.data_dir = str(path)
            xyz_files = sorted(
                (
                    item
                    for item in path.iterdir()
                    if item.is_file()
                    and item.suffix.lower() == '.xyz'
                    and not item.stem.lower().endswith('_gt')
                ),
                key=lambda item: item.name.casefold(),
            )
            if not xyz_files:
                raise ValueError('No input .xyz files found directly in %s' % path)
            cur_sets = [item.stem for item in xyz_files]
            if len({name.casefold() for name in cur_sets}) != len(cur_sets):
                raise ValueError(
                    'Input directory has duplicate .xyz file names: %s' % path
                )
        else:
            raise FileNotFoundError(
                'Direct input path does not exist or is not a file/directory: %s'
                % path
            )

        for shape_name in cur_sets:
            print("  ", shape_name)
        self.cur_sets = cur_sets

    def get_data(self, shape_name):
        pcl = load_data(
            filedir=self.data_dir,
            filename=shape_name + '.xyz',
            dtype=np.float32,
            usecols=(0, 1, 2),
        )[:, :3]
        normal_gt = None
        if self.evaluate:
            normal_gt = load_data(
                filedir=self.data_dir,
                filename=shape_name + '_gt.xyz',
                dtype=np.float32,
            )[:, -3:]
            assert pcl.shape == normal_gt.shape
        return pcl, normal_gt
