from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re 
import os
import torch
import os.path as osp
from sklearn.decomposition import PCA,TruncatedSVD  
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import pickle  

class StanfordCars(object):
    """
    Dataset statistics:
    # 130  (train) + 17 (val) + 49 (test)
    """

    dataset_dir = '/data/cjl_data/StanfordCars'

    def __init__(self,base):
        super(StanfordCars, self).__init__()
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')
        
        wordpkl_path = os.path.join(self.dataset_dir, 'car_label_embeddings.pkl')#chang_
        with open(wordpkl_path, 'rb') as f:  
            self.wordvecdir = pickle.load(f)  
        print(base)
        if(base=="C"):
            pca = PCA(n_components=64)           
            self.wordvecdir = pca.fit_transform(np.array(self.wordvecdir))
            print("PCA OK")

        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> StanfordCars loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path):
        cat_container = sorted(os.listdir(dir_path))
        cats2label = {cat:label for label, cat in enumerate(cat_container)}

        dataset = []
        labels = []
        for cat in cat_container:
            for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):
                if '.jpg' not in img_path:
                    continue
                label = cats2label[cat]
                match = re.search(r'\d+', cat) 
                wordvec = self.wordvecdir[int(match.group(0))-1]
                dataset.append((os.path.join(dir_path, cat, img_path), label, wordvec))
                labels.append(label)

        labels2inds = {}
        for idx, label in enumerate(labels):
            if label not in labels2inds:
                labels2inds[label] = []
            labels2inds[label].append(idx)

        labelIds = sorted(labels2inds.keys())
        return dataset, labels2inds, labelIds
