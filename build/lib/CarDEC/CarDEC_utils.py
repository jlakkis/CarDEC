#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from scipy.sparse import issparse

import scanpy as sc
from anndata import AnnData


# In[ ]:


def normalize_scanpy(adata, batch_key = None, n_high_var=1000, LVG = True, 
                     normalize_samples = True, log_normalize = True, 
                     normalize_features = True):
    
    n, p = adata.shape
    sparsemode = issparse(adata.X)
    
    if batch_key is not None:
        batch = list(adata.obs[batch_key])
        batch = convert_vector_to_encoding(batch)
        batch = np.asarray(batch)
        batch = batch.astype('float32')
    else:
        batch = np.ones((n,), dtype = 'float32')
        norm_by_batch = False
        
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
        
    count = adata.X.copy()
        
    if normalize_samples:
        out = sc.pp.normalize_total(adata, inplace = False)
        obs_ = adata.obs
        var_ = adata.var
        adata = None
        adata = AnnData(out['X'])
        adata.obs = obs_
        adata.var = var_
        
        size_factors = out['norm_factor'] / np.median(out['norm_factor'])
        out = None
    else:
        size_factors = np.ones((adata.shape[0], ))
        
    if not log_normalize:
        adata_ = adata.copy()
    
    sc.pp.log1p(adata)
    
    if n_high_var is not None:
        sc.pp.highly_variable_genes(adata, inplace = True, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, 
                                          n_bins = 20, n_top_genes = n_high_var, batch_key = batch_key)
        
        hvg = adata.var['highly_variable'].values
        
        if not log_normalize:
            adata = adata_.copy()

    else:
        hvg = [True] * adata.shape[1]
        
    if normalize_features:
        batch_list = np.unique(batch)

        if sparsemode:
            adata.X = adata.X.toarray()

        for batch_ in batch_list:
            indices = [x == batch_ for x in batch]
            sub_adata = adata[indices]
            
            sc.pp.scale(sub_adata)
            adata[indices] = sub_adata.X
        
        adata.layers["normalized input"] = adata.X
        adata.X = count
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]
            
    else:
        if sparsemode:   
            adata.layers["normalized input"] = adata.X.toarray()
        else:
            adata.layers["normalized input"] = adata.X
            
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]
        
    if n_high_var is not None:
        del_keys = ['dispersions', 'dispersions_norm', 'highly_variable', 'highly_variable_intersection', 'highly_variable_nbatches', 'means']
        del_keys = [x for x in del_keys if x in adata.var.keys()]
        adata.var = adata.var.drop(del_keys, axis = 1)
            
    y = np.unique(batch)
    num_batch = len(y)
    
    adata.obs['size factors'] = size_factors.astype('float32')
    adata.obs['batch'] = batch
    adata.uns['num_batch'] = num_batch
    
    if sparsemode:
        adata.X = adata.X.toarray()
        
    if not LVG:
        adata = adata[:, adata.var['Variance Type'] == 'HVG']
        
    return adata


# In[ ]:


def build_dir(dir_path):
    subdirs = [dir_path]
    substring = dir_path

    while substring != '':
        splt_dir = os.path.split(substring)
        substring = splt_dir[0]
        subdirs.append(substring)
        
    subdirs.pop()
    subdirs = [x for x in subdirs if os.path.basename(x) != '..']

    n = len(subdirs)
    subdirs = [subdirs[n - 1 - x] for x in range(n)]
    
    for dir_ in subdirs:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)


# In[ ]:


def convert_string_to_encoding(string, vector_key):
    """A function to convert a string to a numeric encoding"""
    return np.argwhere(vector_key == string)[0][0]


# In[ ]:


def convert_vector_to_encoding(vector):
    """A function to convert a vector of strings to a dense numeric encoding"""
    vector_key = np.unique(vector)
    vector_strings = list(vector)
    vector_num = [convert_string_to_encoding(string, vector_key) for string in vector_strings]
    
    return vector_num


# In[ ]:


def find_resolution(adata_, n_clusters, random): 
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        adata = sc.tl.louvain(adata_, resolution = current_res, random_state = random, copy = True)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))
        
        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        
        iteration = iteration + 1
        
    return current_res

