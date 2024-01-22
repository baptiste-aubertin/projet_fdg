# encoding:utf-8
import sys

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from ..configx.configx import ConfigX

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Split ratings into five folds
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



def split_5_folds(configx):
    K = configx.k_fold_num
    names = ['user_id', 'item_id', 'rating']
    if not os.path.isfile(configx.rating_path):
        print("there is no such file:%s" % configx.rating_path)
        sys.exit()
    
    df = pd.read_csv(configx.rating_path, sep=configx.sep, names=names)
    
    # Ensure the user_id and item_id are integers
    df.user_id = df.user_id.astype(int)
    df.item_id = df.item_id.astype(int)

    # Remapping user_id and item_id
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    user_id_map = {old: new for new, old in enumerate(user_ids)}
    item_id_map = {old: new for new, old in enumerate(item_ids)}
    df['user_id'] = df['user_id'].map(user_id_map)
    df['item_id'] = df['item_id'].map(item_id_map)

    users = df.user_id.unique()

    ratings = coo_matrix((df.rating, (df.user_id, df.item_id)))
    
    # Convert to csr format
    try:
        ratings = ratings.tocsr()
    except Exception as e:
        print(f"Error converting to CSR format: {e}")
        return


    rows = list()
    cols = list()
    vals = list()
    nonzeros = list()
    print('D1')
    for k in range(K):
        size_of_bucket = int(ratings.nnz / K)
        if k == K - 1:
            size_of_bucket += ratings.nnz % K
        rows.append(np.zeros(size_of_bucket))
        cols.append(np.zeros(size_of_bucket))
        vals.append(np.zeros(size_of_bucket))
        nonzeros.append(0)
    print('D2')
    for i, user in enumerate(users):
        items = ratings[user, :].indices
        rating_vals = ratings[user, :].data
        index_list = [i for i in range(K)] * int(len(items) / float(K) + 1)
        np.random.shuffle(index_list)
        index_list = np.array(index_list)

        for k in range(K):
            k_index_list = (index_list[:len(items)] == k)
            from_ind = nonzeros[k]
            to_ind = nonzeros[k] + sum(k_index_list)

            if to_ind >= len(rows[k]):
                rows[k] = np.append(rows[k], np.zeros(size_of_bucket))
                cols[k] = np.append(cols[k], np.zeros(size_of_bucket))
                vals[k] = np.append(vals[k], np.zeros(size_of_bucket))
                k_index_list = (index_list[:len(items)] == k)

            rows[k][from_ind:to_ind] = [user] * sum(k_index_list)
            cols[k][from_ind:to_ind] = items[k_index_list]
            vals[k][from_ind:to_ind] = rating_vals[k_index_list]
            nonzeros[k] += sum(k_index_list)
    print('D3')
    if not os.path.exists(configx.rating_cv_path):
        os.makedirs(configx.rating_cv_path)
        print(f'{configx.rating_cv_path} folder has been established.')
    print('D4')
    for k, (row, col, val, nonzero) in enumerate(zip(rows, cols, vals, nonzeros)):
        bucket_df = pd.DataFrame({'user': row[:nonzero], 'item': col[:nonzero], 'rating': val[:nonzero]},
                                 columns=['user', 'item', 'rating'])
        bucket_df.to_csv(os.path.join(configx.rating_cv_path, "%s-%d.csv" % (configx.dataset_name, k)), sep=configx.sep, header=False, index=False)
        print("%s -fold%d data generated finished!" % (configx.dataset_name, k))

    print("All Data Generated Done!")


if __name__ == "__main__":
    configx = ConfigX()
    split_5_folds(configx)
