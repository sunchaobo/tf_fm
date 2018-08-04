#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
import sys

def load_query_embedding(filename):
    query_embedding_dict = {}
    with open(filename) as fh:
        for line in fh:
            items = line.strip().split(' ')
            query = items[0]
            query_embedding = np.array([float(x) for x in items[1:]])
            norm = np.linalg.norm(query_embedding)
            query_embedding /= norm
            query_embedding_dict[query] = query_embedding

    return query_embedding_dict


def process_id(filename, user_dict, query_dict):
    count = 0
    X_row = []
    X_col = []
    X_data = []

    U = []
    Q = []
    y = []
    with open(filename) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            try:
                label = int(parts[0])
            except:
                continue

            uid = parts[1]
            query_info = parts[2]
            try:
                query, user_desc = query_info.split('#')
            except:
                continue

            user_id = len(user_dict)
            if uid in user_dict:
                user_id = user_dict[uid]
            else:
                user_dict[uid] = user_id

            query_id = len(query_dict)
            if query in query_dict:
                query_id = query_dict[query]
            else:
                query_dict[query] = query_id

            tree_idx = 0
            #x = np.zeros(31 * 120)
            c_row = []
            c_col = []
            c_data = []
            for leaf in parts[3:]:
                leaf_idx = int(leaf)
                feature_idx = tree_idx * 31 + leaf_idx
                if feature_idx >= 31 * 120:
                    break

                c_row.append(count)
                c_col.append(feature_idx)
                c_data.append(1.0)
                tree_idx += 1
                #x[feature_idx] = 1.0

            X_row.extend(c_row)
            X_col.extend(c_col)
            X_data.extend(c_data)
            #X.append(x)
            U.append(user_id)
            Q.append(query_id)
            y.append(label)

            count += 1
            if count % 100000 == 0:
                print count

    print len(X_row), len(X_col)
    X = csr_matrix((X_data, (X_row, X_col)), shape=(count, 31 * 120))
    #X = np.array(X)
    U = np.array(U)
    Q = np.array(Q)
    y = np.array(y)
    return X, U, Q, y, user_dict, query_dict

def process(filename, query_embedding_dict):
    count = 0
    X_row = []
    X_col = []
    X_data = []

    X = []
    U = []
    Q = []
    y = []
    with open(filename) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            try:
                label = int(parts[0])
            except:
                continue

            uid = parts[1]
            query_info = parts[2]
            try:
                query, user_desc = query_info.split('#')
                user_embedding = [float(x) for x in user_desc.strip().split(' ')]
            except:
                continue

            tree_idx = 0
            #x = np.zeros(31 * 120)
            c_row = []
            c_col = []
            c_data = []
            for leaf in parts[3:]:
                leaf_idx = int(leaf)
                feature_idx = tree_idx * 31 + leaf_idx
                if feature_idx >= 31 * 120:
                    break

                c_row.append(count)
                c_col.append(feature_idx)
                c_data.append(1.0)
                tree_idx += 1
                #x[feature_idx] = 1.0

            if query not in query_embedding_dict:
                continue

            query_embedding = query_embedding_dict[query]
            if len(query_embedding) != len(user_embedding):
                continue

            X_row.extend(c_row)
            X_col.extend(c_col)
            X_data.extend(c_data)
            #X.append(x)
            U.append(user_embedding)
            Q.append(query_embedding)
            y.append(label)

            count += 1
            if count % 100000 == 0:
                print count

    print len(X_row), len(X_col)
    X = csr_matrix((X_data, (X_row, X_col)), shape=(count, 31 * 120))
    #X = np.array(X)
    U = np.array(U)
    Q = np.array(Q)
    y = np.array(y)
    return X, U, Q, y


if __name__ == '__main__':
    #query_embedding_dict = load_query_embedding(sys.argv[1])
    #process(sys.argv[2], query_embedding_dict)

    process_id(sys.argv[1])
