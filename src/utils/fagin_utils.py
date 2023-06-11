import numpy as np
import random
import math

from collections import Counter

def split_samples_by_class(distributed_data):
    class_data = {}
    id1 = 0
    for data_ptr, target in distributed_data:
        id1 += 1
        if not target in class_data:
            class_data[target] = []
        class_data[target].append((data_ptr, id1))
    return class_data


def get_kth_dist(data_ptr, class_data, aggregation_distances, k):
    k_largest=[float('inf')]*k
    for data_ptr2 in class_data:
        if not (data_ptr, data_ptr2[1]) in aggregation_distances:
            continue
        if aggregation_distances[(data_ptr, data_ptr2[1])] <= max(k_largest):
            k_largest[k_largest.index(max(k_largest))] = aggregation_distances[(data_ptr, data_ptr2[1])]
    return max(k_largest)

def get_sorted_distances(data_ptr, class_data, aggregation_distances):
    distances = []
    for data_ptr2 in class_data:
        if not (data_ptr, data_ptr2[1]) in aggregation_distances:
            continue
        print('append')
        distances.append(aggregation_distances[(data_ptr, data_ptr2[1])])
    return sorted(distances)

def digamma(x):
    if x == 0:
        return float('-inf')
    return math.log(x, math.e) - 0.5 / x
