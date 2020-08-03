import numpy as np
from nifty.ground_truth import overlap


# TODO scalable implementation
def merge_seg_from_node_labels(seg, node_labels, n_threads=None):
    uniques, inverse = np.unique(seg, return_inverse=True)
    return np.array([node_labels[elem]
                     for elem in uniques])[inverse].reshape(seg.shape)


def accumulate_node_labels(seg, labels, return_dict_style=False):
    unique_ids = np.unique(seg)
    ovlp = overlap(seg, labels)

    node_labels = np.array([ovlp.overlapArrays(seg_id, True)[0][0]
                            for seg_id in unique_ids])

    if return_dict_style:
        return np.concatenate([unique_ids[:, None],
                               node_labels[:, None]], axis=1).astype('int64')
    else:
        n_ids = unique_ids[-1] + 1
        dense_node_labels = np.full(n_ids, -1, dtype='int64')
        dense_node_labels[unique_ids] = node_labels
        return dense_node_labels
