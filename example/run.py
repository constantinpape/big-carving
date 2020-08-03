import nifty.distributed as ndist
from carving.big_correction import segmentation_correction
from elf.io import open_file

path = './data/data.n5'
raw_root = 'raw'
ws_root = 'watersheds'
node_label_key = 'node_labels/initial'
save_key = 'node_labels/corrected'

scale = 0
n_scales = 3

with_graph = True

if with_graph:
    graph = ndist.Graph(path, 's0/graph', 4)
    with open_file('./data/data.n5', 'r') as f:
        weights = f['features'][:, 0]
else:
    graph, weights = None, None


segmentation_correction(path, raw_root, scale,
                        path, ws_root, scale,
                        path, node_label_key,
                        path, save_key, n_scales,
                        seg_scale=2, seg_scale_factor=(1, 4, 4),
                        graph=graph, weights=weights)
