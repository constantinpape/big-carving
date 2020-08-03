import os

import napari
import nifty
import nifty.tools as nt
import numpy as np
import vigra

from elf.io import open_file
from elf.color import glasbey

from .util import merge_seg_from_node_labels


# TODO add heuristics to load small enough scales into ram
def _load_multiscale_ds(path, root, start_scale, n_scales):
    f = open_file(path, 'r')
    g = f[root]
    datasets = [g[f's{scale}'] for scale in range(start_scale, start_scale + n_scales)]
    # datasets = g['s1']
    # datasets.n_thredas = 4
    # datasets = datasets[:]
    return datasets


# TODO hidden segments
# TODO return random colors for all node ids
# TODO implement functionality to switch seed and
# to update if we get new node_labels without switching the whole cmap
def _get_random_colors(node_labels, use_glasbey):
    unique_labels = np.unique(node_labels)
    print(len(node_labels))
    print(len(unique_labels))

    n_labels = len(unique_labels)
    if use_glasbey:
        print("Generating colormap ...")
        color_map = glasbey(n_labels) / 255.
        print("... done")
    else:
        color_map = np.random.rand(n_labels, 3)
    assert color_map.shape == (n_labels, 3)
    color_map = np.concatenate([color_map,
                                np.ones((n_labels, 1))], axis=1)
    color_map[0] = [0, 0, 0, 0]
    # color_map[2:] = [0, 0, 0, 0]
    color_map = {label_id: cmap for label_id, cmap in zip(unique_labels, color_map)}

    color_map = {ii: color_map[label] for ii, label in enumerate(node_labels)}
    assert len(color_map) == len(node_labels)

    # seg_id = 1
    return color_map


def _get_cursor_position(viewer, layer_name):
    position = None
    scale = None
    layer_scale = None

    for layer in viewer.layers:
        if layer.selected:
            position = layer.coordinates
            scale = layer.scale
        if layer.name == layer_name:
            layer_scale = layer.scale

    assert position is not None
    scale = (1, 1, 1) if scale is None else scale
    layer_scale = (1, 1, 1) if layer_scale is None else layer_scale

    rel_scale = [sc / lsc for lsc, sc in zip(layer_scale, scale)]
    position = tuple(int(pos * sc) for pos, sc in zip(position, rel_scale))
    return position


def _load_node_labes(initial_path, initial_key, save_path, save_key):
    if os.path.exists(save_path) and save_key in open_file(save_path, 'r'):
        with open_file(save_path, 'r') as f:
            node_labels = f[save_key][:]
    else:
        with open_file(initial_path, 'r') as f:
            node_labels = f[initial_key][:]

    if node_labels.ndim == 2:
        node_labels = node_labels[:, 1]
    assert node_labels.ndim == 1
    return node_labels.astype('uint32')


# For now: have two separate layers for watershed and merged segmentation.
# Eventually it would be nice to handle this via properties and
# switch the display between showing the watershed or label prop.
def segmentation_correction(raw_path, raw_root, raw_scale,
                            ws_path, ws_root, ws_scale,
                            node_label_path, node_label_key,
                            save_path, save_key, n_scales,
                            seg_scale, seg_scale_factor,
                            graph=None, weights=None):

    ds_raw = _load_multiscale_ds(raw_path, raw_root,
                                 raw_scale, n_scales)

    ds_ws = _load_multiscale_ds(ws_path, ws_root,
                                ws_scale, n_scales)
    # assert ds_ws[0].shape == ds_raw[0].shape

    node_labels = _load_node_labes(node_label_path, node_label_key,
                                   save_path, save_key)
    next_id = int(node_labels.max()) + 1
    mask_id = None

    node_label_history = []
    ws_base = ds_ws[seg_scale][:]

    with napari.gui_qt():

        def _seg_from_labels(node_labels):
            seg = merge_seg_from_node_labels(ws_base, node_labels)
            return seg

        viewer = napari.Viewer()
        viewer.add_image(ds_raw, name='raw')
        viewer.add_labels(ds_ws, name='fragments', visible=False)
        seg = _seg_from_labels(node_labels)
        viewer.add_labels(seg, name='segments', scale=seg_scale_factor)
        viewer.add_labels(np.zeros_like(ws_base), scale=seg_scale_factor, name='mask')

        if graph is not None:
            assert weights is not None
            uv_ids = graph.uvIds()
            viewer.add_labels(np.zeros_like(ws_base), scale=seg_scale_factor, name='seeds')
            ws_id = None
            sub_graph, sub_weights = None, None
            sub_nodes, sub_edges = None, None

        # split of fragment from segment
        @viewer.bind_key('Shift-D')
        def split(viewer):
            nonlocal next_id
            nonlocal node_labels
            nonlocal node_label_history

            position = _get_cursor_position(viewer, 'fragments')
            if position is None:
                print("No layer was selected, aborting split")
                return

            # get the segmentation value under the cursor
            frag_id = viewer.layers['fragments'].data[0][position]

            if frag_id == 0:
                print("Cannot split background label, aborting split")
                return

            seg_id = node_labels[frag_id]
            print("Splitting fragment", frag_id, "from segment", seg_id, "and assigning segment id", next_id)
            node_label_history.append(node_labels.copy())

            node_labels[frag_id] = next_id
            next_id += 1
            seg = _seg_from_labels(node_labels)
            viewer.layers['segments'].data = seg

            print("split done")

        # merge two segments
        @viewer.bind_key('Shift-A')
        def merge(viewer):
            nonlocal node_labels
            nonlocal node_label_history

            position = _get_cursor_position(viewer, 'segments')
            if position is None:
                print("No layer was selected, aborting detach")
                return

            # get the segmentation value under the cursor
            seg_id1 = viewer.layers['segments'].data[position]
            if seg_id1 == 0:
                print("Cannot merge background label, aborting merge")
                return

            # get the selected id in the merged seg layer
            seg_id2 = viewer.layers['segments'].selected_label
            if seg_id2 == 0:
                print("Cannot merge into background value")
                return

            node_label_history.append(node_labels.copy())
            print("Merging id", seg_id1, "into id", seg_id2)
            node_labels[node_labels == seg_id1] = seg_id2
            seg = _seg_from_labels(node_labels)
            viewer.layers['segments'].data = seg

            print("Merge done")

        # # toggle hidden mode for the selected segment
        # @viewer.bind_key()
        # def toggle_hidden(viewer):
        #     pass

        # # toggle visibility for hidden segments
        # @viewer.bind_key()
        # def toggle_view_hidden(viewer):
        #     pass

        @viewer.bind_key('m')
        def update_mask(viewer):
            nonlocal mask_id
            seg_id = viewer.layers['segments'].selected_label
            if seg_id == mask_id:
                return
            mask_id = seg_id
            mask = (viewer.layers['segments'].data == seg_id).astype(viewer['mask'].data.dtype)
            viewer.layers['mask'].data = mask
            if 'seeds' in viewer.layers:
                viewer.layers['seeds'].data = np.zeros_like(ws_base)

        @viewer.bind_key('w')
        def watershed(viewer):
            nonlocal ws_id, next_id
            nonlocal node_labels
            nonlocal node_label_history
            nonlocal sub_nodes, sub_edges
            nonlocal sub_graph, sub_weights

            if mask_id is None:
                print("Need to select segment to run watershed")
                return

            if ws_id != mask_id or sub_graph is None:
                print("Computing sub-graph ...")
                sub_nodes = np.where(node_labels == mask_id)[0]
                sub_edges, _ = graph.extractSubgraphFromNodes(sub_nodes, allowInvalidNodes=True)
                sub_weights = weights[sub_edges]

                nodes_relabeled, max_id, mapping = vigra.analysis.relabelConsecutive(sub_nodes,
                                                                                     start_label=0,
                                                                                     keep_zeros=False)
                sub_uvs = uv_ids[sub_edges]
                sub_uvs = nt.takeDict(mapping, sub_uvs)

                n_nodes = max_id + 1
                sub_graph = nifty.graph.undirectedGraph(n_nodes)
                sub_graph.insertEdges(uv_ids)

                mask = viewer.layers['mask'].data
                seeds = viewer.layers['seeds'].data
                seeds[np.logical_not(mask)] = 0

            seed_nodes = np.zeros(sub_graph.numberOfNodes, dtype='uint64')
            for seed_id in np.unique(seeds, next_id):
                seeded = np.unique(ws_base[mask == seed_id])
                if seeded[0] == 0:
                    seeded = seeded[1:]
                seeded = nt.takeDict(mapping, seeded)
                seed_nodes[seeded] = seed_id

            sub_labels = nifty.graph.edgeWeightedWatershedsSegmentation(sub_graph, seed_nodes, sub_weights)
            node_label_history.append(node_labels.copy())

            node_labels[sub_nodes] = sub_labels
            seg = _seg_from_labels(node_labels)
            viewer.layers['segments'].data = seg
            next_id = int(node_labels.max()) + 1

        # # undo the last split / merge action
        @viewer.bind_key('u')
        def undo(viewer):
            nonlocal node_labels
            nonlocal node_label_history
            print("Undo last action")
            node_labels = node_label_history.pop()
            seg = _seg_from_labels(node_labels)
            viewer.layers['segments'].data = seg

        # save the current node labeling to disc
        @viewer.bind_key('s')
        def save_labels(viewer):
            print("saving node labels")
            with open_file(save_path, 'a') as f:
                ds = f.require_dataset(save_key, shape=node_labels.shape,
                                       chunks=node_labels.shape, compression='gzip',
                                       dtype=node_labels.dtype)
                ds[:] = node_labels

        # # print help
        # @viewer.bind_key()
        # def help(viewer):
        #     pass


def segmentation_carving():
    pass
