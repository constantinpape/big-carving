import napari
import numpy as np
from elf.io import open_file


def _load_multiscale_ds(path, root, start_scale, n_scales):
    f = open_file(path, 'r')
    g = f[root]
    datasets = [g[f's{scale}'] for scale in range(start_scale, start_scale + n_scales)]
    return datasets


# TODO hidden segments
# TODO return random colors for all node ids
# TODO implement functionality to switch seed and
# to update if we get new node_labels without switching the whole cmap
def _get_random_colors(node_labels):
    unique_labels = np.unique(node_labels)
    color_map = ''  # TODO unique_labels to color map
    color_map = {ii: color_map[label] for ii, label in enumerate(node_labels)}
    # TODO set zero to background color
    color_map[0] = ''
    return color_map


# For now: have two layers for watershed and merged segmentation
# and set the colormap manually.
# Eventually it would be nice to handle this via properties and
# switch the display between showing the watershed or label prop.
def segmentation_correction(raw_path, raw_root, raw_scale,
                            ws_path, ws_root, ws_scale,
                            node_label_path, node_label_key,
                            save_path, save_key, n_scales):

    ds_raw = _load_multiscale_ds(raw_path, raw_root,
                                 raw_scale, n_scales)

    ds_ws = _load_multiscale_ds(ws_path, ws_root,
                                ws_scale, n_scales)
    assert ds_ws[0].shape == ds_raw[0].shape

    with open_file(node_label_path, 'r') as f:
        node_labels = f[node_label_key][:]
    assert node_labels.ndim == 2
    node_labels = node_labels[:, 1]

    node_colors = _get_random_colors(node_labels)

    last_action = None

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(ds_raw, name='raw')
        viewer.add_labels(ds_ws, name='fragments')
        viewer.add_labels(ds_ws, name='segments', color=node_colors)

        # split of fragment from segment
        @viewer.bind_key('Shift-D')
        def split(viewer):
            pass

        # merge two segments
        @viewer.bind_key('Shift-A')
        def merge(viewer):
            pass

        # toggle hidden mode for the selected segment
        @viewer.bind_key()
        def toggle_hidden(viewer):
            pass

        # toggle visibility for hidden segments
        @viewer.bind_key()
        def toggle_view_hidden(viewer):
            pass

        # undo the last split / merge action
        @viewer.bind_key()
        def undo(viewer):
            pass

        # print help
        @viewer.bind_key()
        def help(viewer):
            pass

        # save the current node labeling to disc
        @viewer.bind_key()
        def save_labels(viewer):
            pass


def segmentation_carving():
    pass
