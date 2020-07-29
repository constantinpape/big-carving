import napari
from elf.io import open_file
from .util import accumulate_node_labels, merge_seg_from_node_labels

MERGED_SEG_KEY = 'merged_segmentation'


def preprocess(project_path,
               raw_path, raw_key,
               seg_path, seg_key,
               node_label_path, node_label_key,
               n_threads):

    def serialize_attr(attrs, name, val):
        if name in attrs:
            val = attrs[name]
        else:
            assert val is not None
            attrs[name] = val
        return val

    with open_file(project_path, 'a') as f:
        attrs = f.attrs
        raw_path = serialize_attr(attrs, 'raw_path', raw_path)
        raw_key = serialize_attr(attrs, 'raw_key', raw_key)
        seg_path = serialize_attr(attrs, 'seg_path', seg_path)
        seg_key = serialize_attr(attrs, 'seg_key', seg_key)
        node_label_path = serialize_attr(attrs, 'node_label_path', node_label_path)
        node_label_key = serialize_attr(attrs, 'node_label_key', node_label_key)
        n_threads = serialize_attr(attrs, 'n_threads', n_threads)
        next_id = attrs.get('next_id', None)

    config = {'n_threads': n_threads}

    # load the primary data
    # TODO
    # add option for lazy loading
    # support pyramidal data sources
    with open_file(raw_path, 'r') as f:
        ds = f[raw_key]
        ds.n_threads = n_threads
        raw = ds[:]
        config['raw'] = raw

    with open_file(seg_path, 'r') as f:
        ds = f[seg_key]
        ds.n_threads = n_threads
        seg = ds[:]
        config['seg'] = seg

    f_project = open_file(project_path, 'r')

    have_merged_seg = MERGED_SEG_KEY in f
    if have_merged_seg:
        ds = f_project[MERGED_SEG_KEY]
        ds.n_threads = n_threads
        merged_seg = ds[:]
    else:
        merged_seg = merge_seg_from_node_labels(seg, node_labels, n_threads)
    config['merged_seg'] = merged_seg

    if have_merged_seg:
        node_labels = accumulate_node_labels(seg, merged_seg, return_dict_style=True)
    else:
        with open_file(node_label_path, 'r') as f:
            node_labels = f[node_label_key][:]

    assert node_labels.ndim == 2
    assert node_labels.shape[1] == 2
    node_labels = dict(zip(node_labels[:, 0], node_labels[:, 1]))
    config['node_labels'] = node_labels

    if next_id is None:
        next_id = int(merged_seg.max()) + 1
    config['next_id'] = next_id

    return config


# NOTE the cursor positionition is only recorded correctly for the selected layer
# so we cycle through all layers and recorded for the selected one
# this will not work so straightforward if we had layers with scale or multi-scale
def _get_cursor_position(viewer, layer_names=['raw', 'seg', 'merged_seg']):
    position = None
    layers = viewer.layers
    # TODO iterate over layers programatically, so that we don't need the layer_names
    # argument
    for name in layer_names:
        layer = layers[name]
        if layer.selected:
            position = layer.coordinates
    if position is not None:
        position = tuple(int(pos) for pos in position)
    return position


# TODO better support for painting, but need to sync more things ...
# all parameters except for project path are stored in the project container
def segmentation_correction(project_path,
                            raw_path=None, raw_key=None,
                            seg_path=None, seg_key=None,
                            node_label_path=None, node_label_key=None,
                            n_threads=8):
    """
    """
    # run preprocessing
    config = preprocess(project_path,
                        raw_path, raw_key,
                        seg_path, seg_key,
                        node_label_path, node_label_key,
                        n_threads=n_threads)

    node_labels = config['node_labels']
    next_id = config['next_id']
    n_threads = config['n_threads']

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(config['raw'], name='raw')
        viewer.add_labels(config['seg'], name='seg')
        viewer.add_labels(config['merged_seg'], name='merged_seg')

        def _update_merged_seg(viewer, node_labels):
            seg = viewer.layers['seg'].data
            merged_seg = merge_seg_from_node_labels(seg, node_labels, n_threads)
            viewer.layers['merged_seg'].data = merged_seg

        # TODO this should be Shift + Right Click
        @viewer.bind_key('Shift-D')
        def split(viewer):
            nonlocal next_id
            nonlocal node_labels

            position = _get_cursor_position(viewer)
            if position is None:
                print("No layer was selected, aborting split")
                return

            # get the segmentation value under the cursor
            label_val = viewer.layers['seg'].data[position]

            if label_val == 0:
                print("Cannot split background label, aborting split")
                return

            prev_id = node_labels[label_val]
            print("Splitting label", label_val, "from id", prev_id, "and assigning id", next_id)

            node_labels[label_val] = next_id
            next_id += 1
            _update_merged_seg(viewer, node_labels)
            print("split done")

        # TODO this should be Shift + Left Click AND should support selecting multiple
        # segments -> check out the new label properties functionality !
        @viewer.bind_key('Shift-A')
        def merge(viewer):
            nonlocal node_labels

            position = _get_cursor_position(viewer)
            if position is None:
                print("No layer was selected, aborting detach")
                return

            # get the segmentation value under the cursor
            label_val = viewer.layers['seg'].data[position]
            if label_val == 0:
                print("Cannot merge background label, aborting merge")
                return

            # get the selected id in the merged seg layer
            src_id = viewer.layers['merged_seg'].selected_label
            if src_id == 0:
                print("Cannot merge into background value")
                return

            target_id = node_labels[label_val]
            print("Merging id", target_id, "into id", src_id)

            node_labels = {k: src_id if v == target_id else v
                           for k, v in node_labels.items()}
            _update_merged_seg(viewer, node_labels)
            print("Merge done")

        @viewer.bind_key('s')
        def save(viewer):
            merged_seg = viewer.layers['merged_seg'].data
            with open_file(project_path, 'a') as f:
                f.attrs['next_id'] = next_id
                ds = f.require_dataset(MERGED_SEG_KEY, shape=merged_seg.shape, dtype=merged_seg.dtype,
                                       compression='gzip', chunks=(64, 64, 64))
                ds.n_threads = n_threads
                ds[:] = merged_seg
            print("saving done")

# TODO
#
# correction with additional graph based watershed functionality
#


def preprocess_with_graph(project_path,
                          raw_path, raw_key,
                          seg_path, seg_key,
                          node_label_path, node_label_key):
    pass


def segmentation_correction_graph_based(project_path,
                                        raw_path=None, raw_key=None,
                                        seg_path=None, seg_key=None,
                                        node_label_path=None, node_label_key=None,
                                        boundary_path=None, boundary_key=None):
    """
    """
    # run preprocessing
    graph, weights = preprocess_with_graph()

    with napari.gui_qt():
        viewer = napari.Viewer()
