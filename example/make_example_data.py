import json
import os
import luigi
import numpy as np
from elf.io import open_file
from mobie.import_data.util import downscale, compute_node_labels
from cluster_tools.relabel import RelabelWorkflow
from paintera_tools.util import compute_graph_and_weights


in_path = '/home/pape/Work/data/cremi/example/sampleA.n5'
raw_key = 'volumes/raw/s0'
ws_key = 'volumes/segmentation/watershed'
seg_key = 'volumes/segmentation/groundtruth'
bd_key = 'volumes/boundaries'

out_path = './data/data.n5'


def relabel(tmp_folder, target, max_jobs):
    task = RelabelWorkflow

    tmp_path = os.path.join(tmp_folder, 'data.n5')
    tmp_key = 'node_labels/relabel'
    chunks = [1, 256, 256]

    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    configs = task.get_config()

    conf = configs['global']
    conf.update({'block_shape': chunks})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['write']
    conf.update({'chunks': chunks})
    with open(os.path.join(config_dir, 'write.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=in_path, input_key=ws_key,
             assignment_path=tmp_path, assignment_key=tmp_key,
             output_path=out_path, output_key='watersheds/s0')
    assert luigi.build([t], local_scheduler=True)


def compute_graph():
    compute_graph_and_weights(in_path, bd_key,
                              out_path, 'watersheds/s0', out_path,
                              tmp_folder='tmp_graph', target='local',
                              max_jobs=4)


def make_example_data():
    target = 'local'
    max_jobs = 4

    resolution = [0.04, 0.004, 0.004]
    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2],
                     [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    chunks = [1, 256, 256]

    tmp_folder = './tmp_example'
    downscale(in_path, raw_key, out_path,
              resolution, scale_factors, chunks,
              tmp_folder, target, max_jobs, block_shape=None,
              library='skimage', metadata_format='paintera', out_key='raw')

    tmp_folder = './tmp_example_ws'
    relabel(tmp_folder, target, max_jobs)
    downscale(in_path, ws_key, out_path,
              resolution, scale_factors, chunks,
              tmp_folder, target, max_jobs, block_shape=None,
              library='vigra', library_kwargs={'order': 0},
              metadata_format='paintera', out_key='watersheds')

    tmp_folder = './tmp_example_seg'
    relabel(tmp_folder, target, max_jobs)
    downscale(in_path, seg_key, out_path,
              resolution, scale_factors, chunks,
              tmp_folder, target, max_jobs, block_shape=None,
              library='vigra', library_kwargs={'order': 0},
              metadata_format='paintera', out_key='segmentation')

    node_labels = compute_node_labels(out_path, 'watersheds/s0',
                                      out_path, 'segmentation/s0',
                                      tmp_folder, target, max_jobs)
    node_labels = np.concatenate([np.arange(len(node_labels))[:, None],
                                  node_labels[:, None]], axis=1)

    with open_file(out_path, 'a') as f:
        f.create_dataset('node_labels/initial', data=node_labels, compression='gzip',
                         chunks=(len(node_labels), 1))


def make_small_example_data():
    bb = np.s_[:25, :512, :512]
    with open_file('./data/data.n5') as f:
        raw = f['raw'][bb]
        ws = f['watersheds'][bb]

    with open_file('./data/small_data.n5') as f:
        pass


if __name__ == '__main__':
    make_example_data()
    # make_small_example_data()
