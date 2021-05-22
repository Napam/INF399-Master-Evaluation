from typing_extensions import final
import numpy as np
from numpy.core.fromnumeric import sort 
import pandas as pd 
from ioulib import Box, iou, plot_box, get_mesh_from_box
from matplotlib import pyplot as plt 
from typing import Iterable, Sequence, Tuple, List, Union
from copy import copy
import pymesh
from matplotlib.ticker import NullFormatter
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})
from debug import debug, debugs, debugt


SPAWNBOX_DIM = np.array((3.5999999046325684, 3.5999999046325684, 3.5999999046325684))
SPAWNBOX_LOC = np.array((0.0, 3.0, 0.0))
BOX_COLS = ["x", "y", "z", "w", "l", "h", "rx", "ry", "rz"]


def get_box(pos_size_rot: Tuple[float, float, float, float, float, float, float, float, float]):
    x, y, z, w, l, h, rx, ry, rz = pos_size_rot
    loc = np.array((x, y, z)) * SPAWNBOX_DIM / 2 + SPAWNBOX_LOC
    dim = np.array((w, l, h)) 
    rot = np.array((rx, ry, rz)) * 2 * np.pi
    return Box(loc=loc, dim=dim, rot=rot)


def get_boxes_from_df(df: pd.DataFrame) -> List[Box]:
    return [get_box(box) for box in df[BOX_COLS].values]


def get_meshes_from_df(df: pd.DataFrame) -> List[pymesh.Mesh]:
    return [get_box(box).get_mesh() for box in df[BOX_COLS].values]


def get_ious_for_imgnr(df_outputs: pd.DataFrame, df_labels: pd.DataFrame, imgnr: int):
    outputs = get_boxes_from_df(df_outputs.query("imgnr==@imgnr"))
    labels = get_boxes_from_df(df_labels.query("imgnr==@imgnr"))


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_boxes(df_outputs, df_labels, imgnr):
    outputs = get_meshes_from_df(df_outputs.query("imgnr==@imgnr"))
    labels = get_meshes_from_df(df_labels.query("imgnr==@imgnr"))
    
    fig = plt.figure(figsize=(8,3.5))
    ax1 = fig.add_subplot(1,3,1,projection='3d')
    ax2 = fig.add_subplot(1,3,2,projection='3d')
    ax3 = fig.add_subplot(1,3,3,projection='3d')

    for ax, azim in zip((ax1, ax2, ax3), (0,40,80)):
        for box in outputs:
            plot_box(box, ax, c='red', linewidth=0.5)
        
        for box in labels:
            plot_box(box, ax, c='k', linewidth=0.5)

        ax.set(xlabel='x', ylabel='y', zlabel='z')
        ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
        ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
        set_axes_equal(ax)
        ax.azim += azim
        ax.set_title(f"azim={ax.azim}")
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle(f'Bounding boxes of imgnr {imgnr}')
    plt.savefig('boxviz.png')


def nms(
    outputs: List[pymesh.Mesh],
    confs: Sequence[float],
    threshold: float=0.5
):
    """
    outputs: list of predictions as pymesh.Mesh polygons
    conds: sequence of confidence scores, len(confs) == len(outputs) should be true
    threshold: iou treshold, [0,1]
    """
    outputs = np.array(outputs, dtype=object)
    confs = confs.copy()
    final_boxes = []
    
    ranks = np.argsort(confs)
    while len(ranks): 
        currbox = outputs[ranks[-1]]
        final_boxes.append(currbox)
        # Always remove the current box from selection
        ious = np.array([iou(currbox, box) for box in outputs[ranks]])
        mask = ious < threshold
        ranks = ranks[mask]
    return final_boxes


def nms_from_df(df_outputs: pd.DataFrame, imgnr: int, threshold: float=0.5):
    df = df_outputs.query(f"imgnr=={imgnr}")
    boxes = get_boxes_from_df(df)
    confs = df.conf.values
    return nms(boxes, confs, threshold)


def _calc_ap(df_outputs: pd.DataFrame, df_labels: pd.DataFrame, threshold: float, results: np.ndarray):
    '''
    df_outputs and df_labels should only contain boxes respective a single class and a single imgnr

    results: np.ndarray, not the whole result matrix, but just for a single class, should use 
                         custom dtype
    '''
    outputs = get_boxes_from_df(df_outputs)
    labels = get_boxes_from_df(df_labels)

    for labelbox in labels:
        for outputbox in outputs:
            debug(iou(outputbox, labelbox))


def calc_ap_from_dfs(
    df_outputs: pd.DataFrame,
    df_labels: pd.DataFrame,
    threshold: float,
    n_classes: int
):
    '''
    Assumes that unique(df_outputs.imgnr) == unique(df_labels.imgnr)
    '''
    results = np.zeros((n_classes), dtype=np.dtype([('TP', int), ('FP', int), ('FN', int)]))

    for (imgnr, dfLabelImgnr) in df_labels.groupby('imgnr'):
        for class_ in range(n_classes):
            dfOuts = df_outputs.query(f"imgnr=={imgnr} & class_=={class_}")
            dfLabels = dfLabelImgnr.query(f"class_=={class_}")
            
            lenOuts = len(dfOuts)
            lenLabels = len(dfLabels)

            # No predictions, no targets -> Do nothing
            if (lenOuts + lenLabels) == 0:
                continue

            # Predictions, but no targets -> False positives
            if (lenOuts > 0) and (lenLabels == 0):
                results[class_]['FP'] += lenOuts
                continue
            
            # No predictions, but there are targets -> False Negatives
            if (lenOuts == 0) and (lenLabels > 0):
                results[class_]['FN'] += lenLabels
                continue

            debug(imgnr)
            debug(class_)
            _calc_ap(dfOuts, dfLabels, threshold, results[class_])
            print()
        # break

if __name__ == '__main__':
    df_outputs = pd.read_csv('nogit_train_output.csv')
    df_labels = pd.read_csv('nogit_train_labels.csv')
    calc_ap_from_dfs(df_outputs, df_labels, 0.5, 6)
    # plot_boxes(df_outputs, df_labels, 10)
