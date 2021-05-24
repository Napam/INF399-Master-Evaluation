from itertools import combinations
from typing_extensions import final
import numpy as np
from numpy.core.fromnumeric import sort 
import pandas as pd 
from ioulib import Box, iou, plot_box, get_mesh_from_box
from matplotlib import pyplot as plt 
from typing import Iterable, Optional, Sequence, Tuple, List, Union
from copy import copy
import pymesh
from matplotlib.ticker import NullFormatter
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})
from debug import debug, debugs, debugt
from tqdm import tqdm
from multiprocessing import Pool


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


def plot_boxes(df_outputs, df_labels, imgnr, thresh: Optional[float] = None):
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
    results: np.ndarray, not the whole result matrix, but just for a single class, should use 
                         custom dtype. Will be mutated

    df_outputs and df_labels should only contain boxes respective a single class and a single imgnr.
    '''
    outputboxes = get_boxes_from_df(df_outputs)
    labelboxes = get_boxes_from_df(df_labels)
    
    for labelbox in labelboxes:
        for i, outputbox in enumerate(outputboxes):
            iou_ = iou(outputbox, labelbox)

            # True positive
            if iou_ > threshold:
                outputboxes.pop(i)
                results['TP'] += 1
                break
        else:
            # False negative if true box is not predicted
            results['FN'] += 1
    
    # Any remaining outputboxes implies false positives
    results['FP'] += len(outputboxes)


def _calc_ap_old(df_outputs: pd.DataFrame, df_labels: pd.DataFrame, threshold: float, results: np.ndarray):
    '''
    results: np.ndarray, not the whole result matrix, but just for a single class, should use 
                         custom dtype. Will be mutated

    df_outputs and df_labels should only contain boxes respective a single class and a single imgnr.
    '''
    outputboxes = get_boxes_from_df(df_outputs)
    labelboxes = get_boxes_from_df(df_labels)
    
    for labelbox in labelboxes:
        for i, outputbox in enumerate(outputboxes):
            iou_ = iou(outputbox, labelbox)

            # True positive
            if iou_ > threshold:
                outputboxes.pop(i)
                results['TP'] += 1
                break
        else:
            # False negative if true box is not predicted
            results['FN'] += 1
    
    # Any remaining outputboxes implies false positives
    results['FP'] += len(outputboxes)


#Returns true positive, false positive and false negative for the batch of bounding boxes for a single image.
def get_single_image_results(gt_boxes, pred_boxes, iou_thr, calc_iou):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

def calc_ap_from_dfs(
    df_outputs: pd.DataFrame,
    df_labels: pd.DataFrame,
    threshold: float,
    n_classes: int
):
    '''
    Assumes that unique(df_outputs.imgnr) == unique(df_labels.imgnr)

    n_classes for FishDETR should include a class for "nothing" as the last class (7)
    '''
    results = np.zeros((n_classes), dtype=np.dtype([('TP', int), ('FP', int), ('FN', int)]))
    tqdmkwargs = {'ascii':True, 'ncols':70, 'desc':f"AP{int(threshold*100)}"}

    imgLabelIter = tqdm(
        df_labels.groupby('imgnr', sort=False),
        ascii = True,
        ncols = 70,
        desc = f"AP{int(threshold*100)}",
        disable=True
    )

    context = {'FP':0}

    for (imgnr, dfLabelImgnr) in imgLabelIter:
        dfOutImgnr = df_outputs.query(f"imgnr=={imgnr}")
        lenOuts = len(dfLabelImgnr)
        lenLabels = len(dfOutImgnr)

        # No predictions, no labels -> Do nothing
        if (lenOuts + lenLabels) == 0:
            continue

        # Predictions, but no labels -> False positives
        if (lenOuts > 0) and (lenLabels == 0):
            continue
        
        # No predictions, but there are labels -> False Negatives
        if (lenOuts == 0) and (lenLabels > 0):
            continue
    
        outputboxes = get_boxes_from_df(df_outputs)
        labelboxes = get_boxes_from_df(df_labels)

        print(get_single_image_results(labelboxes, outputboxes, threshold, iou))


def calc_ap_from_dfs_old(
    df_outputs: pd.DataFrame,
    df_labels: pd.DataFrame,
    threshold: float,
    n_classes: int
):
    '''
    Assumes that unique(df_outputs.imgnr) == unique(df_labels.imgnr)
    '''
    results = np.zeros((n_classes), dtype=np.dtype([('TP', int), ('FP', int), ('FN', int)]))

    imgLabelIter = tqdm(
        df_labels.groupby('imgnr', sort=False),
        ascii = True,
        ncols = 70,
        desc = f"AP{int(threshold*100)}",
        disable=True
    )

    for (imgnr, dfLabelImgnr) in imgLabelIter:
        for class_ in range(n_classes):
            dfOuts = df_outputs.query(f"imgnr=={imgnr} & class_=={class_}")
            dfLabels = dfLabelImgnr.query(f"class_=={class_}")
            
            lenOuts = len(dfOuts)
            lenLabels = len(dfLabels)

            # No predictions, no labels -> Do nothing
            if (lenOuts + lenLabels) == 0:
                continue

            # Predictions, but no labels -> False positives
            if (lenOuts > 0) and (lenLabels == 0):
                results[class_]['FP'] += lenOuts
                continue
            
            # No predictions, but there are labels -> False Negatives
            if (lenOuts == 0) and (lenLabels > 0):
                results[class_]['FN'] += lenLabels
                continue

            _calc_ap(dfOuts, dfLabels, threshold, results[class_])            
    
    TP, FP, FN = results['TP'].sum(), results['FP'].sum(), results['FN'].sum()
    print({'true_positive':TP, 'false_positive':FP, 'false_negative':FN})
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall, results


if __name__ == '__main__':
    df_outputs = pd.read_csv('nogit_train_output.csv')
    df_labels = pd.read_csv('nogit_train_labels.csv')
    # plot_boxes(df_outputs, df_labels, 42)    
    calc_ap_from_dfs(df_outputs.query("imgnr==42"), df_labels.query("imgnr==42"), 0.95, 6)
    calc_ap_from_dfs_old(df_outputs.query("imgnr==42"), df_labels.query("imgnr==42"), 0.95, 6)


    exit()
    precisions = []
    recalls = []
    results = []
    threshs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for thresh in tqdm(threshs, disable=True):
        precision, recall, result = calc_ap_from_dfs(df_outputs, df_labels, thresh, 6)
        precisions.append(precision)
        recalls.append(recall)
        results.append(result)
    
    pd.DataFrame({'thresh':threshs, 'precision':precisions, 'recall':recalls}).to_csv("prcurves.csv", index=False)