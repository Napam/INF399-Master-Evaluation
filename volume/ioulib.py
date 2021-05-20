'''
For n-dim rotated IoU
'''
import pymesh
import numpy as np 
import re
from dataclasses import dataclass
from itertools import combinations
from matplotlib import pyplot as plt


@dataclass
class Box:
    loc: np.ndarray
    dim: np.ndarray
    rot: np.ndarray

    def __post_init__(self):
        self.loc = np.array(self.loc, dtype=float)
        self.dim = np.array(self.dim, dtype=float)
        self.rot = np.array(self.rot, dtype=float)


def mesh_trans(mesh, x: np.ndarray):
    return pymesh.form_mesh(x, mesh.faces)


def get_rotation_matrix(rotation: np.ndarray):
    sz, sy, sx = np.sin(rotation)
    cz, cy, cx = np.cos(rotation)
    return np.array([ # Multiply with rotation matrix
        [cx*cy, cx*sy*sz-sx*cz, cx*sy*cz+sx*sz],
        [sx*cy, sx*sy*sz+cx*cz, sx*sy*cz-cx*sz],
        [  -sy,          cy*sz,          cy*cz],
    ])


def mesh_rot(mesh, r: np.ndarray):
    R = get_rotation_matrix(r)
    mean = mesh.vertices.mean(0)
    return pymesh.form_mesh((mesh.vertices-mean)@R.T+mean, mesh.faces)


def iou(box1: Box, box2: Box):
    box1_ = pymesh.generate_box_mesh(box1.loc-box1.dim/2, box1.loc+box1.dim/2)
    box2_ = pymesh.generate_box_mesh(box2.loc-box2.dim/2, box2.loc+box2.dim/2)
    box1_ = mesh_rot(box1_, box1.rot)
    box2_ = mesh_rot(box2_, box2.rot)
    intersection = pymesh.boolean(box1_, box2_, 'intersection').volume
    union = pymesh.boolean(box1_, box2_, 'union').volume
    return intersection / union


def plot_box(box: Box, ax, **kwargs):
    box_ = pymesh.generate_box_mesh(box.loc-box.dim/2, box.loc+box.dim/2)
    box_ = mesh_rot(box_, box.rot)
    
    combs = np.array(list(combinations(box_.vertices, 2)))
    for i in [0,2,3,7,10,13,16,21,22,24,25,27]:
        ax.plot(*combs[i].T, **kwargs)


if __name__ == '__main__':
    box1 = Box([0,0,0],[4,4,4],[0,0,0])
    box2 = Box([0,0,0],[1,1,1],[0,0,0])
    # print(iou(box1, box2))

    # # plot_box(box1)
    box1.loc[0] += 2
    print(box1)