'''
For n-dim rotated IoU
'''
from typing import Optional, Union
import pymesh
import numpy as np 
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
        self._mesh: Optional[pymesh.Mesh] = None

    def get_mesh(self, recreate: bool = False) -> pymesh.Mesh:
        """
        Returns pymesh.Mesh equivalent of Box object

        recreate: bool, if True it will recreate (and cache) the pymesh.Mesh. Useful if one 
                  mutates box attributes. 
        """
        if (self._mesh is None) or recreate:
            self._mesh = get_mesh_from_box(self)
        return self._mesh


def mesh_trans(mesh: pymesh.Mesh, x: np.ndarray) -> pymesh.Mesh:
    """
    Translates given mesh to give x coordinate.

    mesh: pymesh.Mesh
    x: np.ndarray in R^3
    """
    return pymesh.form_mesh(x, mesh.faces)


def get_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """
    rotation: np.ndarray of shape (3,), rotations around x, y, z axis. The matrix encodes rotations
              in all axes. In radians.
    """
    sz, sy, sx = np.sin(rotation)
    cz, cy, cx = np.cos(rotation)
    return np.array([ # Multiply with rotation matrix
        [cx*cy, cx*sy*sz-sx*cz, cx*sy*cz+sx*sz],
        [sx*cy, sx*sy*sz+cx*cz, sx*sy*cz-cx*sz],
        [  -sy,          cy*sz,          cy*cz],
    ])


def mesh_rot(mesh: pymesh.Mesh, rot: np.ndarray) -> pymesh.Mesh:
    """
    Rotates given pymesh.Mesh be creating a new mesh object with rotated vertices. Rotational point
    around mean of mesh. 

    mesh: pymesh.Mesh
    rot: np.ndarray of shape (3,), rotations around x, y and z axis. Radians.
    """
    R = get_rotation_matrix(rot)
    mean = mesh.vertices.mean(0)
    return pymesh.form_mesh((mesh.vertices-mean)@R.T+mean, mesh.faces)


def get_mesh_from_box(box: Box) -> pymesh.Mesh:
    """
    Returns a pymesh.Mesh object corresponding to box attributes.

    box: Box
    """
    box_ = pymesh.generate_box_mesh(box.loc-box.dim/2, box.loc+box.dim/2)
    if any(box.rot):
        box_ = mesh_rot(box_, box.rot)
    return box_


def iou(box1: Box, box2: Box) -> float:
    """
    box1, box2: Box objects
    """
    box1_ = box1.get_mesh()
    box2_ = box2.get_mesh()
    intersection = pymesh.boolean(box1_, box2_, 'intersection').volume
    union = pymesh.boolean(box1_, box2_, 'union').volume
    return intersection / union


def plot_box(box: Union[Box, pymesh.Mesh], ax: plt.Axes, **kwargs):
    """
    box: Union[Box, pymesh.Mesh]
    ax: plt.Axes
    kwargs: kwargs for ax.plot (which plots each line making up the boxes), should be 3d ax.
    """
    if isinstance(box, Box):
        box = get_mesh_from_box(box)

    combs = np.array(list(combinations(box.vertices, 2)))
    for i in [0,2,3,7,10,13,16,21,22,24,25,27]: # PyMesh order, obtained by testing
        line = ax.plot(*combs[i].T, **kwargs)
    return line

if __name__ == '__main__':
    box1 = Box([0,0,0],[4,4,4],[0,0,0])
    box2 = Box([0,0,0],[1,1,1],[0,0,0])
    # print(iou(box1, box2))

    # # plot_box(box1)
    box1.loc[0] += 2
    print(box1)