'''
Sanity check IoU
'''

from matplotlib import animation
from numpy.lib import poly
import numpy as np 
from itertools import combinations
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
from ioulib import plot_box, iou, Box
import pymesh

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
        
def animate():
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1,projection='3d')
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_ylabel('IoU')

    ax1.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax1.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax1) # IMPORTANT - this is also required
    
    box1 = Box([0,0,0],[1,1,1],[0,0,0])
    box2 = Box([-2,0,0],[3,2,0.5],[0,0,0])

    n_frames = 120

    ious = []
    def update(i):
        print(f'\r\x1b[KFrame: {i+1} / {n_frames}', end="")
        plot_box(box1, ax1, c='b')
        plot_box(box2, ax1, c='r')
        box2.loc[0] += 0.05
        box2.rot += [0.025,0.04,0.1]
        box1.loc[0] += 0.01
        ious.append(iou(box1,box2))
        ax2.plot(ious, c='k')
        ax1.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-2,2), ylim=(-2,2), zlim=(-2,2))
    anim = FuncAnimation(fig, update, interval=30, frames=n_frames, repeat=False)
    anim.save('test.mp4')
    print()

if __name__=='__main__':
    animate()