'''
Sanity check IoU
'''
import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
from ioulib import plot_box, iou, Box
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})


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
    line = ax2.plot([], [], c='k')[0]
    ax2.set(ylabel='IoU', xlabel='Timestep')

    ax1.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax1.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax1) # IMPORTANT - this is also required
    
    box1 = Box([0,0,0],[1,1,1],[0,0,0])
    box2 = Box([-2,0,0],[3,2,0.5],[0,0,0])

    n_frames = 240

    ious = []
    x = []
    def update(i):
        print(f'\r\x1b[KFrame: {i+1} / {n_frames}', end="")
        if i < n_frames / 2:
            ax1.cla()
            plot_box(box1, ax1, c='b')
            plot_box(box2, ax1, c='r')
            box2.loc[0] += 0.05
            box2.rot += [0.025,0.04,0.1]
            box1.loc[0] += 0.01
        else:
            ax1.cla()
            plot_box(box1, ax1, c='b')
            plot_box(box2, ax1, c='r')
            box2.loc[0] -= 0.05
            box2.rot -= [0.025,0.04,0.1]
            box1.loc[0] -= 0.01

        box1.get_mesh(True)
        box2.get_mesh(True)

        ious.append(iou(box1,box2))
        x.append(i)
        line.set_data(x, ious)
        ax2.set(xlim=(-1,x[-1]), ylim=(min(ious)-0.01, max(ious)+0.01))
        ax1.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-2,2), ylim=(-2,2), zlim=(-2,2))
        fig.canvas.draw()

    anim = FuncAnimation(fig, update, interval=30, frames=n_frames, repeat=False)
    anim.save('iouanim.mp4')
    print()

if __name__=='__main__':
    animate()
