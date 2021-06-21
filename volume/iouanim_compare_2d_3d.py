'''
Sanity check IoU
'''
import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
from numpy.typing import ArrayLike
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

    # ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_box_intersection(boxes1: ArrayLike, boxes2: ArrayLike, ax: plt.Axes):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
        
    ax.imshow(np.zeros((50,50)), cmap='gray_r')
    for x1, y1, x2, y2 in boxes1:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, facecolor='none', edgecolor='blue'))
    
    for x1, y1, x2, y2 in boxes2:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, facecolor='none', edgecolor='red'))
        
    lt = np.max([boxes1[:,:2], boxes2[:,:2]], 0)
    rb = np.min([boxes1[:,2:], boxes2[:,2:]], 0)
    
    wh = (rb - lt).clip(0)
    
    boxes = np.concatenate((lt, rb), axis=1)
    for (x, y), (w, h) in zip(boxes[:,:2], wh):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor='red', alpha=0.2))

def intersection2d(boxes1: ArrayLike, boxes2: ArrayLike):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
        
    lt = np.max([boxes1[:,:2], boxes2[:,:2]], 0)
    rb = np.min([boxes1[:,2:], boxes2[:,2:]], 0)
    
    wh = (rb - lt).clip(0)
    return wh.prod()

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA) * max(0, yB - yA)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

        
def animate():
    fig = plt.figure(figsize=(20,10))
    fig.suptitle("3D vs 2D IoU")
    ax_tl = fig.add_subplot(2,2,1,projection='3d')
    ax_bl = fig.add_subplot(2,2,3)
    ax_tr = fig.add_subplot(2,2,2)
    ax_br = fig.add_subplot(2,2,4)
    line3D = ax_bl.plot([], [], c='k')[0]
    line2D = ax_br.plot([], [], c='k')[0]

    ax_bl.set(ylabel='IoU 3D', xlabel='Timestep')
    ax_br.set(ylabel='IoU 2D', xlabel='Timestep')

    ax_tl.set_box_aspect([2,1,1]) # IMPORTANT - this is the new, key line
    ax_tl.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax_tl) # IMPORTANT - this is also required
    
    box1 = Box([0,0.5,0],[2,2,2],[0,0,0])
    box2 = Box([-3,-0.5,0],[2,2,2],[0,0,0])

    n_frames = 148

    ious3D = []
    ious2D = []
    x = []
    def update(i):
        print(f'\r\x1b[KFrame: {i+1} / {n_frames}', end="")
        
        ax_tl.cla()
        plot_box(box1, ax_tl, c='b')
        plot_box(box2, ax_tl, c='r')
        box2.loc[0] += 0.04

        ax_tr.cla()
        xyxy1 = [*(box1.loc[:2] - box1.dim[:2] / 2), *(box1.loc[:2] + box1.dim[:2] / 2)]
        xyxy2 = [*(box2.loc[:2] - box2.dim[:2] / 2), *(box2.loc[:2] + box2.dim[:2] / 2)]

        plot_box_intersection(
            [xyxy1], 
            [xyxy2],
            ax_tr
        )

        box1.get_mesh(True)
        box2.get_mesh(True)

        ious3D.append(iou(box1,box2))        
        
        intsct2d = intersection2d([xyxy1], [xyxy2])
        a1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
        a2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
        ious2D.append(intsct2d / (a1 + a2 - intsct2d))
        # ious2D.append(bb_intersection_over_union(xyxy1, xyxy2))

        x.append(i)
        line3D.set_data(x, ious3D)
        line2D.set_data(x, ious2D)

        ax_bl.set(xlim=(-1,x[-1]), ylim=(min(ious3D)-0.01, max(ious3D)+0.01))
        ax_br.set(xlim=(-1,x[-1]), ylim=(min(ious2D)-0.01, max(ious2D)+0.01))
        ax_tl.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-4,4), ylim=(-2,2), zlim=(-2,2), xticks=[], yticks=[], zticks=[])
        ax_tr.set(xlabel='x', ylabel='y', xlim=(-4,4), ylim=(-2,2), xticks=[], yticks=[])
        fig.canvas.draw()
    
    anim = FuncAnimation(fig, update, interval=30, frames=n_frames, repeat=False)
    anim.save('iouanim_compare.mp4')
    print()

    plt.close(fig)

    plt.plot(x, ious2D, label="2D")
    plt.plot(x, ious3D, label="3D")
    plt.savefig("compare2d3d.pdf")

if __name__=='__main__':
    animate()
