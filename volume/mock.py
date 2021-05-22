from ioulib import Box, iou, get_mesh_from_box

box1 = Box([1,1], [1,1], [0,0])
box2 = Box([0,0], [1,1], [0,0])
print(box1.get_mesh().vertices)
print(box2.get_mesh().vertices)
