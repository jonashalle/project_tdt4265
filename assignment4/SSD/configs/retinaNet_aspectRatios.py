from tops.config import LazyCall as L
from ssd.modeling import retinaNet, AnchorBoxes
#from .retinaNet_subnets import train, optimizer, schedulers, backbone, model, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform
from .retinaNet_subnets import train, anchors, optimizer, schedulers, backbone, model, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

# aspect_ratios_1 with alpha_k = 10 for car and person 
# Better to detect Person, bit worse for cars. Better for detecting small objects. 
# The same for mAP as with original anchors
anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [124, 400]], #Original
    #min_sizes=[[4,4], [8, 8], [16, 16], [32, 32], [48, 48], [64, 64], [86, 86]],#, [128, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    #aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]], #Original
    aspect_ratios=[[2, 4, 6], [2, 4, 6], [2, 3], [2,3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

# #aspect_ratios_2 with alpha_k = 10 for car ans person 
# # min boxes smaller 
# # Result: not better to find smaller objects compared to the one above. worse AP for cars.
# anchors = L(AnchorBoxes)(
#     feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
#     # Strides is the number of pixels (in image space) between each spatial position in the feature map
#     strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
#     #min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [124, 400]], #Original
#     min_sizes=[[8, 8], [16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128]],
#     # aspect ratio is defined per feature map (first index is largest feature map (38x38))
#     # aspect ratio is used to define two boxes per element in the list.
#     # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
#     # Number of boxes per location is in total 2 + 2 per aspect ratio
#     #aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]], #Original
#     aspect_ratios=[[2, 4, 6], [2, 4, 6], [2, 3], [2,3], [2], [2]],
#     image_shape="${train.imshape}",
#     scale_center_variance=0.1,
#     scale_size_variance=0.2
# )

# aspect_ratios_3 with alpha_k = 10 for car ans person 
# Result:
# anchors = L(AnchorBoxes)(
#     feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
#     # Strides is the number of pixels (in image space) between each spatial position in the feature map
#     strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
#     min_sizes=[[16, 16], [26, 26], [40, 40], [64, 64], [86, 86], [128, 128], [124, 400]], #Original
#     # aspect ratio is defined per feature map (first index is largest feature map (38x38))
#     # aspect ratio is used to define two boxes per element in the list.
#     # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
#     # Number of boxes per location is in total 2 + 2 per aspect ratio
#     #aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]], #Original
#     aspect_ratios=[[2, 4, 6], [2, 4], [2, 3], [2, 3], [2], [2]], #small cars are square, persons may be 
#     image_shape="${train.imshape}",
#     scale_center_variance=0.1,
#     scale_size_variance=0.2
# )

