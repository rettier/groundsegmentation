# Common
data_directory = "/data/"
color_width = 1280
color_height = 720
ir_width = 1280
ir_height = 720

# Step 1
topic_infra1 = "/camera/infra1/image_rect_raw"
topic_infra2 = "/camera/infra2/image_rect_raw"
topic_color = "/camera/color/image_raw"
bagfile_folder = "/home/user/Documents/"

# Step 2
debug_step2 = False
change_threshold = 10
sharpest_frame_batch_size = 15

# Step 3
debug_step3 = False


def get_stereo():
    import cv2
    wsize = 9
    P1 = 8 * wsize * wsize
    P2 = 64 * wsize * wsize
    return cv2.StereoSGBM_create(0, 128, wsize, P1, P2)


# Step 4
debug_step4 = False


def make_segmenter(p):
    import pcl
    seg = p.make_segmenter_normals(ksearch=20)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(0.03)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(50)
    seg.set_distance_threshold(0.03)
    return seg


# Step 5
crf_enabled = True
debug_step5 = False
crf_inference_count = 9
crf_label_gt_prob = 0.7

def configure_crf(d, color_image):
    d.addPairwiseGaussian(sxy=6, compat=10)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=color_image, compat=10)


# Step 6
debug_step6 = False
min_obstacle_pixel_count = 5000
