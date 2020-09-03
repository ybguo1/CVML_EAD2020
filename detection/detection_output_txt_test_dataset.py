import natsort as natsort
import numpy as np
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
import os

local_path = "/mmdetection/detection_demo"
config_file = '/mmdetection/detection_demo/faster_rcnn_r101_fpn_1x_full_data/faster_rcnn_r101_fpn_1x_full_data.py'
checkpoint_file = '/mmdetection/detection_demo/faster_rcnn_r101_fpn_1x_full_data/epoch_30.pth'

# Model init
model = init_detector(config_file, checkpoint_file)

# Set test image path
# path = "/txt_to_csv/new_val_img"
path = "/EAD2020-Phase-II-Evaluation/EAD2020-Phase-II-Evaluation/Generalization"
img_list = natsort.natsorted(os.listdir(path))

# Set test txt output path
config = config_file.split('/')[-1].split('.')[0]
checkpoint = checkpoint_file.split('/')[-1].split('.')[0]

con_path = config + '_' + checkpoint + '_txt'
txt_path_ = os.path.join(local_path, con_path)

# if not os.path.exists(txt_path_):
#     os.makedirs(txt_path_)


# txt_output_path = txt_path_
txt_output_path = "/mmdetection/detection_demo/test_result/Generalization"
if not os.path.exists(txt_output_path):
    os.mkdir(txt_output_path)


for img in tqdm(img_list):
    img_path = os.path.join(path, img)
    result = inference_detector(model, img_path)
    txt_path = os.path.join(txt_output_path, img[:-4] + ".txt")

    txt_file = open(txt_path, 'w')
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    for idx, bbox_ in enumerate(bboxes):
        label, score, x1, y1, x2, y2 = model.CLASSES[labels[idx]], bbox_[4], bbox_[0], bbox_[1], bbox_[2], bbox_[3]
        if score >= 0.3:
            info = ("%s\t" + "%g\t" * 5 + "\n") % (label, score, x1, y1, x2, y2)
            txt_file.write(info)

    txt_file.close()

