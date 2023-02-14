import cv2
import uuid
import os
import time

# IMAGE_PATH = "Tensorflow/workspace/images/collectedimages"
# labels = ['hello', 'thanks', 'iloveyou', 'yes', 'no']
# number_imgs = 10

# ============Create models images==============
# for label in labels:
#     os.makedirs(f"Tensorflow/workspace/images/collectedimages/{label}")
#     cap = cv2.VideoCapture(0)
#     print(f"Collecting image for {label}")
#     time.sleep(5)
#
#     for imgnum in range(10):
#         ret, frame = cap.read()
#         imgname = os.path.join(IMAGE_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
#         cv2.imwrite(imgname, frame)
#         cv2.imshow('Frame', frame)
#         time.sleep(2)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
# cv2.destroyAllWindows()

# ============Create Label Map==============

# WORKSPACE_PATH = 'Tensorflow/workspace'
# SCRIPTS_PATH = 'Tensorflow/scripts'
# APIMODEL_PATH = 'Tensorflow/models'
# ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
# IMAGE_PATH = WORKSPACE_PATH + '/images'
# MODEL_PATH = WORKSPACE_PATH + '/models'
# PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
# CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
# CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
#
# labels = [{'name': 'Hello', 'id': 1}, {'name': 'ILoveYou', 'id': 2}, {'name': 'No', 'id': 3},
#           {'name': 'Yes', 'id': 4}, {'name': 'Thanks', 'id': 5}]
#
# with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')

# ============Create TF Records==============

# ----using in jupyter------
# python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}
# python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}

# ----using in terminal------

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
# CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

# python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}
# python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}

# python Tensorflow/scripts/generated_tfrecord.py -x Tensorflow/workspace/images/train -1 Tensorflow/workspace/annotations/train.record
# python Tensorflow/scripts/generated_tfrecord.py -x Tensorflow/workspace/images/test -1 Tensorflow/workspace/annotations/test.record

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

WORKSPACE_PATH = 'Tensorflow/workspace'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
MODEL_PATH = WORKSPACE_PATH + '/models'
CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 2
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)
