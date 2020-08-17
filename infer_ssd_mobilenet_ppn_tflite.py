import json

import numpy as np
import tensorflow as tf
from PIL import Image

with open('/tmp/cache/tflite/index', 'r') as f:
  index_map = json.loads(f.read())
class_names = index_map['labels_list']
class_num = len(class_names)

conf_threshold = 0.6
iou_threshold = 0.45
_scale_factors = [10.0, 10.0, 5.0, 5.0]


def _sigmoid(x):
  s = 1 / (1 + np.exp(-x))
  return s


def get_center_coordinates_and_sizes(bbox):
  ymin, xmin, ymax, xmax = np.transpose(bbox)
  width = xmax - xmin
  height = ymax - ymin
  ycenter = ymin + height / 2.
  xcenter = xmin + width / 2.
  return [ycenter, xcenter, height, width]


def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = list(tensor.shape)
  dynamic_tensor_shape = tensor.shape
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w):
  combined_shape = combined_static_and_dynamic_shape(box_encodings)
  batch_size = combined_shape[0]
  tiled_anchor_boxes = np.tile(
    np.expand_dims(anchors, 0), [batch_size, 1, 1])
  tiled_anchors_boxlist = np.reshape(tiled_anchor_boxes, [-1, 4])
  decoded_boxes = decode_bbox_frcnn(
    np.reshape(box_encodings, [-1, 4]), tiled_anchors_boxlist, ori_h, ori_w)
  decoded_boxes = np.reshape(decoded_boxes, np.stack(
    [combined_shape[0], combined_shape[1], 4]))
  return decoded_boxes


def overlap(x1, x2, x3, x4):
  left = max(x1, x3)
  right = min(x2, x4)
  return right - left


def cal_iou(box, truth):
  w = overlap(box[0], box[2], truth[0], truth[2])
  h = overlap(box[1], box[3], truth[1], truth[3])
  if w <= 0 or h <= 0:
    return 0
  inter_area = w * h
  union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (
      truth[3] - truth[1]) - inter_area
  return inter_area * 1.0 / union_area


def apply_nms(all_boxes, thres):
  res = []

  for cls in range(class_num):
    cls_bboxes = all_boxes[cls]
    sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

    p = dict()
    for i in range(len(sorted_boxes)):
      if i in p:
        continue

      truth = sorted_boxes[i]
      for j in range(i + 1, len(sorted_boxes)):
        if j in p:
          continue
        box = sorted_boxes[j]
        iou = cal_iou(box, truth)
        if iou >= thres:
          p[j] = 1

    for i in range(len(sorted_boxes)):
      if i not in p:
        res.append(sorted_boxes[i])
  return res


def decode_bbox_frcnn(conv_output, anchors, ori_h, ori_w):
  ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
  ty, tx, th, tw = conv_output.transpose()
  ty /= _scale_factors[0]
  tx /= _scale_factors[1]
  th /= _scale_factors[2]
  tw /= _scale_factors[3]

  w = np.exp(tw) * wa
  h = np.exp(th) * ha

  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a

  scale_h = 1
  scale_w = 1
  ymin = np.maximum((ycenter - h / 2.) * scale_h * ori_h, 0)
  xmin = np.maximum((xcenter - w / 2.) * scale_w * ori_w, 0)
  ymax = np.minimum((ycenter + h / 2.) * scale_h * ori_h, ori_h)
  xmax = np.minimum((xcenter + w / 2.) * scale_w * ori_w, ori_w)

  bbox_list = np.transpose(np.stack([ymin, xmin, ymax, xmax]))

  return bbox_list


def postprocess(anchors, box_encodings, class_predictions, ori_h, ori_w):
  detection_boxes = batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w)
  detection_scores_with_background = _sigmoid(class_predictions)
  detection_scores = detection_scores_with_background[:, :, 1:]
  detection_score = np.expand_dims(np.max(detection_scores[:, :, 0:], axis=-1), 2)
  detection_cls = np.expand_dims(np.argmax(detection_scores[:, :, 0:], axis=-1), 2)

  preds = np.concatenate([detection_boxes, detection_score, detection_cls], axis=2)
  res_list = []
  for pred in preds:
    pred = pred[pred[:, 4] >= conf_threshold]
    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
      box = [int(pred[ix, iy]) for iy in range(4)]
      box.append(int(pred[ix, 5]))
      box.append(pred[ix, 4])
      all_boxes[box[4] - 1].append(box)

    res = apply_nms(all_boxes, iou_threshold)
    res_list.append(res)
  return res_list


def preprocess(image):
  image = image.resize((300, 300))
  image = np.array(image)
  image = (2.0 / 255.0) * image - 1.0
  image = np.float32(image)
  image = image[np.newaxis, :, :, :]
  return image


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/tmp/cache/tflite/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# preprocess
origin_img = Image.open('/tmp/cache/tflite/1.jpg')
origin_w, origin_h = origin_img.size
img = preprocess(origin_img)

interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

anchors = interpreter.get_tensor(output_details[0]['index'])
box_encodings = interpreter.get_tensor(output_details[1]['index'])
class_predictions_with_background = interpreter.get_tensor(output_details[2]['index'])

result = postprocess(anchors, box_encodings, class_predictions_with_background, origin_h, origin_w)
response = {'detection_classes': [],
            'detection_boxes': [],
            'detection_scores': []}

from PIL import ImageDraw

img_draw = ImageDraw.Draw(origin_img)

for i in range(len(result[0])):
  ymin = result[0][i][0]
  xmin = result[0][i][1]
  ymax = result[0][i][2]
  xmax = result[0][i][3]
  score = result[0][i][5]
  label = result[0][i][4]

  if score < conf_threshold:
    continue
  response['detection_classes'].append(
    class_names[int(label)])
  response['detection_boxes'].append([
    str(ymin), str(xmin),
    str(ymax), str(xmax)
  ])
  response['detection_scores'].append(str(score))

  img_draw.rectangle([(xmin, ymin), (xmax, ymax)])
  img_draw.text((xmax - 70, ymin + 5), "%s=%.3f" % (class_names[int(label)], score))

print(response)
origin_img.show()
