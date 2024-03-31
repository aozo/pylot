"""Implements an operator that detects obstacles."""
import logging
import time

import erdos

import numpy as np

import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage

import tensorflow as tf

import json
import os
from PIL import Image
import cv2
from pylot.perception.detection.yolo.common.data_utils import preprocess_image
from pylot.perception.detection.yolo.yolo3.postprocess_np import yolo3_postprocess_np
from pylot.perception.detection.yolo.yolo5.postprocess_np import yolo5_postprocess_np
from pylot.perception.detection.ensemble import calculate_ensemble

logger = logging.getLogger(__name__)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BoundingBox2D):
            bbox = {
                'xmin': "{}".format(obj.x_min),
                'xmax': "{}".format(obj.x_max),
                'ymin': "{}".format(obj.y_min),
                'ymax': "{}".format(obj.y_max)
            }
            return bbox
            
        return obj.__dict__

class DetectionOperator(erdos.Operator):
    """Detects obstacles using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        model_path(:obj:`str`): Path to the model pb file.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.ReadStream,
                 obstacles_stream: erdos.WriteStream, model_path: str, flags):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [obstacles_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._obstacles_stream = obstacles_stream

        pylot.utils.set_tf_loglevel(logging.ERROR)

        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[self._flags.obstacle_detection_gpu_index]],
            'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[self._flags.obstacle_detection_gpu_index], True)

        # Multiple models may be loaded as a part of an ensemble, so it's stored in a list
        self._model = []

        # Load the model from the saved_model format file.
        self._model.append(tf.saved_model.load(model_path))

        print("[DEBUG] Detection model 1: %s" % (model_path))

        if self._flags.obstacle_detection_model_paths_2:
            self._model.append(tf.saved_model.load(self._flags.obstacle_detection_model_paths_2))
            print("[DEBUG] Detection model 2: %s" % (self._flags.obstacle_detection_model_paths_2))

        if self._flags.obstacle_detection_model_paths_3:
            self._model.append(tf.saved_model.load(self._flags.obstacle_detection_model_paths_3))
            print("[DEBUG] Detection model 3: %s" % (self._flags.obstacle_detection_model_paths_3))

        self._model_names = []
        self._model_names.append(self._flags.obstacle_detection_model_names)

        if self._flags.obstacle_detection_model_names_2:
            self._model_names.append(self._flags.obstacle_detection_model_names_2)

        if self._flags.obstacle_detection_model_names_3:
            self._model_names.append(self._flags.obstacle_detection_model_names_3)

        self.path_yolo_anchors = []
        self.path_yolo_anchors.append(self._flags.path_yolo_anchors)

        if self._flags.path_yolo_anchors_2:
            self.path_yolo_anchors.append(self._flags.path_yolo_anchors_2)

        if self._flags.path_yolo_anchors_3:
            self.path_yolo_anchors.append(self._flags.path_yolo_anchors_3)

        # Assume that class names are the same for all models if used as part of an ensemble
        if 'yolo' in self._model_names[0][0]:
            self._coco_labels = load_coco_labels(self._flags.path_coco_labels, is_zero_index=True)
        else:
            self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0

        for idx, _ in enumerate(self._model):
            # Serve some junk image to load up the model.
            if 'yolo' in self._model_names[idx][0]:
                self.__run_yolo_model(np.zeros((108, 192, 3), dtype='uint8'), idx)
            else:
                self.__run_model(np.zeros((108, 192, 3), dtype='uint8'), idx)

    @staticmethod
    def connect(camera_stream: erdos.ReadStream,
                time_to_decision_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        """
        obstacles_stream = erdos.WriteStream()
        return [obstacles_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
        # Sending top watermark because the operator is not flowing
        # watermarks.
        self._obstacles_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    def on_msg_camera_stream(self, msg: erdos.Message,
                             obstacles_stream: erdos.WriteStream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        start_time = time.time()
        # The models expect BGR images.
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        model_detections = {}
        for idx, _ in enumerate(self._model):
            model_detections[idx] = []
            if 'yolo' in self._model_names[idx][0]:
                num_detections, res_boxes, res_scores, res_classes = self.__run_yolo_model(
                    msg.frame.frame, idx)
            else:
                num_detections, res_boxes, res_scores, res_classes = self.__run_model(
                    msg.frame.frame, idx)

            for i in range(num_detections):
                # Convert tensors to native python types
                py_bboxes = res_boxes.numpy().tolist()
                py_scores = res_scores.numpy().tolist()
                model_detections[idx].append((py_bboxes[i], py_scores[i], res_classes[i]))
                print("[DEBUG] Model %d: Detected %s (%f)" % (idx, self._coco_labels[res_classes[i]], res_scores[i]))

        # If 3 models, then enable ensemble
        if len(model_detections) == 3:
            num_detections, py_bboxes, py_scores, res_classes = calculate_ensemble(
                    model_detections[0],
                    model_detections[1],
                    model_detections[2],
                    msg.frame.camera_setup.width,
                    msg.frame.camera_setup.height)
            # Convert back to tensors
            res_boxes = tf.constant(py_bboxes, dtype=tf.float32)
            res_scores = tf.constant(py_scores, dtype=tf.float32)

        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >=
                        self._flags.obstacle_detection_min_score_threshold):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] *
                                    msg.frame.camera_setup.width),
                                int(res_boxes[i][3] *
                                    msg.frame.camera_setup.width),
                                int(res_boxes[i][0] *
                                    msg.frame.camera_setup.height),
                                int(res_boxes[i][2] *
                                    msg.frame.camera_setup.height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id))
                        self._unique_id += 1
                    else:
                        self._logger.warning(
                            'Ignoring non essential detection {}'.format(
                                self._coco_labels[res_classes[i]]))
            else:
                self._logger.warning('Filtering unknown class: {}'.format(
                    res_classes[i]))

        self._logger.debug('@{}: {} obstacles: {}'.format(
            msg.timestamp, self.config.name, obstacles))

        # Extract only necessary fields
        json_obstacles = []
        for obj in obstacles:
            json_obstacles.append(
                {'id': obj.id,
                 'label': obj.label,
                 'bbox': obj.bounding_box,
                 'confidence': "{}".format(obj.confidence)
                }
            )

        # Prepare the complete json object
        obstacle_json = json.dumps([obj for obj in json_obstacles], cls=CustomEncoder)
        obstacle_json = obstacle_json.replace("\"", "")
        obstacle = {
            "timestamp": "{}".format(msg.timestamp),
            "obstacles": obstacle_json
        }
        
        # Append the json object to the json file
        filename = "{}/obstacles.json".format(self._flags.data_path)
        with open(filename, "a") as file:
            json.dump(obstacle, file)
            file.write(os.linesep)
            file.close()

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        # Send out obstacles.
        obstacles_stream.send(
            ObstaclesMessage(msg.timestamp, obstacles, runtime))
        obstacles_stream.send(erdos.WatermarkMessage(msg.timestamp))

        if self._flags.log_detector_output:
            msg.frame.annotate_with_bounding_boxes(msg.timestamp, obstacles,
                                                   None, self._bbox_colors)
            msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                           'detector-{}'.format(self.config.name))

    def __run_model(self, image_np, model_idx):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model[model_idx].signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        if self._flags.tf_model_zoo:
            boxes = result['detection_boxes']
            scores = result['detection_scores']
            classes = result['detection_classes']
            num_detections = result['num_detections']
        else:
            boxes = result['boxes']
            scores = result['scores']
            classes = result['classes']
            num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return num_detections, res_boxes, res_scores, res_classes

    def __run_yolo_model(self, image_np, model_idx):
        model_input_shape = (416, 416)

        # Convert from BGR format to RGB which YOLO expects
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(image)
        image_data = preprocess_image(img, model_input_shape)

        # Original image shape, in (height, width) format
        image_shape = img.size[::-1]

        infer = self._model[model_idx].signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_data, name="serving_default_image_input:0"))

        # Extract prediction arrays from each of the output tensors into a list
        prediction = [np.array(result["predict_conv_1"]), np.array(result["predict_conv_2"]), np.array(result["predict_conv_3"])]
        with open(self.path_yolo_anchors[model_idx]) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

        num_classes = len(self._coco_labels)

        if self._model_names[model_idx][0] == 'yolo3':
            boxes, classes, scores = yolo3_postprocess_np(prediction, image_shape, anchors, num_classes, model_input_shape, elim_grid_sense=False)
        else:
            boxes, classes, scores = yolo5_postprocess_np(prediction, image_shape, anchors, num_classes, model_input_shape, elim_grid_sense=True)

        # Normalize bounding box to range [0, 1]
        norm_boxes = []
        (h, w) = image_shape
        for (xmin, ymin, xmax, ymax) in boxes:
            # Order of bounding box coordinates are switched to be consistent
            # with what Pylot expects
            norm_boxes.append([ymin / h, xmin / w, ymax / h, xmax / w])

        # Pylot expects bounding boxes and scores to be returned as a Tensor
        return len(norm_boxes), tf.constant(norm_boxes, dtype=tf.float32), tf.constant(scores, dtype=tf.float32), classes
