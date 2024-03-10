"""Implements an operator that detects traffic lights."""
import logging

import erdos

import numpy as np

import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight, \
    TrafficLightColor
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.messages import TrafficLightsMessage

import tensorflow as tf

from PIL import Image
import cv2
from pylot.perception.detection.yolo.common.data_utils import preprocess_image
from pylot.perception.detection.yolo.yolo3.postprocess_np import yolo3_postprocess_np

logger = logging.getLogger(__name__)

class TrafficLightDetOperator(erdos.Operator):
    """Detects traffic lights using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        traffic_lights_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.ReadStream,
                 traffic_lights_stream: erdos.WriteStream, flags):
        # Register a callback on the camera input stream.
        camera_stream.add_callback(self.on_frame, [traffic_lights_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._traffic_lights_stream = traffic_lights_stream
        # Load the model from the model file.
        pylot.utils.set_tf_loglevel(logging.ERROR)

        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[self._flags.traffic_light_det_gpu_index]], 'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[self._flags.traffic_light_det_gpu_index], True)

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(
            self._flags.traffic_light_det_model_path)

        logger.debug('\n\n*********Path of the model using now: %s', self._flags.traffic_light_det_model_path)
        self._model_names = self._flags.obstacle_detection_model_names

        if 'yolo' in self._model_names:
            # These labels map to the class output from the YOLO detector
            self._labels = {
                0: TrafficLightColor.GREEN,
                1: TrafficLightColor.YELLOW,
                2: TrafficLightColor.RED,
            }
        else:
            self._labels = {
                1: TrafficLightColor.GREEN,
                2: TrafficLightColor.YELLOW,
                3: TrafficLightColor.RED,
                4: TrafficLightColor.OFF
            }

        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0
        # Serve some junk image to load up the model.
        if 'yolo' in self._model_names:
            self.__run_yolo_model(np.zeros((108, 192, 3), dtype='uint8'))
        else:
            self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))

    @staticmethod
    def connect(camera_stream: erdos.ReadStream,
                time_to_decision_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages for traffic lights.
        """
        traffic_lights_stream = erdos.WriteStream()
        return [traffic_lights_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
        self._traffic_lights_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    def on_frame(self, msg: erdos.Message,
                 traffic_lights_stream: erdos.WriteStream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg: A :py:class:`~pylot.perception.messages.FrameMessage`.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.TrafficLightsMessage`
                messages for traffic lights.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        if 'yolo' in self._model_names:
            boxes, scores, labels = self.__run_yolo_model(
                msg.frame.as_rgb_numpy_array())
        else:
            boxes, scores, labels = self.__run_model(
                msg.frame.as_rgb_numpy_array())

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, msg.frame.camera_setup.height,
            msg.frame.camera_setup.width)

        self._logger.debug('@{}: {} detected traffic lights {}'.format(
            msg.timestamp, self.config.name, traffic_lights))

        traffic_lights_stream.send(
            TrafficLightsMessage(msg.timestamp, traffic_lights))
        traffic_lights_stream.send(erdos.WatermarkMessage(msg.timestamp))

        if self._flags.log_traffic_light_detector_output:
            msg.frame.annotate_with_bounding_boxes(msg.timestamp,
                                                   traffic_lights)
            msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                           'tl-detector-{}'.format(self.config.name))

    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_labels = [
            self._labels[int(label)] for label in classes[0][:num_detections]
        ]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return res_boxes, res_scores, res_labels

    def __run_yolo_model(self, image_np):
        model_input_shape = (416, 416)

        # Convert from BGR format to RGB which YOLO expects
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(image)
        image_data = preprocess_image(img, model_input_shape)

        # Original image shape, in (height, width) format
        image_shape = img.size[::-1]

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_data, name="serving_default_image_input:0"))

        # Extract prediction arrays from each of the output tensors into a list
        prediction = [np.array(result["predict_conv_1"]), np.array(result["predict_conv_2"]), np.array(result["predict_conv_3"])]
        with open(self._flags.path_yolo_anchors) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

        boxes, classes, scores = yolo3_postprocess_np(prediction, image_shape, anchors, 3, model_input_shape, elim_grid_sense=False)

        # Normalize bounding box to range [0, 1]
        norm_boxes = []
        (h, w) = image_shape
        for (xmin, ymin, xmax, ymax) in boxes:
            # Order of bounding box coordinates are switched to be consistent
            # with what Pylot expects
            norm_boxes.append([ymin / h, xmin / w, ymax / h, xmax / w])

        num_detections = len(classes)
        res_labels = [
            self._labels[int(label)] for label in classes[:]
        ]

        # Pylot expects bounding boxes and scores to be returned as a Tensor
        return tf.constant(norm_boxes, dtype=tf.float32), tf.constant(scores, dtype=tf.float32), res_labels

    def __convert_to_detected_tl(self, boxes, scores, labels, height, width):
        traffic_lights = []
        for index in range(len(scores)):
            if scores[
                    index] > self._flags.traffic_light_det_min_score_threshold:
                bbox = BoundingBox2D(
                    int(boxes[index][1] * width),  # x_min
                    int(boxes[index][3] * width),  # x_max
                    int(boxes[index][0] * height),  # y_min
                    int(boxes[index][2] * height)  # y_max
                )
                traffic_lights.append(
                    TrafficLight(scores[index],
                                 labels[index],
                                 id=self._unique_id,
                                 bounding_box=bbox))
                self._unique_id += 1
        return traffic_lights
