import cv2
import numpy as np
import time
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class YoloDetector:
    """
    Object detector using YOLOv4-tiny with CUDA and cuDNN acceleration when available
    """

    def __init__(
        self,
        config_path,
        weights_path,
        labels_path,
        conf_threshold=0.01,  # Extremely low threshold for testing
        nms_threshold=0.4,
        draw_bboxes=True,
        input_size=416,
        detect_every_n_frames=1,
        use_roi=False,
        roi_top=0.0,
        roi_bottom=1.0,
    ):
        for file_path in [config_path, weights_path, labels_path]:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.draw_bboxes = draw_bboxes
        self.input_size = input_size
        self.detect_every_n_frames = detect_every_n_frames
        self.use_roi = use_roi
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.frame_count = 0

        try:
            cv2.setUseOptimized(True)
            logger.info("OpenCV optimizations enabled")
        except:
            logger.warning("Could not enable OpenCV optimizations")

        # Load class labels
        try:
            with open(labels_path, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.class_names)} classes from {labels_path}")
        except Exception as e:
            logger.error(f"Error loading labels from {labels_path}: {e}")
            self.class_names = []

        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.class_names) or 80, 3), dtype=np.uint8
        )

        # Load the network
        logger.info(f"Loading YOLO model from {config_path} and {weights_path}...")
        try:
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            # Check for CUDA availability
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

            if cuda_available:
                logger.info("CUDA is available! Setting up GPU acceleration")
                # Use CUDA backend
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                # Use regular CUDA precision instead of FP16
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using CUDA with default precision")
            else:
                logger.warning("CUDA is not available, using CPU")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            ln = self.net.getLayerNames()
            output_layers_indices = self.net.getUnconnectedOutLayers()

            if len(output_layers_indices.shape) > 1:
                self.output_layers = [ln[i[0] - 1] for i in output_layers_indices]
            else:
                self.output_layers = [ln[i - 1] for i in output_layers_indices]

            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

        logger.info(
            f"Confidence threshold: {self.conf_threshold}, NMS threshold: {self.nms_threshold}"
        )
        logger.info(
            f"Input size: {self.input_size}, Frame skip: {self.detect_every_n_frames}"
        )
        if self.use_roi:
            logger.info(
                f"Using ROI: {self.roi_top * 100}% to {self.roi_bottom * 100}% of frame height"
            )

        self.detection_results = []

    def run(self, img_arr, throttle=None):
        if img_arr is None:
            logger.error("Input image is None")
            return img_arr, []

        image = img_arr.copy()
        height, width = image.shape[:2]

        font_size = 0.4
        thickness = 1
        x_pos = 5
        y_start = 15
        line_spacing = 15

        cv2.putText(
            image,
            "Det Active",
            (x_pos, y_start + line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            thickness,
        )

        self.frame_count += 1

        if self.frame_count % self.detect_every_n_frames != 0:
            if self.draw_bboxes and self.detection_results:
                for detection in self.detection_results:
                    try:
                        class_name = detection["class"]
                        class_id = self.class_names.index(class_name)
                        color = [int(c) for c in self.colors[class_id]]
                        x, y, w, h = detection["box"]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = f"{class_name}: {detection['confidence']:.2f}"
                        cv2.putText(
                            image,
                            text,
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )
                    except Exception as e:
                        logger.error(f"Error redrawing detection: {e}")

            # Add frame skipping indicator
            cv2.putText(
                image,
                f"Skip: {self.frame_count % self.detect_every_n_frames}/{self.detect_every_n_frames}",
                (x_pos, y_start + 2 * line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 0, 0),
                thickness,
            )

            return image, self.detection_results

        # Extract region of interest if enabled
        if self.use_roi:
            roi_y_start = int(height * self.roi_top)
            roi_y_end = int(height * self.roi_bottom)
            roi_image = image[roi_y_start:roi_y_end, 0:width]
        else:
            roi_image = image
            roi_y_start = 0

        # Create a blob and pass it through the network
        start_time = time.time()

        # Ensure image is not empty
        if roi_image.size == 0:
            logger.error("ROI image is empty!")
            return image, []

        # Create blob from image - this is a performance-critical step
        blob = cv2.dnn.blobFromImage(
            roi_image,
            1 / 255.0,
            (self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)

        # Run inference
        try:
            outputs = self.net.forward(self.output_layers)
            inference_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return image, []

        cv2.putText(
            image,
            f"Inf: {inference_time:.2f}s",
            (x_pos, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 255, 0),
            thickness,
        )

        boxes = []
        confidences = []
        class_ids = []

        roi_height, roi_width = roi_image.shape[:2]
        detection_count = 0

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    detection_count += 1
                    # Scale bounding box coordinates back to original image
                    box = detection[0:4] * np.array(
                        [roi_width, roi_height, roi_width, roi_height]
                    )
                    (center_x, center_y, box_width, box_height) = box.astype("int")

                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))

                    if self.use_roi:
                        y += roi_y_start

                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        cv2.putText(
            image,
            f"Pot: {len(boxes)}",
            (x_pos, y_start + 2 * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            thickness,
        )

        indices = []
        if boxes and confidences:
            try:
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences, self.conf_threshold, self.nms_threshold
                )
            except Exception as e:
                logger.error(f"Error during NMS: {e}")

        # Prepare detection results
        self.detection_results = []

        detection_count = len(indices) if len(indices) > 0 else 0
        cv2.putText(
            image,
            f"Det: {detection_count}",
            (x_pos, y_start + 3 * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            thickness,
        )

        # Add GPU/CPU indicator
        backend_info = "GPU" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "CPU"
        cv2.putText(
            image,
            f"Using: {backend_info}",
            (x_pos, y_start + 4 * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 255),
            thickness,
        )

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]

                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"unknown_{class_id}"

                self.detection_results.append(
                    {"class": class_name, "confidence": confidence, "box": (x, y, w, h)}
                )

                if self.draw_bboxes:
                    color = [int(c) for c in self.colors[class_id % len(self.colors)]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(
                        image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                    )

        if self.detection_results:
            classes_found = set(d["class"] for d in self.detection_results)
            logger.info(
                f"Detected {len(self.detection_results)} objects: {', '.join(classes_found)}"
            )

        return image, self.detection_results

    def shutdown(self):
        pass
