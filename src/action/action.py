import math
import time
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from torchvision import transforms
from ultralytics import YOLO


class RobotState(Enum):
    WAITING = 0
    FORWARD = 1
    TURNING_LEFT = 2
    TURNING_RIGHT = 3
    STOP_RED = 4
    STOP_SIGN = 5  


class RobotController(Node):
    def __init__(self):
        super().__init__("robot_controller")

        self.declare_parameter("yolo_model_path", "/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic_light/signs/transfer/runs/detect/train4/weights/best.pt")
        self.declare_parameter("traffic_cls_model_path", "/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic_light/traffic_light_data/runs/traffic_light_efficientnet/final_with_classes.pt")
        
        self.declare_parameter("wait_seconds", 3.0)

        self.declare_parameter("normal_speed", 0.15)
        self.declare_parameter("slow_speed", 0.1)
        self.declare_parameter("slow_duration_sec", 5.0)
        self.declare_parameter("turn_angular_speed", 0.5)
        self.declare_parameter("turn_angle_deg", 90.0)

        self.declare_parameter("front_wall_threshold", 0.50)
        self.declare_parameter("wall_follow_trigger", 0.35)
        self.declare_parameter("show_debug_image", True)
        
        self.declare_parameter("sign_roi_x_min_ratio", 0.25)
        self.declare_parameter("sign_roi_x_max_ratio", 0.75)
        self.declare_parameter("min_sign_box_area", 1500)
        self.declare_parameter("sign_stable_frames", 3)
        self.declare_parameter("turn_sign_trigger_distance", 1.2)

        self.yolo_model_path         = self.get_parameter("yolo_model_path").value
        self.traffic_cls_model_path  = self.get_parameter("traffic_cls_model_path").value
        self.wait_seconds            = float(self.get_parameter("wait_seconds").value)
        self.slow_duration_sec       = float(self.get_parameter("slow_duration_sec").value)
        self.normal_speed            = float(self.get_parameter("normal_speed").value)
        self.slow_speed              = float(self.get_parameter("slow_speed").value)
        
        self.turn_angular_speed = float(self.get_parameter("turn_angular_speed").value)
        turn_angle_deg = float(self.get_parameter("turn_angle_deg").value)
        self.turn_duration_sec = math.radians(turn_angle_deg) / self.turn_angular_speed

        self.front_wall_threshold    = float(self.get_parameter("front_wall_threshold").value)
        self.wall_follow_trigger     = float(self.get_parameter("wall_follow_trigger").value)
        self.show_debug_image        = bool(self.get_parameter("show_debug_image").value)
        
        self.sign_roi_x_min_ratio    = float(self.get_parameter("sign_roi_x_min_ratio").value)
        self.sign_roi_x_max_ratio    = float(self.get_parameter("sign_roi_x_max_ratio").value)
        self.min_sign_box_area       = int(self.get_parameter("min_sign_box_area").value)
        self.sign_stable_frames      = int(self.get_parameter("sign_stable_frames").value)
        self.turn_sign_trigger_distance = float(self.get_parameter("turn_sign_trigger_distance").value)

        self.cmd_pub   = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub  = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.timer     = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.bridge = CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Device: {self.device}")

        self.yolo = YOLO(self.yolo_model_path)
        self.traffic_classes = ["green", "red", "yellow"]  
        self.traffic_model   = self._load_traffic_classifier(self.traffic_cls_model_path)
        self.traffic_tf      = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


        self.state            = RobotState.WAITING
        self.start_time       = time.time()
        self.state_start_time = time.time()

        self.current_speed = self.normal_speed
        self.slow_until_time: Optional[float] = None

        self.prev_sign_name: Optional[str] = None
        self.prev_sign_bbox: Optional[Tuple[int, int, int, int]] = None
        self.stable_sign_count = 0


        self.pending_turn: Optional[str] = None  # "left" | "right"

        self.last_detected_sign:        Optional[str] = None
        self.last_detected_light_color: Optional[str] = None

        self.scan_msg:     Optional[LaserScan] = None
        self.current_image: Optional[np.ndarray] = None

        self.last_sign_trigger_time = 0.0
        self.sign_cooldown_sec      = 3.0


        self.stop_sign_forward_duration = 5.0

        self.get_logger().info("ROS2 robot_controller started.")


    def _load_traffic_classifier(self, model_path: str):
        from torchvision.models import efficientnet_b0

        checkpoint = torch.load(model_path, map_location=self.device)

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 3)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

            if "classes" in checkpoint:
                self.traffic_classes = checkpoint["classes"]

        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

        else:
            self.get_logger().warn("Unknown checkpoint format")

        model.to(self.device)
        model.eval()

        return model

    def scan_callback(self, msg: LaserScan):
        self.scan_msg = msg

    def image_callback(self, msg: Image):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")

    def publish_cmd(self, linear_x: float = 0.0, angular_z: float = 0.0):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def get_range_min_360(self, start_deg: float, end_deg: float) -> float:
        if self.scan_msg is None:
            return float("inf")

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.scan_msg.angle_min
        angle_inc = self.scan_msg.angle_increment
        n = len(ranges)

        start_deg = start_deg % 360.0
        end_deg = end_deg % 360.0

        start_idx = int((math.radians(start_deg) - angle_min) / angle_inc)
        end_idx   = int((math.radians(end_deg)   - angle_min) / angle_inc)

        start_idx %= n
        end_idx %= n

        if start_idx <= end_idx:
            sector = ranges[start_idx:end_idx + 1]
        else:
            sector = np.concatenate([ranges[start_idx:], ranges[:end_idx + 1]])

        return float(np.min(sector))


    def get_range_percentile_360(self, start_deg: float, end_deg: float, q: float = 30.0) -> float:
        if self.scan_msg is None:
            return float("inf")

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.scan_msg.angle_min
        angle_inc = self.scan_msg.angle_increment
        n = len(ranges)

        start_deg = start_deg % 360.0
        end_deg = end_deg % 360.0

        start_idx = int((math.radians(start_deg) - angle_min) / angle_inc)
        end_idx   = int((math.radians(end_deg)   - angle_min) / angle_inc)

        start_idx %= n
        end_idx %= n

        if start_idx <= end_idx:
            sector = ranges[start_idx:end_idx + 1]
        else:
            sector = np.concatenate([ranges[start_idx:], ranges[:end_idx + 1]])

        sector = sector[np.isfinite(sector)]
        if len(sector) == 0:
            return float("inf")

        return float(np.percentile(sector, q))

    def get_front_distance(self) -> float:
        return min(
            self.get_range_percentile_360(345, 359, q=20),
            self.get_range_percentile_360(0, 15, q=20)
        )

    def get_left_distance(self) -> float:
        return self.get_range_percentile_360(75, 105, q=30)

    def get_right_distance(self) -> float:
        return self.get_range_percentile_360(255, 285, q=30)

    def detect_sign(self) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:

        if self.current_image is None:
            return None, None, None

        frame = self.current_image.copy()
        h, w = frame.shape[:2]

        roi_x1 = int(w * self.sign_roi_x_min_ratio)
        roi_x2 = int(w * self.sign_roi_x_max_ratio)
        roi = frame[:, roi_x1:roi_x2]

        results = self.yolo(roi, verbose=False)

        best_name = None
        best_box = None
        best_score = -1e9

        cv2.rectangle(frame, (roi_x1, 0), (roi_x2, h - 1), (255, 200, 0), 2)
        cv2.putText(frame, "ROI", (roi_x1 + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.7:
                    continue

                cls_id = int(box.cls[0])
                name = self.yolo.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                x1 += roi_x1
                x2 += roi_x1

                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h

                if area < self.min_sign_box_area:
                    continue

                cx = (x1 + x2) / 2.0
                img_cx = w / 2.0
                center_offset = abs(cx - img_cx) / img_cx   
                center_score = 1.0 - center_offset

                score = conf + 0.35 * center_score + 0.0001 * area

                if name == "trafficlight" and self.last_detected_light_color:
                    label = f"{name}:{self.last_detected_light_color}"
                else:
                    label = f"{name}:{conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if score > best_score:
                    best_score = score
                    best_name = name
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"SELECTED: {best_name}", (x1, min(h - 10, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return best_name, best_box, frame

    def classify_traffic_light(self, image: np.ndarray,
                                bbox: Tuple[int, int, int, int]) -> Optional[str]:
        if image is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.traffic_tf(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.traffic_model(tensor)
            probs  = F.softmax(logits, dim=1)
            pred   = int(torch.argmax(probs, dim=1).item())

        return self.traffic_classes[pred] if pred < len(self.traffic_classes) else None

    def can_trigger_sign(self) -> bool:
        return (time.time() - self.last_sign_trigger_time) > self.sign_cooldown_sec

    def set_state(self, new_state: RobotState):
        if self.state != new_state:
            self.state            = new_state
            self.state_start_time = time.time()
            self.get_logger().info(f"STATE -> {self.state.name}")

    # =========================================================
    # 狀態面板（右側 HUD）
    # =========================================================
    def draw_status_panel(self,
                          debug_img: np.ndarray,
                          sign_name: Optional[str],
                          detected_light_color: Optional[str]) -> np.ndarray:
                          
        h, w    = debug_img.shape[:2]
        panel_w = 340
        panel   = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)  

        def put(text, y, color=(220, 220, 220), scale=0.58, thickness=1):
            cv2.putText(panel, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                        cv2.LINE_AA)

        def hline(y, color=(60, 60, 80)):
            cv2.line(panel, (8, y), (panel_w - 8, y), color, 1)

        cv2.rectangle(panel, (0, 0), (panel_w, 42), (40, 40, 60), -1)
        put("ROBOT MONITOR", 28, (0, 220, 255), 0.70, 2)

        put("STATE", 65, (120, 120, 160), 0.50)
        state_color = {
            "WAITING":       (160, 160, 160),
            "FORWARD":       (60,  230, 60),
            "TURNING_LEFT":  (60,  200, 255),
            "TURNING_RIGHT": (60,  200, 255),
            "STOP_RED":      (60,  60,  255),
            "STOP_SIGN":     (60,  100, 255),
        }.get(self.state.name, (220, 220, 220))
        put(self.state.name, 92, state_color, 0.72, 2)

        put(f"Speed : {self.current_speed:.3f} m/s", 120, (180, 255, 180))
        put(f"Pending: {self.pending_turn or 'None'}", 145, (200, 200, 100))

        hline(158)

        put("SIGN DETECTION", 178, (120, 120, 160), 0.50)

        if sign_name:

            if sign_name == "trafficlight" and detected_light_color:
                display_name = f"{sign_name} ({detected_light_color.upper()})"
            else:
                display_name = sign_name

            put(display_name, 205, (255, 215, 0), 0.72, 2)

            if sign_name == "trafficlight" and detected_light_color:
                action_map = {
                    "red":    ("Action: STOP", (60, 60, 255)),
                    "yellow": ("Action: SLOW", (0, 200, 255)),
                    "green":  ("Action: GO",   (60, 230, 60)),
                }
                action_text, action_color = action_map.get(
                    detected_light_color, ("Action: ?", (150,150,150))
                )

            else:
                action_map = {
                    "crosswalk":  ("Action: SLOW DOWN",        (0, 200, 255)),
                    "speedlimit": ("Action: SLOW DOWN",        (0, 200, 255)),
                    "stop":       ("Action: FWD 5s -> STOP",   (60, 80, 255)),
                    "turn_left":  ("Action: PENDING LEFT 90deg",  (60, 255, 180)),
                    "turn_right": ("Action: PENDING RIGHT 90deg", (60, 255, 180)),
                }
                action_text, action_color = action_map.get(
                    sign_name, ("Action: (unknown)", (150,150,150))
                )

            put(action_text, 232, action_color, 0.60, 1)

        else:
            put("None detected", 205, (90, 90, 110))
            put("Action: -",     232, (90, 90, 110))

        hline(248)

        put("TRAFFIC LIGHT", 268, (120, 120, 160), 0.50)

        if detected_light_color:
            light_bgr = {
                "red":    (40,  40,  220),
                "yellow": (40,  200, 240),
                "green":  (40,  220, 40),
            }.get(detected_light_color, (180, 180, 180))

            put(detected_light_color.upper(), 295, light_bgr, 0.80, 2)

            cx = panel_w - 28
            cy = 288
            cv2.circle(panel, (cx, cy), 16, light_bgr, -1)
            cv2.circle(panel, (cx, cy), 16, (255, 255, 255), 1)
        else:
            put("N/A", 295, (90, 90, 110))

        hline(312)

        put("LIDAR DISTANCE (m)", 332, (120, 120, 160), 0.50)

        front_dist = self.get_front_distance()
        left_dist  = self.get_left_distance()
        right_dist = self.get_right_distance()

        def dist_color(d: float):
            if d < 0.35: return (60,  60,  220)   
            if d < 0.60: return (40,  200, 240)   
            return (40, 220, 40)                   

        def dist_bar(d: float, y: int):
            bar_max   = panel_w - 130
            bar_len   = min(int(bar_max * min(d, 2.0) / 2.0), bar_max)
            bar_color = dist_color(d)
            cv2.rectangle(panel, (115, y - 12), (115 + bar_len, y), bar_color, -1)

        put(f"Front: {front_dist:.2f}", 358, dist_color(front_dist), 0.62)
        dist_bar(front_dist, 358)

        put(f"Left : {left_dist:.2f}", 385, dist_color(left_dist), 0.62)
        dist_bar(left_dist, 385)

        put(f"Right: {right_dist:.2f}", 412, dist_color(right_dist), 0.62)
        dist_bar(right_dist, 412)

        hline(428)

        if self.state == RobotState.STOP_SIGN:
            elapsed = time.time() - self.state_start_time
            remain  = max(0.0, self.stop_sign_forward_duration - elapsed)
            put("STOP COUNTDOWN", 448, (120, 120, 160), 0.50)
            bar_w = panel_w - 24
            filled = int(bar_w * remain / self.stop_sign_forward_duration)
            cv2.rectangle(panel, (12, 454), (12 + bar_w, 470), (60, 60, 80), -1)
            cv2.rectangle(panel, (12, 454), (12 + filled, 470), (60, 80, 255), -1)
            put(f"{remain:.1f} s remaining", 490, (100, 120, 255), 0.62)

        if panel.shape[0] < h:
            pad   = np.zeros((h - panel.shape[0], panel_w, 3), dtype=np.uint8)
            pad[:] = (25, 25, 35)
            panel = np.vstack([panel, pad])

        return np.hstack([debug_img, panel])

    def control_loop(self):
        now = time.time()
        if self.slow_until_time is not None and now >= self.slow_until_time:
            self.current_speed = self.normal_speed
            self.slow_until_time = None
            self.get_logger().info("Slow duration ended -> resume normal speed")

        sign_name_raw, bbox_raw, debug_img = self.detect_sign()

        if sign_name_raw is not None:
            if sign_name_raw == self.prev_sign_name:
                self.stable_sign_count += 1
            else:
                self.prev_sign_name = sign_name_raw
                self.prev_sign_bbox = bbox_raw
                self.stable_sign_count = 1
        else:
            self.prev_sign_name = None
            self.prev_sign_bbox = None
            self.stable_sign_count = 0

        if self.stable_sign_count >= self.sign_stable_frames:
            sign_name = sign_name_raw
            bbox = bbox_raw
        else:
            sign_name = None
            bbox = None

        detected_light_color = None
        if sign_name == "trafficlight" and bbox is not None and self.current_image is not None:
            detected_light_color = self.classify_traffic_light(self.current_image, bbox)

        self.last_detected_sign = sign_name
        self.last_detected_light_color = detected_light_color

        if debug_img is not None and self.show_debug_image:
            display = self.draw_status_panel(debug_img, sign_name, detected_light_color)
            cv2.imshow("Robot Monitor", display)
            cv2.waitKey(1)

        if self.state == RobotState.WAITING:
            remain = self.wait_seconds - (now - self.start_time)
            if remain > 0:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(f"Waiting... {remain:.1f}s")
                return
            else:
                self.current_speed = self.normal_speed
                self.set_state(RobotState.FORWARD)
                
        if self.state == RobotState.TURNING_LEFT:
            if now - self.state_start_time < self.turn_duration_sec:
                self.publish_cmd(0.0, self.turn_angular_speed)
                return
            else:
                self.pending_turn = None
                self.set_state(RobotState.FORWARD)

        if self.state == RobotState.TURNING_RIGHT:
            if now - self.state_start_time < self.turn_duration_sec:
                self.publish_cmd(0.0, -self.turn_angular_speed)
                return
            else:
                self.pending_turn = None
                self.set_state(RobotState.FORWARD)

        if self.state == RobotState.STOP_RED:
            if detected_light_color == "green":
                self.current_speed = self.normal_speed
                self.get_logger().info("Traffic light: GREEN -> resume")
                self.set_state(RobotState.FORWARD)

            elif detected_light_color == "yellow":
                self.current_speed = self.slow_speed
                self.slow_until_time = now + self.slow_duration_sec
                self.get_logger().info(
                    f"Traffic light: YELLOW -> resume slowly for {self.slow_duration_sec:.1f}s"
                )
                self.set_state(RobotState.FORWARD)

            else:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info("Traffic light: RED -> stop")
                return

        if self.state == RobotState.STOP_SIGN:
            elapsed = now - self.state_start_time
            if elapsed < self.stop_sign_forward_duration:
                self.publish_cmd(self.current_speed, 0.0)
                self.get_logger().info(
                    f"STOP sign: moving {elapsed:.1f}/{self.stop_sign_forward_duration}s")
            else:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info("STOP sign: robot stopped permanently.")
            return

        if self.state == RobotState.FORWARD:

            front_dist = self.get_front_distance()
            left_dist  = self.get_left_distance()
            right_dist = self.get_right_distance()

            allow_turn_sign = front_dist < self.turn_sign_trigger_distance

            if sign_name and self.can_trigger_sign():

                if sign_name == "crosswalk":
                    self.current_speed = self.slow_speed
                    self.slow_until_time = now + self.slow_duration_sec
                    self.last_sign_trigger_time = now
                    self.get_logger().info(
                        f"Sign: crosswalk -> slow down for {self.slow_duration_sec:.1f}s"
                    )

                elif sign_name == "speedlimit":
                    self.current_speed = self.slow_speed
                    self.slow_until_time = now + self.slow_duration_sec
                    self.last_sign_trigger_time = now
                    self.get_logger().info(
                        f"Sign: speedlimit -> slow down for {self.slow_duration_sec:.1f}s"
                    )

                elif sign_name == "stop":
                    self.last_sign_trigger_time = now
                    self.get_logger().info("Sign: stop -> forward 5s then stop")
                    self.set_state(RobotState.STOP_SIGN)
                    return
                    
                elif sign_name == "trafficlight":
                    if detected_light_color == "red":
                        self.last_sign_trigger_time = now
                        self.get_logger().info("Sign: trafficlight red -> stop")
                        self.set_state(RobotState.STOP_RED)
                        self.publish_cmd(0.0, 0.0)
                        return

                    elif detected_light_color == "yellow":
                        self.current_speed = self.slow_speed
                        self.last_sign_trigger_time = now
                        self.get_logger().info(
                            f"Sign: trafficlight yellow -> slow down "
                        )

                    elif detected_light_color == "green":
                        self.current_speed = self.normal_speed
                        self.get_logger().info("Sign: trafficlight green -> keep moving")

                elif sign_name == "turn_left":
                    if allow_turn_sign:
                        self.pending_turn = "left"
                        self.last_sign_trigger_time = now
                        self.get_logger().info(
                            f"Sign: turn_left -> pending left turn (front_dist={front_dist:.2f})"
                        )
                    else:
                        self.get_logger().info(
                            f"Ignore turn_left: too far from junction (front_dist={front_dist:.2f})"
                        )

                elif sign_name == "turn_right":
                    if allow_turn_sign:
                        self.pending_turn = "right"
                        self.last_sign_trigger_time = now
                        self.get_logger().info(
                            f"Sign: turn_right -> pending right turn (front_dist={front_dist:.2f})"
                        )
                    else:
                        self.get_logger().info(
                            f"Ignore turn_right: too far from junction (front_dist={front_dist:.2f})"
                        )


            self.get_logger().info(
                f"FORWARD | spd={self.current_speed:.3f} | "
                f"F={front_dist:.2f} L={left_dist:.2f} R={right_dist:.2f} | "
                f"sign={sign_name} light={detected_light_color} pending={self.pending_turn}"
            )

            if front_dist < self.front_wall_threshold:
                self.get_logger().info("Front wall detected -> deciding turn direction")

                if self.pending_turn == "left":
                    self.set_state(RobotState.TURNING_LEFT)
                elif self.pending_turn == "right":
                    self.set_state(RobotState.TURNING_RIGHT)
                else:
                    if left_dist < self.wall_follow_trigger and right_dist >= self.wall_follow_trigger:
                        self.set_state(RobotState.TURNING_RIGHT)
                    elif right_dist < self.wall_follow_trigger and left_dist >= self.wall_follow_trigger:
                        self.set_state(RobotState.TURNING_LEFT)
                    else:
                        if left_dist >= right_dist:
                            self.set_state(RobotState.TURNING_LEFT)
                        else:
                            self.set_state(RobotState.TURNING_RIGHT)
                return

            self.publish_cmd(self.current_speed, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
