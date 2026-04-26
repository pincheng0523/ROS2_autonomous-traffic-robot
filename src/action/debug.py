#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarDebugNode(Node):
    def __init__(self):
        super().__init__("lidar_debug_node")

        self.scan_msg: Optional[LaserScan] = None

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        # 每 0.5 秒輸出一次
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info("Lidar debug node started. Listening on /scan")

    def scan_callback(self, msg: LaserScan):
        self.scan_msg = msg

    def get_sector_percentile(self, start_deg: float, end_deg: float, q: float = 30.0) -> float:
        if self.scan_msg is None:
            return float("inf")

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.scan_msg.angle_min
        angle_inc = self.scan_msg.angle_increment

        start_idx = int((math.radians(start_deg) - angle_min) / angle_inc)
        end_idx = int((math.radians(end_deg) - angle_min) / angle_inc)

        start_idx = max(0, min(start_idx, len(ranges) - 1))
        end_idx = max(0, min(end_idx, len(ranges) - 1))

        if start_idx <= end_idx:
            sector = ranges[start_idx:end_idx + 1]
        else:
            sector = np.concatenate([ranges[start_idx:], ranges[:end_idx + 1]])

        sector = sector[np.isfinite(sector)]
        if len(sector) == 0:
            return float("inf")

        return float(np.percentile(sector, q))

    def timer_callback(self):
        if self.scan_msg is None:
            self.get_logger().info("Waiting for /scan ...")
            return

        sectors = [
            ("m180", -180, -150),
            ("m150", -150, -120),
            ("m120", -120,  -90),
            ("m090",  -90,  -60),
            ("m060",  -60,  -30),
            ("m030",  -30,    0),
            ("p000",    0,   30),
            ("p030",   30,   60),
            ("p060",   60,   90),
            ("p090",   90,  120),
            ("p120",  120,  150),
            ("p150",  150,  180),
        ]

        results = []
        for name, a, b in sectors:
            d = self.get_sector_percentile(a, b, q=30)
            if math.isinf(d):
                results.append(f"{name}=inf")
            else:
                results.append(f"{name}={d:.2f}")

        self.get_logger().info("SCAN12 | " + " | ".join(results))


def main(args=None):
    rclpy.init(args=args)
    node = LidarDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
