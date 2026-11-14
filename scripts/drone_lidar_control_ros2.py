#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State, PositionTarget
from sensor_msgs.msg import LaserScan

class DroneLidarController(Node):
    def __init__(self):
        super().__init__('drone_lidar_control_node')
        
        # Variabel untuk menyimpan data LiDAR & status drone
        self.dist_right = 10.0
        self.dist_left = 10.0
        self.dist_up = 10.0
        self.dist_down = 10.0
        self.fcu_state = State()

        # Publisher untuk kecepatan (Velocity Setpoint)
        self.speedy_pub = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 10)

        # Subscribers untuk State dan 4 LiDAR (sesuai model SDF)
        self.create_subscription(State, "/mavros/state", self.fcu_state_cb, 10)
        self.create_subscription(LaserScan, "/lidar_right", self.lidar_right_cb, 10)
        self.create_subscription(LaserScan, "/lidar_left", self.lidar_left_cb, 10)
        self.create_subscription(LaserScan, "/lidar_up", self.lidar_up_cb, 10)
        self.create_subscription(LaserScan, "/lidar_down", self.lidar_down_cb, 10)

        # Service Clients untuk MAVROS
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")

    # --- Callbacks ---
    def fcu_state_cb(self, data):
        self.fcu_state = data

    def lidar_right_cb(self, data):
        if len(data.ranges) > 0:
            self.dist_right = data.ranges[0]

    def lidar_left_cb(self, data):
        if len(data.ranges) > 0:
            self.dist_left = data.ranges[0]

    def lidar_up_cb(self, data):
        if len(data.ranges) > 0:
            self.dist_up = data.ranges[0]

    def lidar_down_cb(self, data):
        if len(data.ranges) > 0:
            self.dist_down = data.ranges[0]

    # --- Fungsi Service MAVROS (ROS 2) ---
    def set_arm(self, armed):
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Menunggu service arming...")
        req = CommandBool.Request()
        req.value = armed
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_mode(self, mode_str):
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Menunggu service set_mode...")
        req = SetMode.Request()
        req.custom_mode = mode_str
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_takeoff(self, alt_target):
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Menunggu service takeoff...")
        req = CommandTOL.Request()
        req.altitude = float(alt_target)
        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    # --- Fungsi Kontrol Kecepatan ---
    def move_at_velocity(self, lin_x=0.0, lin_y=0.0, lin_z=0.0):
        speed_data = PositionTarget()
        speed_data.coordinate_frame = 8
        speed_data.type_mask = 1991

        speed_data.velocity.x = float(lin_y)
        speed_data.velocity.y = float(-lin_x)
        speed_data.velocity.z = float(lin_z)

        self.speedy_pub.publish(speed_data)

    def guided_and_arm(self):
        rate = self.create_rate(2)
        self.get_logger().info("Mengubah mode ke GUIDED...")
        while self.fcu_state.mode != 'GUIDED':
            self.set_mode('GUIDED')
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()

        time.sleep(1)
        self.get_logger().info("Melakukan Arming...")
        while not self.fcu_state.armed:
            self.set_arm(True)
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()
        time.sleep(1)

    # --- Logika Utama Misi ---
    def run_mission(self):
        rate = self.create_rate(10)  # 10 Hz loop rate

        self.guided_and_arm()
        self.get_logger().info("Taking off ke ketinggian 3 meter...")
        self.set_takeoff(3.0)
        time.sleep(10)

        self.get_logger().info("Mulai terbang maju sambil memantau sensor LiDAR Kiri & Kanan...")
        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time < 15):
            # Memproses callback agar data subscription terus ter-update
            rclpy.spin_once(self, timeout_sec=0.01)

            # Cetak informasi jarak sensor ke terminal
            self.get_logger().info(
                f"LiDAR -> Kiri: {self.dist_left:.2f}m | Kanan: {self.dist_right:.2f}m | Bawah: {self.dist_down:.2f}m"
            )

            # Logika Penghindaran Sederhana
            if self.dist_right < 0.8:
                self.get_logger().warn("Dinding terlalu dekat di KANAN! Geser ke KIRI.")
                self.move_at_velocity(0.5, 0.5, 0)
            elif self.dist_left < 0.8:
                self.get_logger().warn("Dinding terlalu dekat di KIRI! Geser ke KANAN.")
                self.move_at_velocity(0.5, -0.5, 0)
            else:
                # Jalur aman, maju lurus
                self.move_at_velocity(1.0, 0, 0)

            rate.sleep()

        self.get_logger().info("Stopping drone...")
        self.move_at_velocity(0, 0, 0)
        time.sleep(2)

        self.get_logger().info("Landing...")
        self.set_mode('LAND')
        time.sleep(10)

def main(args=None):
    rclpy.init(args=args)
    controller = DroneLidarController()
    try:
        controller.run_mission()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
