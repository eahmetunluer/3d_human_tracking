from ultralytics import YOLO
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler
import numpy as np
import torch
import torchvision.transforms.functional as F
from collections import deque
from std_msgs.msg import ColorRGBA
import math

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")
model.to('cuda')  # Move the model to GPU

# Initialize ROS node
rospy.init_node('person_tracker', anonymous=True)

# Create a CvBridge object
bridge = CvBridge()

# Initialize global variables
latest_color_frame = None
latest_depth_frame = None
camera_info = None

# Create publishers
tracked_image_pub = rospy.Publisher('tracked_image', Image, queue_size=10)
person_position_pub = rospy.Publisher('person_position', PointStamped, queue_size=10)
person_marker_pub = rospy.Publisher('person_marker', Marker, queue_size=10)
camera_marker_pub = rospy.Publisher('camera_marker', Marker, queue_size=10)
ring_marker_pub = rospy.Publisher('distance_ring', Marker, queue_size=10)

def color_image_callback(msg):
    global latest_color_frame
    latest_color_frame = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_image_callback(msg):
    global latest_depth_frame
    depth_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # Convert uint16 to float32 and ensure it's writable
    latest_depth_frame = np.array(depth_frame, dtype=np.float32)

def camera_info_callback(msg):
    global camera_info
    camera_info = msg

# Subscribe to the necessary topics
rospy.Subscriber("/camera/color/image_raw", Image, color_image_callback)
rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_image_callback)
rospy.Subscriber("/camera/color/camera_info", CameraInfo, camera_info_callback)

# Add a rate limiter
rate = rospy.Rate(10)  # 10 Hz

def pixel_to_3d(x, y, depth, camera_info):
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]
    
    z = depth / 1000.0  # Convert depth from mm to meters
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    return torch.tensor([x, y, z], device='cuda')  # Return a GPU tensor

def create_person_marker(x, y, z, tracker_id, action=Marker.ADD):
    marker = Marker()
    marker.header.frame_id = "camera_color_optical_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "person_markers"
    marker.id = tracker_id
    marker.type = Marker.SPHERE
    marker.action = action
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    return marker

def create_camera_marker():
    marker = Marker()
    marker.header.frame_id = "camera_color_optical_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "camera_marker"
    marker.id = 0
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    
    # Set the orientation to point along the camera's z-axis (forward)
    q = quaternion_from_euler(-np.pi/2, 0, -np.pi/2)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]
    
    marker.scale.x = 0.2  # Arrow length
    marker.scale.y = 0.02  # Arrow width
    marker.scale.z = 0.02  # Arrow height
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    return marker

def get_median_depth(depth_frame, x1, y1, x2, y2, kernel_size=5):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Define the region of interest
    roi_x1 = max(0, center_x - kernel_size // 2)
    roi_x2 = min(depth_frame.shape[1], center_x + kernel_size // 2 + 1)
    roi_y1 = max(0, center_y - kernel_size // 2)
    roi_y2 = min(depth_frame.shape[0], center_y + kernel_size // 2 + 1)
    
    # Extract the region of interest
    roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Calculate the median depth, ignoring zero values
    valid_depths = roi[roi > 0]
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    else:
        return 0  # Return 0 if no valid depth values are found

def create_ring_marker(inner_radius, outer_radius, r, g, b, a):
    marker = Marker()
    marker.header.frame_id = "camera_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "distance_ring"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a

    # Create ring points
    num_points = 100
    
    for i in range(num_points):
        theta1 = i * 2 * math.pi / num_points
        theta2 = (i + 1) * 2 * math.pi / num_points
        
        inner_x1 = inner_radius * math.cos(theta1)
        inner_y1 = inner_radius * math.sin(theta1)
        outer_x1 = outer_radius * math.cos(theta1)
        outer_y1 = outer_radius * math.sin(theta1)
        
        inner_x2 = inner_radius * math.cos(theta2)
        inner_y2 = inner_radius * math.sin(theta2)
        outer_x2 = outer_radius * math.cos(theta2)
        outer_y2 = outer_radius * math.sin(theta2)
        
        # First triangle
        marker.points.append(Point(x=inner_x1, y=inner_y1, z=0))
        marker.points.append(Point(x=outer_x1, y=outer_y1, z=0))
        marker.points.append(Point(x=inner_x2, y=inner_y2, z=0))
        
        # Second triangle
        marker.points.append(Point(x=inner_x2, y=inner_y2, z=0))
        marker.points.append(Point(x=outer_x1, y=outer_y1, z=0))
        marker.points.append(Point(x=outer_x2, y=outer_y2, z=0))

    return marker

class PersonTracker:
    def __init__(self, tracker_id, window_size=3):
        self.tracker_id = tracker_id
        self.positions = deque(maxlen=window_size)
    
    def update(self, x, y, z):
        self.positions.append(np.array([x, y, z]))
    
    def get_smoothed_position(self):
        if not self.positions:
            return None
        return np.mean(self.positions, axis=0)

def create_point_stamped(x, y, z):
    point_msg = PointStamped()
    point_msg.header.frame_id = "camera_color_optical_frame"
    point_msg.header.stamp = rospy.Time.now()
    point_msg.point.x = x
    point_msg.point.y = y
    point_msg.point.z = z
    return point_msg

# Dictionary to store PersonTracker objects
person_trackers = {}

while not rospy.is_shutdown():
    if latest_color_frame is not None and latest_depth_frame is not None and camera_info is not None:
        # Convert frames to GPU tensors
        color_frame_gpu = torch.from_numpy(latest_color_frame).float().permute(2, 0, 1).unsqueeze(0).to('cuda')
        depth_frame_gpu = torch.from_numpy(latest_depth_frame).to('cuda')

        # Normalize color frame to [0, 1] range
        color_frame_gpu = color_frame_gpu / 255.0

        # Perform tracking using YOLOv8
        results = model.track(source=color_frame_gpu, classes=0, persist=True, tracker="bytetrack.yaml")

        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
            tracker_id = int(result.id[0]) if result.id is not None else 0
            
            # Get median depth for the tracked person
            depth = get_median_depth(latest_depth_frame, x1, y1, x2, y2)
            
            # Calculate 3D coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            x, y, z = pixel_to_3d(center_x, center_y, depth, camera_info)
            
            # Update or create PersonTracker
            if tracker_id not in person_trackers:
                person_trackers[tracker_id] = PersonTracker(tracker_id)
            person_trackers[tracker_id].update(x.item(), y.item(), z.item())
            
            # Get smoothed position
            smoothed_position = person_trackers[tracker_id].get_smoothed_position()
            
            if smoothed_position is not None:
                # Publish smoothed 3D position
                point_msg = create_point_stamped(smoothed_position[0], smoothed_position[1], smoothed_position[2])
                person_position_pub.publish(point_msg)
                
                # Publish marker for RViz
                marker = create_person_marker(smoothed_position[0], smoothed_position[1], smoothed_position[2], tracker_id)
                person_marker_pub.publish(marker)

        # Publish camera marker
        camera_marker = create_camera_marker()
        camera_marker_pub.publish(camera_marker)

        # Publish ring marker
        ring = create_ring_marker(0.3, 3.0, 0.0, 1.0, 0.0, 0.2)  # Blue, very transparent
        ring_marker_pub.publish(ring)

        # Convert back to CPU and numpy for publishing
        color_frame_np = color_frame_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
        color_frame_np = (color_frame_np * 255).astype(np.uint8)

        # Publish the tracked image (without drawing)
        tracked_image_msg = bridge.cv2_to_imgmsg(color_frame_np, "bgr8")
        tracked_image_pub.publish(tracked_image_msg)

    rate.sleep()

# Release resources
cv2.destroyAllWindows()