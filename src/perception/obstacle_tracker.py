import numpy as np
import pybullet as p
import cv2
from filterpy.kalman import KalmanFilter

class ObstacleTracker:
    def __init__(self, n_obstacles=2, dt=1/240., camera_settings=None):
        self.n_obstacles = n_obstacles
        self.dt = dt
        self.camera_settings = camera_settings
        
        # tracker parameters
        tracker_settings = self.camera_settings.get("tracker_settings", {})
        self.process_noise = tracker_settings.get("process_noise", 0.1)
        self.measurement_noise = tracker_settings.get("measurement_noise", 0.1)
        self.initial_covariance = tracker_settings.get("initial_covariance", 100.0)
        self.visualization_box_size = tracker_settings.get("visualization_box_size", 0.3)
        
        self.initialized = False
        
        # kalman filters for each obstacle
        self.filters = []
        for _ in range(n_obstacles):
            kf = KalmanFilter(dim_x=7, dim_z=4)  # state: [x,y,z,vx,vy,vz,r], measurement: [x,y,z,r]
            
            # state transition matrix F
            kf.F = np.array([
                [1, 0, 0, dt, 0, 0, 0],  # x = x + vx*dt
                [0, 1, 0, 0, dt, 0, 0],  # y = y + vy*dt
                [0, 0, 1, 0, 0, dt, 0],  # z = z + vz*dt
                [0, 0, 0, 1, 0, 0, 0],   # vx = vx
                [0, 0, 0, 0, 1, 0, 0],   # vy = vy
                [0, 0, 0, 0, 0, 1, 0],   # vz = vz
                [0, 0, 0, 0, 0, 0, 1],   # r = r
            ])
            
            # observation matrix H
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, 0, 0, 0],  # y
                [0, 0, 1, 0, 0, 0, 0],  # z
                [0, 0, 0, 0, 0, 0, 1],  # r
            ])
            
            # process noise covariance Q
            kf.Q = np.eye(7) * self.process_noise
            
            # measurement noise covariance R
            kf.R = np.eye(4) * self.measurement_noise
            
            # initial state covariance P
            kf.P *= self.initial_covariance
            
            # initial state x
            kf.x = np.zeros(7)
            
            self.filters.append(kf)

    def convert_depth_to_meters(self, depth_buffer):
        """Convert depth buffer to metric depth."""
        far = self.camera_settings["far"]
        near = self.camera_settings["near"]

        return far * near / (far - (far - near) * depth_buffer)    
    
    # TODO: review CV1 image to world coordinate conversion
    def pixel_to_world(self, pixel_x, pixel_y, depth):
        """Convert pixel coordinates to world coordinates."""
        width = self.camera_settings["width"]
        height = self.camera_settings["height"]
        fov = self.camera_settings["fov"]
        
        # 1. camera param
        cam_pos = np.array(self.camera_settings["stat_cam_pos"])
        target_pos = np.array(self.camera_settings["stat_cam_target_pos"])
        
        # 2. pixel to NDC
        ndc_x = (2.0 * pixel_x - width) / width
        ndc_y = -(2.0 * pixel_y - height) / height
        
        # 3. NDC to camera space
        aspect = width / height
        tan_half_fov = np.tan(np.deg2rad(fov / 2))
        cam_x = ndc_x * aspect * tan_half_fov * depth
        cam_y = ndc_y * tan_half_fov * depth
        cam_z = depth
        
        # 4. create camera space basis
        forward = target_pos - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 5. camera space to world space
        R = np.column_stack([right, up, forward])
        cam_point = np.array([cam_x, cam_y, cam_z])
        world_point = cam_pos + R @ cam_point
        
        return world_point
    
    def calculate_metric_radius(self, area, depth):
        """Calculate sphere radius with corrective factor"""
        width = self.camera_settings["width"]
        fov_rad = np.deg2rad(self.camera_settings["fov"])
        
        # projected pixel radius
        pixel_radius = np.sqrt(area / np.pi)
        
        # calculate base radius
        base_radius = (pixel_radius * depth * 2 * np.tan(fov_rad/2)) / width
        
        # TODO: review this corrective factor. It is a rough estimation.
        corrective_factor = 2.85
        metric_radius = base_radius * corrective_factor
        
        # # Debug
        # print(f"\nRadius Calculation Debug:")
        # print(f"Area: {area} pixels")
        # print(f"Pixel radius: {pixel_radius} pixels")
        # print(f"Depth: {depth}m")
        # print(f"Base radius: {base_radius}m")
        # print(f"Corrected radius: {metric_radius}m")
        
        return metric_radius
    
    def detect_obstacles(self, rgb, depth, seg):
        """Detect red spheres using segmentation mask"""
        detections = []
        potential_balls = []
        
        # 1. find red objects in segmentation mask
        for obj_id in np.unique(seg)[1:]:
            mask = (seg == obj_id).astype(np.uint8)
            masked_rgb = rgb * mask[..., None]
            non_zero = masked_rgb[mask > 0]
            
            if len(non_zero) == 0:
                continue
                
            mean_color = np.mean(non_zero, axis=0)
            r, g, b = mean_color[:3]
            
            # loosen the red color detection
            is_red = (r > 150 and  # high R
                    max(g, b) < r * 0.5 and  # G and B much lower than R
                    abs(g - b) < 20)  # G and B close
            
            if not is_red:
                continue
                
            # find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            for contour in contours:
                area = cv2.contourArea(contour)
                # adjust area constraint based on actual ball size
                if area < 500 or area > 5000:  # can be tuned
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity < 0.8:  # circularity constraint
                    continue
                    
                # calc contour center
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                    
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                # check depth
                depth_value = depth[cy, cx]
                if not (0 < depth_value < 1):
                    continue
                    
                metric_depth = self.convert_depth_to_meters(depth_value)
                
                # if depth is out of range, skip
                if not (0.5 < metric_depth < 3.0):
                    continue
                    
                # pixel to world
                world_pos = self.pixel_to_world(cx, cy, metric_depth)
                
                # check world position
                if not (0.5 < world_pos[2] < 3.0):
                    continue
                    
                if abs(world_pos[0]) > 2.0 or abs(world_pos[1]) > 2.0:
                    continue
                
                # criteria passed, add to potential balls
                potential_balls.append({
                    'id': obj_id,
                    'center': (cx, cy),
                    'world_pos': world_pos,
                    'depth': metric_depth,
                    'area': area,
                    'circularity': circularity,
                    'color': (r, g, b)
                })
                
                # print(f"\nPotential ball found:")
                # print(f"  ID: {obj_id}")
                # print(f"  RGB: {(r, g, b)}")
                # print(f"  Position: ({cx}, {cy})")
                # print(f"  Depth: {metric_depth:.3f}m")
                # print(f"  World pos: {world_pos}")
                # print(f"  Area: {area}")
                # print(f"  Circularity: {circularity}")
        
        # 2. cluster potential balls to avoid duplicate detection
        if potential_balls:
            # 按z坐标排序，优先选择近处的球
            potential_balls.sort(key=lambda x: x['depth'])
            
            # 选择不同空间位置的球（避免重复检测同一个球）
            selected_balls = []
            for ball in potential_balls:
                # 检查是否与已选择的球太近
                too_close = False
                for selected in selected_balls:
                    dist = np.linalg.norm(ball['world_pos'] - selected['world_pos'])
                    if dist < 0.5:  # 如果距离小于0.5米，认为是同一个球
                        too_close = True
                        break
                
                if not too_close:
                    selected_balls.append(ball)
                    if len(selected_balls) >= self.n_obstacles:
                        break
            
            # 3. 转换为检测结果格式
            for ball in selected_balls:
                world_pos = ball['world_pos']
                metric_radius = self.calculate_metric_radius(ball['area'], ball['depth'])
                
                print(f"\nConfirmed ball detection:")
                print(f"  ID: {ball['id']}")
                print(f"  World position: {world_pos}")
                print(f"  Metric radius: {metric_radius:.3f}m")
                
                detections.append(np.array([world_pos[0], world_pos[1], world_pos[2], metric_radius]))
        
        return detections
    
    def update(self, detections):
        """Update Kalman filters with new detections."""
        if len(detections) == 0:
            for kf in self.filters:
                kf.predict()
            predicted_positions = np.zeros((self.n_obstacles, 3))
            for i, kf in enumerate(self.filters):
                predicted_positions[i] = kf.x[:3]
            return predicted_positions
            
        # init upon first detection
        if not self.initialized:
            for i, detection in enumerate(detections):
                if i < len(self.filters):
                    self.filters[i].x[:4] = detection # set initial position
            self.initialized = True
            
        # predict
        for kf in self.filters:
            kf.predict()
            
        # update
        detections = sorted(detections, key=lambda x: x[0]) # sort by x position, otherwise wrong comparison
        for i, (kf, detection) in enumerate(zip(self.filters, detections)):
            kf.update(detection)
            
        # get predicted positions
        predicted_positions = np.zeros((self.n_obstacles, 3))
        for i, kf in enumerate(self.filters):
            predicted_positions[i] = kf.x[:3]
            
        return predicted_positions
    
    # 3d bounding box
    def visualize_tracking_3d(self, tracked_positions):
        """Visualize tracking boxes in 3D space"""
        debug_ids = []
        
        for i, pos in enumerate(tracked_positions):
            # 从对应的卡尔曼滤波器中获取估计的半径
            estimated_radius = self.filters[i].x[6]  # 第7维是半径
            # 使用直径作为box的边长(比半径略大一点以便观察)
            size = estimated_radius # 可以根据需要调整这个系数
            half_size = size/2
            
            # 定义立方体的8个顶点
            corners = [
                [pos[0]-half_size, pos[1]-half_size, pos[2]-half_size],
                [pos[0]+half_size, pos[1]-half_size, pos[2]-half_size],
                [pos[0]+half_size, pos[1]+half_size, pos[2]-half_size],
                [pos[0]-half_size, pos[1]+half_size, pos[2]-half_size],
                [pos[0]-half_size, pos[1]-half_size, pos[2]+half_size],
                [pos[0]+half_size, pos[1]-half_size, pos[2]+half_size],
                [pos[0]+half_size, pos[1]+half_size, pos[2]+half_size],
                [pos[0]-half_size, pos[1]+half_size, pos[2]+half_size]
            ]
            
            # 画线的代码保持不变...
            # 底部四条边
            debug_ids.append(p.addUserDebugLine(corners[0], corners[1], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[1], corners[2], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[2], corners[3], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[3], corners[0], [0, 1, 0]))
            
            # 顶部四条边
            debug_ids.append(p.addUserDebugLine(corners[4], corners[5], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[5], corners[6], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[6], corners[7], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[7], corners[4], [0, 1, 0]))
            
            # 竖直四条边
            debug_ids.append(p.addUserDebugLine(corners[0], corners[4], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[1], corners[5], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[2], corners[6], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[3], corners[7], [0, 1, 0]))
        
        return debug_ids