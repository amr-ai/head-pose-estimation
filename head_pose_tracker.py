import cv2
import numpy as np
from sort import Sort
from collections import defaultdict
import time
import os
from ultralytics import YOLO

class HeadPoseTracker:
    def __init__(self, video_source="0"): #put your video path here
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO('yolov8n-pose.pt')
        self.tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)
        self.cheat_counter = 0
        self.cheat_timers = defaultdict(lambda: {"dir": None, "start": None, "screenshot_taken": False})
        
        self.direction_changes = defaultdict(lambda: {
            "current_dir": None,
            "look_count": 0,
            "last_change_time": 0,
            "look_times": []
        })

        os.makedirs("screenshots", exist_ok=True)
        
        # Performance optimizations
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        self.last_results = {}  # Cache results for skipped frames
        
        # Reduce model input size for faster inference
        self.model_input_size = 640  # Smaller input size

    def estimate_pose(self, keypoints):
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        
        # حساب المسافة النسبية بين العين والأذن
        left_ear_rel = left_ear[0] - left_eye[0]
        right_ear_rel = right_ear[0] - right_eye[0]
        
        direction = "Forward"
        cheat_flag = 0
        threshold = 12
        
        if left_ear_rel > threshold and right_ear_rel > -threshold:
            direction = "Left"
            cheat_flag = 1
        elif right_ear_rel < -threshold and left_ear_rel < threshold:
            direction = "Right"
            cheat_flag = 1
            
        return direction, cheat_flag, (left_ear_rel + right_ear_rel) / 2

    def check_direction_changes(self, track_id, direction, current_time):
        tracking_data = self.direction_changes[track_id]
        
        if direction not in ["Left", "Right", "Forward"]:
            return False
        
        if tracking_data["current_dir"] is None:
            tracking_data["current_dir"] = direction
            if direction in ["Left", "Right"]:
                tracking_data["look_count"] = 1
                tracking_data["look_times"] = [(direction, current_time)]
            return False
            
        if direction != tracking_data["current_dir"]:
            if direction in ["Left", "Right"] and tracking_data["current_dir"] == "Forward":
                tracking_data["look_count"] += 1
                tracking_data["look_times"].append((direction, current_time))
            
            tracking_data["current_dir"] = direction
            tracking_data["last_change_time"] = current_time
            
            tracking_data["look_times"] = [(d, t) for d, t in tracking_data["look_times"] 
                                         if current_time - t <= 15]
            tracking_data["look_count"] = len(tracking_data["look_times"])
            
            if tracking_data["look_count"] >= 3:
                return True
                
        return False

    def save_screenshot(self, frame, track_id, cheat_type, x1, y1, x2, y2):
        padding = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"screenshots/{cheat_type}_ID{track_id}_{timestamp}.jpg", frame[y1:y2, x1:x2])

    def process_frame(self, frame, force_detection=False):
        self.frame_count += 1
        
        # Skip frames for performance - only detect every nth frame
        if not force_detection and self.frame_count % self.frame_skip != 0:
            # Use cached results from previous frame
            return self.draw_cached_results(frame)
        
        # Resize frame for faster YOLO inference
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        scale = min(self.model_input_size / w, self.model_input_size / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))
        else:
            resized_frame = frame
            scale = 1
        
        # Run YOLO on resized frame
        results = self.model(resized_frame, verbose=False)
        bboxes = []
        keypoints_list = []
        
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
                
            for box, kpts in zip(result.boxes, result.keypoints):
                # Scale back to original frame size
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                
                # Filter small detections
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue
                    
                bboxes.append([x1, y1, x2, y2, box.conf.item()])
                
                # Scale keypoints back
                scaled_kpts = []
                for kp in kpts.xy[0].tolist():
                    scaled_kpts.append([kp[0]/scale, kp[1]/scale])
                keypoints_list.append((x1, y1, x2, y2, scaled_kpts))
        
        # Update tracker
        tracked = self.tracker.update(np.array(bboxes)) if bboxes else np.empty((0, 5))
        
        # Cache results for next frames
        self.last_results = {
            'tracked': tracked,
            'keypoints_list': keypoints_list
        }
        
        return self.draw_results(original_frame, tracked, keypoints_list)

    def draw_cached_results(self, frame):
        if not self.last_results:
            return frame
            
        return self.draw_results(frame, 
                               self.last_results['tracked'], 
                               self.last_results['keypoints_list'])

    def draw_results(self, frame, tracked, keypoints_list):
        for track in tracked:
            x1, y1, x2, y2, track_id = map(int, track)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Find matching keypoints (optimized matching)
            best_match = None
            max_iou = 0
            for kp_data in keypoints_list:
                kp_x1, kp_y1, kp_x2, kp_y2, keypoints = kp_data
                
                # Quick overlap check before calculating IoU
                if not (x1 < kp_x2 and x2 > kp_x1 and y1 < kp_y2 and y2 > kp_y1):
                    continue
                    
                xi1, yi1 = max(x1, kp_x1), max(y1, kp_y1)
                xi2, yi2 = min(x2, kp_x2), min(y2, kp_y2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                
                if inter_area > 0:
                    union_area = (x2-x1)*(y2-y1) + (kp_x2-kp_x1)*(kp_y2-kp_y1) - inter_area
                    iou = inter_area/union_area if union_area > 0 else 0
                    if iou > max_iou:
                        max_iou = iou
                        best_match = keypoints
            
            if best_match and max_iou > 0.3:
                direction, cheat, angle = self.estimate_pose(best_match)
                now = time.time()
                
                # Check for cheating patterns
                multiple_looks = self.check_direction_changes(track_id, direction, now)
                timer = self.cheat_timers[track_id]

                if direction in ["Left", "Right"]:
                    if timer["dir"] == direction and timer["start"] and now - timer["start"] > 5:
                        cheat = 1
                        if not timer["screenshot_taken"]:
                            self.save_screenshot(frame, track_id, "continuous", x1, y1, x2, y2)
                            self.cheat_timers[track_id]["screenshot_taken"] = True
                            self.cheat_counter += 1
                    else:
                        self.cheat_timers[track_id] = {"dir": direction, "start": now, "screenshot_taken": False}
                else:
                    self.cheat_timers[track_id] = {"dir": None, "start": None, "screenshot_taken": False}
                
                if multiple_looks:
                    cheat = 1
                    self.save_screenshot(frame, track_id, "multiple", x1, y1, x2, y2)
                    self.direction_changes[track_id]["look_count"] = 0
                    self.direction_changes[track_id]["look_times"] = []
                    self.cheat_counter += 1

                # Display information
                text_y = y1 - 10 if y1 > 30 else y2 + 20
                cv2.putText(frame, f"ID: {track_id}", (x1, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                text_y = y2 + 30 if y1 > 30 else y2 + 50
                status = f"{direction} ({int(angle)})"
                if cheat:
                    status += " (CHEAT)"
                cv2.putText(frame, status, (x1, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if cheat else (0, 255, 0), 2)
                
                # Draw direction arrow
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if direction == "Left":
                    cv2.arrowedLine(frame, (center_x + 20, center_y), (center_x - 20, center_y), (0, 0, 255), 2)
                elif direction == "Right":
                    cv2.arrowedLine(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)

        # Display cheat counter
        cv2.putText(frame, f"Cheating Cases: {self.cheat_counter}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Exam Surveillance", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HeadPoseTracker()
    tracker.run()
