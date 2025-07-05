import cv2
import numpy as np
import mediapipe as mp
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd
from tqdm import tqdm
import gradio as gr
import tempfile
import shutil
import zipfile
from datetime import datetime

class MPIPoseEstimator:
    """MediaPipe Pose Estimation for Video Processing - Gradio Version"""
    
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose estimator
        
        Args:
            model_complexity: 0, 1, or 2. Higher = more accurate but slower
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # MPI-INF-3DHP joint mapping to MediaPipe landmarks
        self.mpi_to_mediapipe_mapping = {
            0: 0,   # head_top -> nose (approximate)
            1: 0,   # neck -> nose (approximate, will adjust)
            2: 12,  # right_shoulder -> right_shoulder
            3: 14,  # right_elbow -> right_elbow
            4: 16,  # right_wrist -> right_wrist
            5: 11,  # left_shoulder -> left_shoulder
            6: 13,  # left_elbow -> left_elbow
            7: 15,  # left_wrist -> left_wrist
            8: 24,  # right_hip -> right_hip
            9: 26,  # right_knee -> right_knee
            10: 28, # right_ankle -> right_ankle
            11: 23, # left_hip -> left_hip
            12: 25, # left_knee -> left_knee
            13: 27, # left_ankle -> left_ankle
            14: 0,  # spine -> nose (approximate)
            15: 24, # pelvis -> right_hip (approximate)
            16: 20, # right_hand -> right_index
            17: 19  # left_hand -> left_index
        }
        
        self.results_cache = {}

    def estimate_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose using MediaPipe
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Pose landmarks and confidence scores
        """
        try:
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                landmarks = []
                confidences = []
                
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                    confidences.append(landmark.visibility)
                    
                return {
                    'landmarks': np.array(landmarks),
                    'confidences': np.array(confidences),
                    'pose_landmarks': results.pose_landmarks
                }
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            
        return None

    def create_skeleton_2d(self, landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Create 2D skeleton visualization
        
        Args:
            landmarks: Pose landmarks array
            width: Image width
            height: Image height
            
        Returns:
            Skeleton visualization image
        """
        # Create blank image
        skeleton_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Define pose connections (MediaPipe pose connections)
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Body
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28),
            (28, 30), (28, 32)
        ]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for landmark in landmarks:
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            points.append((x, y))
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(skeleton_img, points[start_idx], points[end_idx], (0, 255, 0), 2)
        
        # Draw joints
        for i, point in enumerate(points):
            cv2.circle(skeleton_img, point, 4, (0, 0, 255), -1)
            cv2.putText(skeleton_img, str(i), (point[0]+5, point[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return skeleton_img

    def identify_challenging_conditions_video(self, frame: np.ndarray, pose_result: Optional[Dict]) -> List[str]:
        """
        Identify challenging conditions in video frame
        """
        conditions = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Lighting conditions
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                conditions.append('low_light')
            elif mean_brightness > 200:
                conditions.append('overexposed')
            
            # Blur detection
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:
                conditions.append('motion_blur')
            
            # Low confidence detection
            if pose_result and 'confidences' in pose_result:
                avg_confidence = np.mean(pose_result['confidences'])
                if avg_confidence < 0.5:
                    conditions.append('low_confidence')
            
            # No detection
            if pose_result is None:
                conditions.append('no_detection')
        
        except Exception as e:
            print(f"Error identifying challenging conditions: {e}")
            conditions.append('analysis_error')
        
        return conditions

    def process_image(self, image_path: str, output_dir: str = './results') -> Dict:
        """
        Process single image for pose estimation
        
        Args:
            image_path: Path to input image file
            output_dir: Output directory for results
            
        Returns:
            Processing results and analysis
        """
        # Validate input file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image file: {image_path}")
        
        height, width = image.shape[:2]
        print(f"Processing image: {image_path}")
        print(f"Image size: {width}x{height}")
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process timing
        start_time = cv2.getTickCount()
        pose_result = self.estimate_pose(image_rgb)
        end_time = cv2.getTickCount()
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        
        # Create annotated image
        annotated_image = image.copy()
        
        results = {
            'pose_result': pose_result,
            'processing_time': processing_time,
            'challenging_conditions': [],
            'summary': {}
        }
        
        if pose_result:
            # Draw pose landmarks
            try:
                image_rgb_copy = image_rgb.copy()
                self.mp_drawing.draw_landmarks(
                    image_rgb_copy,
                    pose_result['pose_landmarks'],
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3, circle_radius=2)
                )
                annotated_image = cv2.cvtColor(image_rgb_copy, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error drawing landmarks: {e}")
            
            # Calculate confidence
            avg_confidence = np.mean(pose_result['confidences'])
            
            # Add text annotations
            cv2.putText(annotated_image, f'Confidence: {avg_confidence:.3f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_image, f'Processing Time: {processing_time:.3f}s', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Create skeleton visualization
            skeleton_image = self.create_skeleton_2d(pose_result['landmarks'], width, height)
            
            # Save results
            output_path = os.path.join(output_dir, 'pose_estimation_output.jpg')
            skeleton_path = os.path.join(output_dir, 'skeleton_2d.jpg')
            
            cv2.imwrite(output_path, annotated_image)
            cv2.imwrite(skeleton_path, skeleton_image)
            
            results['summary'] = {
                'detection_successful': True,
                'avg_confidence': avg_confidence,
                'processing_time': processing_time,
                'output_image_path': output_path,
                'skeleton_path': skeleton_path
            }
        else:
            cv2.putText(annotated_image, 'No Pose Detected', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            output_path = os.path.join(output_dir, 'pose_estimation_output.jpg')
            cv2.imwrite(output_path, annotated_image)
            
            results['summary'] = {
                'detection_successful': False,
                'processing_time': processing_time,
                'output_image_path': output_path
            }
        
        # Identify challenging conditions
        results['challenging_conditions'] = self.identify_challenging_conditions_video(image_rgb, pose_result)
        
        return results

    def process_webcam(self, duration_seconds: int = 10, output_dir: str = './results') -> Dict:
        """
        Process webcam feed for pose estimation
        
        Args:
            duration_seconds: Duration to record in seconds
            output_dir: Output directory for results
            
        Returns:
            Processing results
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)  # Use default camera
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        # Get webcam properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if not available
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Webcam properties: {width}x{height} @ {fps}fps")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, 'webcam_pose_estimation.mp4')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Process frames
        results = {
            'frame_results': [],
            'pose_confidences': [],
            'challenging_conditions': [],
            'frame_numbers': [],
            'processing_times': [],
            'joint_data': []
        }
        
        frame_count = 0
        successful_detections = 0
        total_frames = int(fps * duration_seconds)
        
        print(f"Recording for {duration_seconds} seconds ({total_frames} frames)")
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = cv2.getTickCount()
                
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Estimate pose
                pose_result = self.estimate_pose(frame_rgb)
                
                # Calculate processing time and latency
                end_time = cv2.getTickCount()
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                latency_ms = processing_time * 1000
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                if pose_result:
                    successful_detections += 1
                    
                    # Draw pose landmarks
                    try:
                        frame_rgb_copy = frame_rgb.copy()
                        self.mp_drawing.draw_landmarks(
                            frame_rgb_copy,
                            pose_result['pose_landmarks'],
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        annotated_frame = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"Error drawing landmarks: {e}")
                    
                    avg_confidence = np.mean(pose_result['confidences'])
                    results['pose_confidences'].append(avg_confidence)
                    results['frame_results'].append(pose_result['landmarks'])
                    
                    # Add overlay text
                    cv2.putText(annotated_frame, f'Confidence: {avg_confidence:.3f}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Latency: {latency_ms:.1f}ms', 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}', 
                               (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                else:
                    results['pose_confidences'].append(0.0)
                    results['frame_results'].append(None)
                    cv2.putText(annotated_frame, 'No Pose Detected', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f'Latency: {latency_ms:.1f}ms', 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Store data
                results['frame_numbers'].append(frame_count)
                results['processing_times'].append(processing_time)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                
                # Show progress
                if frame_count % 30 == 0:  # Every second
                    progress = frame_count / total_frames
                    print(f'Progress: {progress:.1%}')
        
        finally:
            cap.release()
            out.release()
        
        # Calculate statistics
        detection_rate = successful_detections / frame_count if frame_count > 0 else 0
        avg_processing_time = np.mean(results['processing_times']) if results['processing_times'] else 0
        avg_latency = avg_processing_time * 1000
        
        results['summary'] = {
            'total_frames': frame_count,
            'successful_detections': successful_detections,
            'detection_rate': detection_rate,
            'avg_processing_time': avg_processing_time,
            'avg_latency_ms': avg_latency,
            'webcam_fps': fps,
            'realtime_capable': avg_processing_time < (1.0 / fps),
            'output_video_path': output_video_path,
            'duration_seconds': duration_seconds
        }
        
        return results

    def process_video(self, video_path: str, output_dir: str = './results', progress_callback=None) -> Dict:
        """
        Process video file for pose estimation
        
        Args:
            video_path: Path to input video file
            output_dir: Output directory for results
            progress_callback: Callback function for progress updates
            
        Returns:
            Processing results and analysis
        """
        # Validate input file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise ValueError("Video file is empty")
        
        print(f"Processing video: {video_path}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, 'frames')
        poses_dir = os.path.join(output_dir, 'pose_frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)
        
        # Open video with error handling
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Try different backends
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            for backend in backends:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    break
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties with validation
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate video properties
        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties: fps={fps}, frames={total_frames}, size={width}x{height}")
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Prepare video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, 'pose_estimation_output.mp4')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError("Cannot create output video writer")
        
        # Process frames
        results = {
            'frame_results': [],
            'pose_confidences': [],
            'challenging_conditions': [],
            'frame_numbers': [],
            'processing_times': [],
            'joint_data': []
        }
        
        frame_count = 0
        successful_detections = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = cv2.getTickCount()
                
                # Validate frame
                if frame is None or frame.size == 0:
                    print(f"Warning: Invalid frame at {frame_count}")
                    frame_count += 1
                    continue
                
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Estimate pose
                pose_result = self.estimate_pose(frame_rgb)
                
                # Calculate processing time and latency
                end_time = cv2.getTickCount()
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                latency_ms = processing_time * 1000  # Convert to milliseconds
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                if pose_result:
                    successful_detections += 1
                    
                    # Draw pose landmarks on frame
                    try:
                        frame_rgb_copy = frame_rgb.copy()
                        self.mp_drawing.draw_landmarks(
                            frame_rgb_copy,
                            pose_result['pose_landmarks'],
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        
                        # Convert back to BGR for video writer
                        annotated_frame = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"Warning: Error drawing landmarks on frame {frame_count}: {e}")
                        annotated_frame = frame.copy()
                    
                    # Calculate average confidence
                    avg_confidence = np.mean(pose_result['confidences'])
                    results['pose_confidences'].append(avg_confidence)
                    
                    # Store pose landmarks
                    results['frame_results'].append(pose_result['landmarks'])
                    
                    # Create skeleton for this frame
                    skeleton_frame = self.create_skeleton_2d(pose_result['landmarks'], width, height)
                    
                    # Store joint data for export
                    joint_data = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'joints': pose_result['landmarks'].tolist(),
                        'confidences': pose_result['confidences'].tolist(),
                        'processing_time': processing_time,
                        'latency_ms': latency_ms
                    }
                    results['joint_data'].append(joint_data)
                    
                    # Add confidence, latency, and processing time text to frame
                    cv2.putText(annotated_frame, f'Confidence: {avg_confidence:.3f}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Latency: {latency_ms:.1f}ms', 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Processing: {processing_time:.3f}s', 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save skeleton frame
                    if frame_count % max(1, total_frames // 50) == 0:  # Save every 2% of frames
                        skeleton_path = os.path.join(poses_dir, f'skeleton_{frame_count:06d}.jpg')
                        cv2.imwrite(skeleton_path, skeleton_frame)

                else:
                    results['pose_confidences'].append(0.0)
                    results['frame_results'].append(None)
                    results['joint_data'].append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'joints': None,
                        'confidences': None,
                        'processing_time': processing_time,
                        'latency_ms': latency_ms
                    })
                    
                    # Add "No Pose Detected" text with latency info
                    cv2.putText(annotated_frame, 'No Pose Detected', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f'Latency: {latency_ms:.1f}ms', 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Identify challenging conditions
                challenging_conditions = self.identify_challenging_conditions_video(frame_rgb, pose_result)
                results['challenging_conditions'].append(challenging_conditions)
                
                # Add frame info
                results['frame_numbers'].append(frame_count)
                results['processing_times'].append(processing_time)
                
                # Add frame number to frame
                cv2.putText(annotated_frame, f'Frame: {frame_count}', 
                        (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Save sample frames
                if frame_count % max(1, total_frames // 20) == 0:  # Save every 5% of frames
                    try:
                        frame_path = os.path.join(frames_dir, f'frame_{frame_count:06d}.jpg')
                        pose_path = os.path.join(poses_dir, f'pose_{frame_count:06d}.jpg')
                        cv2.imwrite(frame_path, frame)
                        cv2.imwrite(pose_path, annotated_frame)
                    except Exception as e:
                        print(f"Warning: Could not save frame {frame_count}: {e}")
                
                frame_count += 1
                
                # Update progress
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        
        finally:
            # Clean up
            cap.release()
            out.release()
        
        # Calculate final statistics
        detection_rate = successful_detections / frame_count if frame_count > 0 else 0
        avg_processing_time = np.mean(results['processing_times']) if results['processing_times'] else 0
        video_duration = frame_count / fps if fps > 0 else 0
        
        results['summary'] = {
            'total_frames': frame_count,
            'successful_detections': successful_detections,
            'detection_rate': detection_rate,
            'avg_processing_time': avg_processing_time,
            'realtime_capable': avg_processing_time < (1.0 / fps),
            'output_video_path': output_video_path,
            'video_duration': video_duration,
            'video_fps': fps,
            'video_resolution': f"{width}x{height}"
        }
        
        return results

    def export_joint_data(self, results: Dict, output_dir: str):
        """Export joint data in various formats"""
        
        try:
            # JSON export
            json_path = os.path.join(output_dir, 'joint_data.json')
            with open(json_path, 'w') as f:
                json.dump(results['joint_data'], f, indent=2)
            
            # CSV export
            csv_path = os.path.join(output_dir, 'joint_data.csv')
            csv_data = []
            
            for frame_data in results['joint_data']:
                if frame_data['joints'] is not None:
                    for i, (joint, confidence) in enumerate(zip(frame_data['joints'], frame_data['confidences'])):
                        csv_data.append({
                            'frame': frame_data['frame'],
                            'timestamp': frame_data['timestamp'],
                            'joint_id': i,
                            'x': joint[0],
                            'y': joint[1],
                            'z': joint[2],
                            'confidence': confidence
                        })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
            
            return json_path, csv_path
        
        except Exception as e:
            print(f"Error exporting joint data: {e}")
            return None, None

def create_analysis_plots(results: Dict, output_dir: str):
    """Create analysis plots"""
    
    try:
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Processing time over frames
        if results['processing_times']:
            axes[0, 0].plot(results['frame_numbers'], results['processing_times'])
            axes[0, 0].set_title('Processing Time per Frame')
            axes[0, 0].set_xlabel('Frame Number')
            axes[0, 0].set_ylabel('Processing Time (seconds)')
            axes[0, 0].grid(True)
        
        # Confidence scores over frames
        if results['pose_confidences']:
            confidences = [c if c > 0 else None for c in results['pose_confidences']]
            axes[0, 1].plot(results['frame_numbers'], confidences, 'g-', alpha=0.7)
            axes[0, 1].set_title('Pose Detection Confidence')
            axes[0, 1].set_xlabel('Frame Number')
            axes[0, 1].set_ylabel('Confidence Score')
            axes[0, 1].grid(True)
        
        # Detection success rate (rolling average)
        if results['pose_confidences']:
            window_size = max(1, len(results['pose_confidences']) // 20)
            detection_success = [1 if c > 0 else 0 for c in results['pose_confidences']]
            rolling_avg = pd.Series(detection_success).rolling(window=window_size).mean()
            
            axes[1, 0].plot(results['frame_numbers'], rolling_avg)
            axes[1, 0].set_title(f'Detection Success Rate (Rolling Avg, window={window_size})')
            axes[1, 0].set_xlabel('Frame Number')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].grid(True)
        
        # Challenging conditions distribution
        all_conditions = []
        for conditions in results['challenging_conditions']:
            all_conditions.extend(conditions)
        
        if all_conditions:
            condition_counts = {}
            for condition in all_conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            conditions, counts = zip(*sorted(condition_counts.items()))
            axes[1, 1].bar(conditions, counts)
            axes[1, 1].set_title('Challenging Conditions Distribution')
            axes[1, 1].set_xlabel('Condition Type')
            axes[1, 1].set_ylabel('Number of Frames')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            axes[1, 1].text(0.5, 0.5, 'No challenging\nconditions detected', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Challenging Conditions Distribution')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'analysis_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating analysis plots: {e}")
        return None

def process_video_gradio(video_file, model_complexity, min_detection_confidence, min_tracking_confidence, progress=gr.Progress()):
    """
    Process video through Gradio interface
    """
    if video_file is None:
        return None, None, "Please upload a video file.", None, None
    
    # Validate file path
    if not isinstance(video_file, str):
        return None, None, "Invalid file format received.", None, None
    
    if not os.path.exists(video_file):
        return None, None, f"Video file not found: {video_file}", None, None
    
    try:
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, 'results')
        
        print(f"Processing file: {video_file}")
        print(f"File exists: {os.path.exists(video_file)}")
        print(f"File size: {os.path.getsize(video_file)} bytes")
        
        # Copy the uploaded file to temp directory to avoid permission issues
        temp_video_path = os.path.join(temp_dir, 'input_video.mp4')
        shutil.copy2(video_file, temp_video_path)
        
        # Initialize estimator
        estimator = MPIPoseEstimator(
            model_complexity=int(model_complexity),
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Process video with progress tracking
        def progress_callback(prog, message):
            progress(prog, desc=message)
        
        progress(0.1, desc="Starting video processing...")
        results = estimator.process_video(temp_video_path, output_dir, progress_callback)
        
        progress(0.8, desc="Generating analysis plots...")
        # Generate analysis plots
        plot_path = create_analysis_plots(results, output_dir)
        
        progress(0.9, desc="Exporting joint data...")
        # Export joint data
        json_path, csv_path = estimator.export_joint_data(results, output_dir)
        
        # Create summary report
        summary = results['summary']
        report = f"""
# Pose Estimation Results

## Video Information
- **Duration:** {summary['video_duration']:.2f} seconds
- **Resolution:** {summary['video_resolution']}
- **FPS:** {summary['video_fps']:.1f}
- **Total Frames:** {summary['total_frames']}

## Detection Performance
- **Successful Detections:** {summary['successful_detections']} frames
- **Detection Rate:** {summary['detection_rate']:.2%}
- **Average Processing Time:** {summary['avg_processing_time']:.3f}s per frame
- **Real-time Capable:** {'Yes' if summary['realtime_capable'] else 'No'}

## Challenging Conditions
"""
        
        # Add challenging conditions summary
        all_conditions = []
        for conditions in results['challenging_conditions']:
            all_conditions.extend(conditions)
        
        if all_conditions:
            condition_counts = {}
            for condition in all_conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            for condition, count in sorted(condition_counts.items()):
                percentage = (count / summary['total_frames']) * 100
                report += f"- **{condition.replace('_', ' ').title()}:** {count} frames ({percentage:.1f}%)\n"
        else:
            report += "- No challenging conditions detected\n"
        
        progress(1.0, desc="Processing complete!")
        
        # Create downloadable zip file
        zip_path = os.path.join(temp_dir, 'pose_estimation_results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add output video
            if os.path.exists(summary['output_video_path']):
                zipf.write(summary['output_video_path'], 'pose_estimation_output.mp4')
            
            # Add joint data files
            if json_path and os.path.exists(json_path):
                zipf.write(json_path, 'joint_data.json')
            if csv_path and os.path.exists(csv_path):
                zipf.write(csv_path, 'joint_data.csv')
            
            # Add analysis plots
            if plot_path and os.path.exists(plot_path):
                zipf.write(plot_path, 'analysis_plots.png')
        
        return (
            summary['output_video_path'],  # Output video
            plot_path,                     # Analysis plots
            report,                        # Summary report
            zip_path,                      # Download zip
            json_path                      # Joint data JSON
        )
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, error_msg, None, None
def process_image_gradio(image_file, model_complexity, min_detection_confidence, min_tracking_confidence):
    """Process image through Gradio interface"""
    if image_file is None:
        return None, None, "Please upload an image file.", None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, 'results')
        
        # Initialize estimator
        estimator = MPIPoseEstimator(
            model_complexity=int(model_complexity),
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Process image
        results = estimator.process_image(image_file, output_dir)
        
        # Create report
        summary = results['summary']
        if summary['detection_successful']:
            report = f"""
# Pose Estimation Results - Image

## Detection Results
- **Pose Detected:** Yes
- **Confidence:** {summary['avg_confidence']:.3f}
- **Processing Time:** {summary['processing_time']:.3f} seconds

## Challenging Conditions
"""
            if results['challenging_conditions']:
                for condition in results['challenging_conditions']:
                    report += f"- {condition.replace('_', ' ').title()}\n"
            else:
                report += "- No challenging conditions detected\n"
        else:
            report = f"""
# Pose Estimation Results - Image

## Detection Results
- **Pose Detected:** No
- **Processing Time:** {summary['processing_time']:.3f} seconds

Possible reasons:
- No person visible in image
- Person too small or partially occluded
- Poor lighting conditions
- Low image quality
"""
        
        return (
            summary['output_image_path'],
            summary.get('skeleton_path'),
            report,
            summary['output_image_path']
        )
        
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}", None

# ==== MODIFICATION 6: Add webcam processing to Gradio interface ====

def process_webcam_gradio(duration, model_complexity, min_detection_confidence, min_tracking_confidence, progress=gr.Progress()):
    """Process webcam feed through Gradio interface"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, 'results')
        
        # Initialize estimator
        estimator = MPIPoseEstimator(
            model_complexity=int(model_complexity),
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        progress(0.1, desc="Starting webcam capture...")
        
        # Process webcam
        results = estimator.process_webcam(int(duration), output_dir)
        
        progress(0.8, desc="Generating report...")
        
        # Create report
        summary = results['summary']
        report = f"""
# Webcam Pose Estimation Results

## Recording Information
- **Duration:** {summary['duration_seconds']} seconds
- **Total Frames:** {summary['total_frames']}
- **Webcam FPS:** {summary['webcam_fps']:.1f}

## Detection Performance
- **Successful Detections:** {summary['successful_detections']} frames
- **Detection Rate:** {summary['detection_rate']:.2%}
- **Average Processing Time:** {summary['avg_processing_time']:.3f}s per frame
- **Average Latency:** {summary['avg_latency_ms']:.1f}ms
- **Real-time Capable:** {'Yes' if summary['realtime_capable'] else 'No'}
"""
        
        progress(1.0, desc="Complete!")
        
        return (
            summary['output_video_path'],
            report,
            summary['output_video_path']
        )
        
    except Exception as e:
        return None, f"Error processing webcam: {str(e)}", None
    
# Create Gradio interface
def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="MediaPipe Pose Estimation", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎯 MediaPipe Pose Estimation
        
        Upload a video file to perform real-time pose estimation and joint tracking.
        The system will detect human poses and export joint coordinates and analysis.
        
        **Supported formats:** MP4, AVI, MOV, MKV, WMV
        """)
        
        with gr.Tabs():
            with gr.Tab("📹 Video Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload & Settings")
                        
                        # Upload status and timer
                        with gr.Row():
                            upload_status = gr.Textbox(
                                label="Upload Status",
                                value="No file selected",
                                interactive=False,
                                elem_id="upload_status"
                            )
                            upload_timer = gr.Textbox(
                                label="Upload Time",
                                value="00:00",
                                interactive=False,
                                elem_id="upload_timer"
                            )
                        
                        # File upload with progress tracking
                        video_input = gr.File(
                            label="Upload Video File",
                            file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv"],
                            type="filepath"
                        )
                        
                        # File information display
                        file_info = gr.Textbox(
                            label="File Information",
                            value="",
                            interactive=False,
                            visible=False
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            model_complexity = gr.Slider(
                                minimum=0,
                                maximum=2,
                                value=1,
                                step=1,
                                label="Model Complexity (0=Fast, 2=Accurate)",
                                info="Higher values are more accurate but slower"
                            )
                            
                            min_detection_confidence = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                label="Minimum Detection Confidence",
                                info="Minimum confidence for pose detection"
                            )
                            
                            min_tracking_confidence = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                label="Minimum Tracking Confidence",
                                info="Minimum confidence for pose tracking"
                            )
                        
                        process_btn = gr.Button(
                            "🚀 Process Video",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        
                        with gr.Tabs():
                            with gr.Tab("Output Video"):
                                output_video = gr.Video(
                                    label="Pose Estimation Output"
                                )
                            
                            with gr.Tab("Analysis"):
                                analysis_plots = gr.Image(
                                    label="Performance Analysis"
                                )
                                
                                summary_report = gr.Markdown(
                                    label="Summary Report"
                                )
                            
                            with gr.Tab("Downloads"):
                                gr.Markdown("### Download Results")
                                
                                download_zip = gr.File(
                                    label="Complete Results Package"
                                )
                                
                                joint_data_json = gr.File(
                                    label="Joint Data (JSON)"
                                )
            
            with gr.Tab("📷 Image Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Image")
                        
                        image_input = gr.Image(
                            label="Upload Image File",
                            type="filepath"
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            img_model_complexity = gr.Slider(
                                minimum=0, maximum=2, value=1, step=1,
                                label="Model Complexity"
                            )
                            img_min_detection_confidence = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.5,
                                label="Minimum Detection Confidence"
                            )
                            img_min_tracking_confidence = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.5,
                                label="Minimum Tracking Confidence"
                            )
                        
                        process_image_btn = gr.Button("🚀 Process Image", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        
                        with gr.Tabs():
                            with gr.Tab("Output Image"):
                                output_image = gr.Image(label="Pose Estimation Output")
                            
                            with gr.Tab("Skeleton View"):
                                skeleton_image = gr.Image(label="2D Skeleton Visualization")
                            
                            with gr.Tab("Analysis"):
                                image_report = gr.Markdown(label="Analysis Report")
                        
                        image_download = gr.File(label="Download Result")
            
            with gr.Tab("📸 Webcam Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Webcam Settings")
                        
                        duration_slider = gr.Slider(
                            minimum=5, maximum=60, value=10, step=1,
                            label="Recording Duration (seconds)"
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            cam_model_complexity = gr.Slider(
                                minimum=0, maximum=2, value=1, step=1,
                                label="Model Complexity"
                            )
                            cam_min_detection_confidence = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.5,
                                label="Minimum Detection Confidence"
                            )
                            cam_min_tracking_confidence = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.5,
                                label="Minimum Tracking Confidence"
                            )
                        
                        process_webcam_btn = gr.Button("🎥 Start Webcam Recording", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        
                        webcam_output_video = gr.Video(label="Webcam Pose Estimation")
                        webcam_report = gr.Markdown(label="Analysis Report")
                        webcam_download = gr.File(label="Download Result")
        
        # Event handlers
        def update_upload_status(file_path):
            """Update upload status and file information"""
            import time
            if file_path is None:
                return "No file selected", "00:00", "", False
            
            try:
                # Get file information
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                
                # Convert file size to readable format
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                elif file_size < 1024 * 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{file_size / (1024 * 1024 * 1024):.1f} GB"
                
                # Get video information using OpenCV
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    file_info_text = f"""📹 **{file_name}**
📏 Size: {size_str}
🎞️ Resolution: {width}x{height}
⏱️ Duration: {duration:.1f}s
🎬 FPS: {fps:.1f}
📊 Frames: {frame_count}"""
                else:
                    file_info_text = f"""📹 **{file_name}**
📏 Size: {size_str}
⚠️ Could not read video properties"""
                
                status = "✅ Upload Complete - Video Ready"
                timer = "Upload Complete"
                
                return status, timer, file_info_text, True
                
            except Exception as e:
                return f"❌ Error reading file: {str(e)}", "Error", "", False
        
        # File upload change handler
        video_input.change(
            fn=update_upload_status,
            inputs=[video_input],
            outputs=[upload_status, upload_timer, file_info, file_info]
        )
        
        # Event bindings
        process_btn.click(
            fn=process_video_gradio,
            inputs=[
                video_input,
                model_complexity,
                min_detection_confidence,
                min_tracking_confidence
            ],
            outputs=[
                output_video,
                analysis_plots,
                summary_report,
                download_zip,
                joint_data_json
            ],
            show_progress=True
        )
        
        process_image_btn.click(
            fn=process_image_gradio,
            inputs=[image_input, img_model_complexity, img_min_detection_confidence, img_min_tracking_confidence],
            outputs=[output_image, skeleton_image, image_report, image_download]
        )

        process_webcam_btn.click(
            fn=process_webcam_gradio,
            inputs=[duration_slider, cam_model_complexity, cam_min_detection_confidence, cam_min_tracking_confidence],
            outputs=[webcam_output_video, webcam_report, webcam_download],
            show_progress=True
        )
        
        # Add JavaScript for real-time upload monitoring
        interface.load(
            None,
            None,
            None,
            js="""
            function() {
                // Upload monitoring script
                let uploadStartTime = null;
                let uploadTimer = null;
                
                function formatTime(seconds) {
                    const mins = Math.floor(seconds / 60);
                    const secs = Math.floor(seconds % 60);
                    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                }
                
                function startUploadTimer() {
                    uploadStartTime = Date.now();
                    uploadTimer = setInterval(() => {
                        const elapsed = Math.floor((Date.now() - uploadStartTime) / 1000);
                        const timerElement = document.querySelector('#upload_timer input');
                        const statusElement = document.querySelector('#upload_status input');
                        
                        if (timerElement) {
                            timerElement.value = formatTime(elapsed);
                        }
                        if (statusElement && statusElement.value === 'No file selected') {
                            statusElement.value = '⏳ Uploading... Please wait';
                        }
                    }, 1000);
                }
                
                function stopUploadTimer() {
                    if (uploadTimer) {
                        clearInterval(uploadTimer);
                        uploadTimer = null;
                    }
                }
                
                // Monitor file input changes
                const fileInput = document.querySelector('input[type="file"]');
                if (fileInput) {
                    fileInput.addEventListener('change', function(e) {
                        if (e.target.files.length > 0) {
                            startUploadTimer();
                            const statusElement = document.querySelector('#upload_status input');
                            if (statusElement) {
                                statusElement.value = '📤 Starting upload...';
                            }
                        }
                    });
                }
                
                // Monitor for upload completion
                const observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'childList' || mutation.type === 'characterData') {
                            const statusElement = document.querySelector('#upload_status input');
                            if (statusElement && statusElement.value.includes('✅')) {
                                stopUploadTimer();
                            }
                        }
                    });
                });
                
                // Start observing
                const targetNode = document.body;
                observer.observe(targetNode, {
                    childList: true,
                    subtree: true,
                    characterData: true
                });
                
                return [];
            }
            """
        )
        
        gr.Markdown("""
        ### 📝 Usage Instructions
        
        1. **Upload Video**: Choose a video file containing human subjects (max 200MB recommended)
        2. **Adjust Settings**: Modify model complexity and confidence thresholds if needed
        3. **Process**: Click the "Process Video" button to start pose estimation
        4. **Review Results**: Check the output video, analysis plots, and summary
        5. **Download**: Get the complete results package or individual files
        
        ### 📊 Output Files
        
        - **Output Video**: Video with pose landmarks overlaid
        - **Joint Data (JSON/CSV)**: Raw joint coordinates and confidence scores
        - **Analysis Plots**: Performance metrics and challenging conditions
        - **Complete Package**: ZIP file with all results
        
        ### ⚙️ Model Settings
        
        - **Model Complexity 0**: Fastest, basic accuracy
        - **Model Complexity 1**: Balanced speed and accuracy (recommended)
        - **Model Complexity 2**: Most accurate, slower processing
        
        ### 🔧 Troubleshooting
        
        - **Upload Status**: Monitor the upload status and timer above
        - Wait for "✅ Upload Complete" before processing
        - Ensure video file is not corrupted
        - Try different video formats if upload fails
        - Reduce video resolution/duration for faster processing
        - Check that the video contains visible human subjects
        - Large files may take longer to upload - be patient!
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True  # Enable debug mode for better error messages
    )