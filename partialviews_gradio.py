import cv2
import numpy as np
import gradio as gr
import tempfile
import os
from typing import Tuple, Optional
import threading
import time

class VideoPartialViewProcessor:
    def __init__(self):
        self.processing = False
        
    def apply_partial_view(self, frame: np.ndarray, view_type: str, intensity: float = 0.5, **kwargs) -> np.ndarray:
        """Apply different types of partial views to the frame"""
        h, w = frame.shape[:2]
        
        if view_type == "Top Half":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:int(h * intensity), :] = 255
            
        elif view_type == "Bottom Half":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(h * (1 - intensity)):, :] = 255
            
        elif view_type == "Left Half":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:, :int(w * intensity)] = 255
            
        elif view_type == "Right Half":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:, int(w * (1 - intensity)):] = 255
            
        elif view_type == "Center Circle":
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = int(min(w, h) * intensity * 0.5)
            cv2.circle(mask, center, radius, 255, -1)
            
        elif view_type == "Center Rectangle":
            mask = np.zeros((h, w), dtype=np.uint8)
            margin_h = int(h * (1 - intensity) / 2)
            margin_w = int(w * (1 - intensity) / 2)
            mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
            
        elif view_type == "Random Patches":
            mask = np.zeros((h, w), dtype=np.uint8)
            num_patches = int(30 * intensity)
            patch_size = kwargs.get('patch_size', 50)
            for _ in range(num_patches):
                x = np.random.randint(0, max(1, w - patch_size))
                y = np.random.randint(0, max(1, h - patch_size))
                patch_w = np.random.randint(20, patch_size)
                patch_h = np.random.randint(20, patch_size)
                mask[y:y+patch_h, x:x+patch_w] = 255
                
        elif view_type == "Horizontal Stripes":
            mask = np.zeros((h, w), dtype=np.uint8)
            stripe_height = max(1, int(h * 0.05 * intensity))
            gap = max(1, int(stripe_height * (2 - intensity)))
            for i in range(0, h, stripe_height + gap):
                mask[i:i+stripe_height, :] = 255
                
        elif view_type == "Vertical Stripes":
            mask = np.zeros((h, w), dtype=np.uint8)
            stripe_width = max(1, int(w * 0.05 * intensity))
            gap = max(1, int(stripe_width * (2 - intensity)))
            for i in range(0, w, stripe_width + gap):
                mask[:, i:i+stripe_width] = 255
                
        elif view_type == "Diagonal Split":
            mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if (i + j) / (h + w) < intensity:
                        mask[i, j] = 255
                        
        elif view_type == "Corner Quad":
            mask = np.zeros((h, w), dtype=np.uint8)
            mid_h, mid_w = h // 2, w // 2
            if intensity > 0.75:  # All quadrants
                mask[:, :] = 255
            elif intensity > 0.5:  # Top quadrants
                mask[:mid_h, :] = 255
            elif intensity > 0.25:  # Top-left quadrant
                mask[:mid_h, :mid_w] = 255
            else:  # Small top-left corner
                corner_h = int(mid_h * intensity * 4)
                corner_w = int(mid_w * intensity * 4)
                mask[:corner_h, :corner_w] = 255
                
        elif view_type == "Radial Sectors":
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            num_sectors = max(1, int(8 * intensity))
            for i in range(h):
                for j in range(w):
                    angle = np.arctan2(i - center[1], j - center[0])
                    sector = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                    if sector < num_sectors:
                        mask[i, j] = 255
                        
        else:  # "None" or default
            return frame
        
        # Apply mask with black background
        result = frame.copy()
        result[mask == 0] = [0, 0, 0]  # Black out non-visible areas
        return result
    
    def process_video(self, input_video_path: str, view_type: str, intensity: float, 
                     patch_size: int = 50, progress_callback=None) -> str:
        """Process entire video and save filtered version"""
        if not input_video_path or not os.path.exists(input_video_path):
            raise ValueError("Invalid input video path")
            
        self.processing = True
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply partial view filter
                filtered_frame = self.apply_partial_view(
                    frame, view_type, intensity, patch_size=patch_size
                )
                
                # Write frame to output video
                out.write(filtered_frame)
                
                frame_count += 1
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress)
                    
                if not self.processing:  # Check for cancellation
                    break
                    
        finally:
            # Release everything
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            self.processing = False
        
        return output_path
    
    def preview_frame(self, input_video_path: str, view_type: str, intensity: float, 
                     patch_size: int = 50, frame_number: int = 0) -> np.ndarray:
        """Generate preview of a single frame with partial view applied"""
        if not input_video_path or not os.path.exists(input_video_path):
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set frame position
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            frame_pos = min(frame_number, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Apply partial view filter
        filtered_frame = self.apply_partial_view(
            frame, view_type, intensity, patch_size=patch_size
        )
        
        # Convert BGR to RGB for display
        return cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)

# Initialize processor
processor = VideoPartialViewProcessor()

def process_video_wrapper(input_video, view_type, intensity, patch_size, progress=gr.Progress()):
    """Wrapper for video processing with progress updates"""
    def progress_callback(prog):
        progress(prog, desc=f"Processing video... {int(prog*100)}%")
    
    try:
        output_path = processor.process_video(
            input_video, view_type, intensity, patch_size, progress_callback
        )
        return output_path, "✅ Video processing completed successfully!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def preview_wrapper(input_video, view_type, intensity, patch_size, frame_number):
    """Wrapper for frame preview"""
    try:
        preview_frame = processor.preview_frame(
            input_video, view_type, intensity, patch_size, frame_number
        )
        return preview_frame
    except Exception as e:
        return np.zeros((480, 640, 3), dtype=np.uint8)

def get_video_info(input_video):
    """Get basic video information"""
    if not input_video or not os.path.exists(input_video):
        return "No video selected"
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return "Invalid video file"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return f"📹 Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames} | Duration: {duration:.2f}s"

# Create Gradio interface
with gr.Blocks(title="Video Partial View Filter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Video Partial View Filter
    
    Apply partial view filters to videos for pose estimation testing.
    Perfect for testing robustness of pose estimation models with occluded or partially visible subjects.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📁 Input Video")
            input_video = gr.File(
                label="Upload Video File",
                file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv"],
                type="filepath"
            )
            video_info = gr.Textbox(
                label="Video Information",
                interactive=False,
                value="No video selected"
            )
            
            # Filter settings
            gr.Markdown("### ⚙️ Filter Settings")
            view_type = gr.Dropdown(
                choices=[
                    "None",
                    "Top Half",
                    "Bottom Half", 
                    "Left Half",
                    "Right Half",
                    "Center Circle",
                    "Center Rectangle",
                    "Random Patches",
                    "Horizontal Stripes",
                    "Vertical Stripes",
                    "Diagonal Split",
                    "Corner Quad",
                    "Radial Sectors"
                ],
                value="Top Half",
                label="Partial View Type"
            )
            
            intensity = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Filter Intensity"
            )
            
            patch_size = gr.Slider(
                minimum=20,
                maximum=100,
                value=50,
                step=5,
                label="Patch Size (for Random Patches)",
                visible=False
            )
            
        with gr.Column(scale=2):
            # Preview section
            gr.Markdown("### 👁️ Preview")
            frame_number = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                label="Preview Frame Number"
            )
            
            preview_image = gr.Image(
                label="Filtered Preview",
                height=400
            )
            
            # Process section
            gr.Markdown("### 🚀 Process Video")
            process_btn = gr.Button(
                "Process Video",
                variant="primary",
                size="lg"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to process video"
            )
            
            output_video = gr.File(
                label="Download Processed Video",
                interactive=False
            )
    
    # Event handlers
    input_video.change(
        fn=get_video_info,
        inputs=[input_video],
        outputs=[video_info]
    )
    
    # Show/hide patch size slider based on view type
    def update_patch_visibility(view_type):
        return gr.update(visible=(view_type == "Random Patches"))
    
    view_type.change(
        fn=update_patch_visibility,
        inputs=[view_type],
        outputs=[patch_size]
    )
    
    # Update preview when parameters change
    preview_inputs = [input_video, view_type, intensity, patch_size, frame_number]
    
    for input_component in preview_inputs:
        input_component.change(
            fn=preview_wrapper,
            inputs=preview_inputs,
            outputs=[preview_image]
        )
    
    # Process video
    process_btn.click(
        fn=process_video_wrapper,
        inputs=[input_video, view_type, intensity, patch_size],
        outputs=[output_video, status_text]
    )
    
    # Examples
    gr.Markdown("""
    ### 📝 Usage Tips:
    - **MPI-INF-3DHP Dataset**: Upload videos from your dataset for testing
    - **Filter Types**: Experiment with different partial views to simulate occlusion scenarios
    - **Intensity**: Controls how much of the frame is visible (0.1 = minimal, 1.0 = maximum)
    - **Preview**: Use the frame slider to preview effects before processing the entire video
    - **Output**: Processed videos are saved in MP4 format for compatibility
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )