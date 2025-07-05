import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

def create_motion_blur_kernel(size, angle):
    """Create a motion blur kernel based on size and angle"""
    kernel = np.zeros((size, size))
    
    # Calculate the center of the kernel
    center = size // 2
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Create line kernel
    for i in range(size):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        
        # Ensure coordinates are within bounds
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
    
    return kernel

def apply_motion_blur_frame(frame, blur_strength, blur_angle):
    """Apply motion blur to a single frame"""
    if blur_strength <= 0:
        return frame
    
    # Create motion blur kernel
    kernel_size = max(3, int(blur_strength))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    kernel = create_motion_blur_kernel(kernel_size, blur_angle)
    
    # Apply the motion blur filter
    blurred_frame = cv2.filter2D(frame, -1, kernel)
    
    return blurred_frame

def process_video_with_motion_blur(input_video_path, blur_strength, blur_angle, progress=gr.Progress()):
    """Process video and apply motion blur to all frames"""
    if input_video_path is None:
        return None, "Please upload a video file"
    
    try:
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file. Make sure it's a valid video format."
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get original file extension and create output filename
        input_path = Path(input_video_path)
        original_extension = input_path.suffix.lower()
        
        # Create temporary output file with appropriate extension
        temp_dir = tempfile.mkdtemp()
        
        # For better compatibility, we'll use .mp4 for output regardless of input format
        output_path = os.path.join(temp_dir, f"motion_blurred_video.mp4")
        
        # Define codec - try different codecs for better compatibility
        codecs_to_try = ['mp4v', 'XVID', 'MJPG', 'X264']
        out = None
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
                out.release()
            except:
                continue
        
        if out is None or not out.isOpened():
            # Fallback: try without specifying codec
            out = cv2.VideoWriter(output_path, -1, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return None, "Error: Could not create output video file"
        
        frame_count = 0
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply motion blur to the frame
            blurred_frame = apply_motion_blur_frame(frame, blur_strength, blur_angle)
            
            # Write the frame
            out.write(blurred_frame)
            
            frame_count += 1
            
            # Update progress
            if total_frames > 0:
                progress_percent = (frame_count / total_frames)
                progress(progress_percent, desc=f"Processing frame {frame_count}/{total_frames}")
        
        # Release everything
        cap.release()
        out.release()
        
        return output_path, f"Successfully processed {frame_count} frames with motion blur (Original: {original_extension.upper()}, Output: MP4)"
        
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

def create_sample_preview(input_video_path, blur_strength, blur_angle):
    """Create a preview of the first frame with motion blur applied"""
    if input_video_path is None:
        return None
    
    try:
        cap = cv2.VideoCapture(input_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Apply motion blur to the frame
        blurred_frame = apply_motion_blur_frame(frame, blur_strength, blur_angle)
        
        # Convert BGR to RGB for display
        preview_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
        
        return preview_frame
        
    except Exception as e:
        print(f"Preview error: {str(e)}")
        return None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Video Motion Blur Filter", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎬 Video Motion Blur Filter
        
        Upload a video and apply motion blur effects to test your models' robustness against motion artifacts.
        
        **Supported Formats:** MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP
        
        **Features:**
        - Adjustable blur strength (intensity)
        - Configurable blur angle (direction)
        - Real-time preview of first frame
        - Full video processing with progress tracking
        - Multi-format input support with MP4 output
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Settings")
                
                video_input = gr.File(
                    label="Upload Video",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".3gp"],
                    type="filepath"
                )
                
                with gr.Group():
                    gr.Markdown("#### Motion Blur Parameters")
                    
                    blur_strength = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Blur Strength",
                        info="Higher values create stronger motion blur effect"
                    )
                    
                    blur_angle = gr.Slider(
                        minimum=0,
                        maximum=360,
                        value=0,
                        step=1,
                        label="Blur Angle (degrees)",
                        info="Direction of motion blur (0° = horizontal right)"
                    )
                
                with gr.Row():
                    preview_btn = gr.Button("🔍 Preview First Frame", variant="secondary")
                    process_btn = gr.Button("🎯 Process Full Video", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                
                preview_output = gr.Image(
                    label="Preview (First Frame)",
                    type="numpy"
                )
                
                video_output = gr.Video(
                    label="Processed Video"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        # Create angle direction reference
        with gr.Row():
            gr.Markdown("""
            ### 🧭 Blur Angle Reference
            - **0°**: Horizontal blur (left ↔ right)
            - **90°**: Vertical blur (up ↕ down)  
            - **45°**: Diagonal blur (↗ direction)
            - **135°**: Diagonal blur (↖ direction)
            """)
        
        # Event handlers
        preview_btn.click(
            fn=create_sample_preview,
            inputs=[video_input, blur_strength, blur_angle],
            outputs=preview_output
        )
        
        process_btn.click(
            fn=process_video_with_motion_blur,
            inputs=[video_input, blur_strength, blur_angle],
            outputs=[video_output, status_output]
        )
        
        # Auto-update preview when parameters change
        for component in [blur_strength, blur_angle]:
            component.change(
                fn=create_sample_preview,
                inputs=[video_input, blur_strength, blur_angle],
                outputs=preview_output
            )
        
        video_input.change(
            fn=create_sample_preview,
            inputs=[video_input, blur_strength, blur_angle],
            outputs=preview_output
        )
    
    return interface

# Launch the application
if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    print("🚀 Starting Video Motion Blur Filter...")
    print("📁 Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP")
    print("📤 Output format: MP4 (for maximum compatibility)")
    print("🔍 Use 'Preview First Frame' to see the effect")
    print("🎯 Use 'Process Full Video' to apply blur to entire video")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        show_error=True
    )