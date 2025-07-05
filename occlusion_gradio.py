import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse
import gradio as gr
import tempfile
import shutil


class VideoOccluder:
    """Apply different types of occlusions to videos.

    The only functional change from the previous version is that occlusion
    objects now update more slowly so they do not appear to "flicker" every
    frame.  This is achieved by reseeding Python's RNG with a value derived
    from the *segment* index rather than the absolute frame index.  All
    occlusion shapes remain fully random, but they persist for
    ``refresh_rate`` consecutive frames (default = 5) before new shapes are
    generated.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, refresh_rate: int = 5):
        """Create a new :class:`VideoOccluder`.

        Parameters
        ----------
        refresh_rate
            Number of consecutive frames for which the same occlusion pattern
            should be reused.  A value of 1 will behave like the original fast
            version (new occlusions every frame).  A value between 5 and 10
            usually gives a pleasant, slow‑moving effect.
        """
        # How many frames the same occlusion pattern should stay on screen
        self.refresh_rate = max(1, int(refresh_rate))

        # Accepted video formats
        self.supported_formats = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".flv",
            ".wmv",
            ".m4v",
            ".3gp",
            ".webm",
            ".ogv",
            ".ts",
            ".mts",
        ]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def validate_video_format(self, video_path):
        """Return *True* if *video_path* has a supported extension."""
        file_ext = Path(video_path).suffix.lower()
        return file_ext in self.supported_formats

    def get_video_info(self, video_path):
        """Return width / height / fps / frame‑count / duration for *video_path*."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            / cap.get(cv2.CAP_PROP_FPS),
        }
        cap.release()
        return info

    # ------------------------------------------------------------------
    # Occlusion primitives (unchanged)
    # ------------------------------------------------------------------

    def create_random_rectangle_occlusion(self, frame, intensity=0.3):
        h, w = frame.shape[:2]
        occluded_frame = frame.copy()

        num_occlusions = int(intensity * 10)
        for _ in range(num_occlusions):
            rect_w = random.randint(int(w * 0.05), int(w * 0.15))
            rect_h = random.randint(int(h * 0.05), int(h * 0.15))
            x = random.randint(0, max(1, w - rect_w))
            y = random.randint(0, max(1, h - rect_h))

            color_type = random.choice(["black", "white", "random"])
            if color_type == "black":
                color = (0, 0, 0)
            elif color_type == "white":
                color = (255, 255, 255)
            else:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

            cv2.rectangle(occluded_frame, (x, y), (x + rect_w, y + rect_h), color, -1)

        return occluded_frame

    def create_circular_occlusion(self, frame, intensity=0.3):
        h, w = frame.shape[:2]
        occluded_frame = frame.copy()

        num_occlusions = int(intensity * 8)
        for _ in range(num_occlusions):
            radius = random.randint(int(w * 0.03), int(w * 0.10))
            center_x = random.randint(radius, w - radius)
            center_y = random.randint(radius, h - radius)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.circle(occluded_frame, (center_x, center_y), radius, color, -1)

        return occluded_frame

    def create_line_occlusion(self, frame, intensity=0.3):
        h, w = frame.shape[:2]
        occluded_frame = frame.copy()

        num_occlusions = int(intensity * 15)
        for _ in range(num_occlusions):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            thickness = random.randint(3, 8)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.line(occluded_frame, (x1, y1), (x2, y2), color, thickness)

        return occluded_frame

    def create_noise_occlusion(self, frame, intensity=0.3):
        occluded_frame = frame.copy()

        noise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
        mask = np.random.random(frame.shape[:2]) < intensity * 0.1
        mask = np.stack([mask] * 3, axis=-1)
        occluded_frame = np.where(mask, noise, occluded_frame)

        return occluded_frame

    def create_blur_occlusion(self, frame, intensity=0.3):
        h, w = frame.shape[:2]
        occluded_frame = frame.copy()

        num_regions = int(intensity * 5)
        for _ in range(num_regions):
            region_w = random.randint(int(w * 0.1), int(w * 0.3))
            region_h = random.randint(int(h * 0.1), int(h * 0.3))
            x = random.randint(0, max(1, w - region_w))
            y = random.randint(0, max(1, h - region_h))

            region = occluded_frame[y : y + region_h, x : x + region_w]
            blurred_region = cv2.GaussianBlur(region, (51, 51), 0)
            occluded_frame[y : y + region_h, x : x + region_w] = blurred_region

        return occluded_frame

    # ------------------------------------------------------------------
    # Mixed & dispatcher
    # ------------------------------------------------------------------

    def create_mixed_occlusion(self, frame, intensity=0.3):
        occlusion_types = [
            "rectangle",
            "circle",
            "line",
            "noise",
            "blur",
        ]
        selected_types = random.sample(occlusion_types[:-1], random.randint(2, 4))

        occluded_frame = frame.copy()
        for occlusion_type in selected_types:
            occluded_frame = self.apply_occlusion(
                occluded_frame,
                occlusion_type,
                intensity * 0.7,
                # Frame index intentionally omitted so we do not nest the
                # slow‑down logic recursively.
            )
        return occluded_frame

    def apply_occlusion(self, frame, occlusion_type, intensity=0.3, frame_index=None):
        """Apply the specified *occlusion_type* to *frame*.

        For smooth playback we *reuse* the same random seed for an entire block
        of ``refresh_rate`` frames.  This makes the occlusion pattern change
        only every *n* frames instead of every single frame.
        """

        occlusion_functions = {
            "rectangle": self.create_random_rectangle_occlusion,
            "circle": self.create_circular_occlusion,
            "line": self.create_line_occlusion,
            "noise": self.create_noise_occlusion,
            "blur": self.create_blur_occlusion,
            "mixed": self.create_mixed_occlusion,
        }

        if occlusion_type not in occlusion_functions:
            print(f"Unknown occlusion type: {occlusion_type}")
            return frame

        # ------------------------------------------------------------------
        # Slow‑down logic
        # ------------------------------------------------------------------
        if frame_index is not None:
            # Quantise the frame index so that *seed_val* stays the same for
            # ``refresh_rate`` consecutive frames.
            seed_val = frame_index // self.refresh_rate

            # Preserve caller RNG state, reseed, then restore.
            state = random.getstate()
            random.seed(seed_val)
            occluded = occlusion_functions[occlusion_type](frame, intensity)
            random.setstate(state)
            return occluded

        # Fallback: behave like the original implementation (fast changes)
        return occlusion_functions[occlusion_type](frame, intensity)

    # ------------------------------------------------------------------
    # Video processing helpers (updated to pass frame_index)
    # ------------------------------------------------------------------

    def _write_with_occlusion(self, cap, out, total_frames, occlusion_type, intensity, progress_callback=None):
        """Internal helper shared by *process_video* and *process_video_gradio*."""
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            occluded_frame = self.apply_occlusion(frame, occlusion_type, intensity, frame_count)
            out.write(occluded_frame)

            frame_count += 1
            if progress_callback and frame_count % 30 == 0:
                progress = frame_count / total_frames
                progress_callback(progress, f"Processing: {frame_count}/{total_frames} frames")

        return frame_count

    def process_video_gradio(
        self,
        input_path,
        output_path,
        occlusion_type="mixed",
        intensity=0.3,
        progress_callback=None,
    ):
        """Process *input_path* and write result to *output_path* (Gradio)."""

        if not self.validate_video_format(input_path):
            return False, (
                "Error: Unsupported video format. Supported formats: "
                f"{self.supported_formats}"
            )

        video_info = self.get_video_info(input_path)
        if not video_info:
            return False, "Error: Could not read video file"

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info["fps"],
            (video_info["width"], video_info["height"]),
        )

        try:
            processed_frames = self._write_with_occlusion(
                cap,
                out,
                video_info["frame_count"],
                occlusion_type,
                intensity,
                progress_callback,
            )
        except Exception as exc:
            cap.release()
            out.release()
            return False, f"Error during processing: {exc}"
        finally:
            cap.release()
            out.release()

        return True, f"Video processing completed successfully! Processed {processed_frames} frames."

    def process_video(
        self,
        input_path,
        output_path,
        occlusion_type="mixed",
        intensity=0.3,
        preview=False,
    ):
        """CLI helper: identical to previous behaviour, just slower occlusions."""

        if not self.validate_video_format(input_path):
            print(
                f"Error: Unsupported video format. Supported formats: {self.supported_formats}"
            )
            return False

        video_info = self.get_video_info(input_path)
        if not video_info:
            print("Error: Could not read video file")
            return False

        print(f"Processing video: {input_path}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"FPS: {video_info['fps']:.2f}")
        print(f"Duration: {video_info['duration']:.2f} seconds")
        print(f"Total frames: {video_info['frame_count']}")

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info["fps"],
            (video_info["width"], video_info["height"]),
        )

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                occluded_frame = self.apply_occlusion(
                    frame,
                    occlusion_type,
                    intensity,
                    frame_count,
                )
                out.write(occluded_frame)

                if preview and frame_count % 30 == 0:
                    preview_frame = cv2.resize(
                        occluded_frame,
                        (
                            min(800, video_info["width"]),
                            min(600, video_info["height"]),
                        ),
                    )
                    cv2.imshow("Occlusion Preview", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / video_info["frame_count"]) * 100
                    print(
                        f"Progress: {progress:.1f}% ({frame_count}/{video_info['frame_count']} frames)"
                    )
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"Video processing completed: {output_path}")
        return True


# ----------------------------------------------------------------------
# Gradio front‑end (unchanged except for using the new, slower occluder)
# ----------------------------------------------------------------------

def create_gradio_interface():
    """Return a complete Gradio *Blocks* demo."""

    def process_video_gradio(video_file, occlusion_type, intensity, progress=gr.Progress()):
        if video_file is None:
            return None, "Please upload a video file first."

        occluder = VideoOccluder()  # Uses default refresh_rate=5
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
            output_path = temp_output.name

        def update_progress(prog, message):
            progress(prog, desc=message)

        success, message = occluder.process_video_gradio(
            video_file, output_path, occlusion_type, intensity, update_progress
        )

        if success:
            return output_path, message
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None, message

    def get_video_info_gradio(video_file):
        if video_file is None:
            return "No video uploaded"

        occluder = VideoOccluder()
        info = occluder.get_video_info(video_file)
        if not info:
            return "Could not read video information"

        return f"""
        **Video Information:**
        - Resolution: {info['width']} x {info['height']}
        - FPS: {info['fps']:.2f}
        - Duration: {info['duration']:.2f} seconds
        - Total Frames: {info['frame_count']}
        - File Size: {os.path.getsize(video_file) / (1024*1024):.1f} MB
        """

    with gr.Blocks(
        title="Video Occlusion Tool for Pose Estimation",
        theme=gr.themes.Soft() if hasattr(gr, "themes") else None,
        css="""
        .gradio-container { max-width: 1200px !important; }
        .video-container { max-height: 500px; }
        """,
    ) as demo:
        gr.Markdown(
            """
        # 🎬 Video Occlusion Tool for Pose Estimation Testing

        This tool applies various occlusion effects to videos for testing pose
        estimation models (MediaPipe and Martinez) under challenging
        conditions.  Perfect for MPI‑INF‑3DHP dataset processing.
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📹 Input Video")
                gr.Markdown(
                    "**Supported formats:** MP4, AVI, MOV, MKV, FLV, WMV, M4V, 3GP, WebM, OGV, TS, MTS"
                )
                video_input = gr.Video(label="Upload Video File")
                video_info_display = gr.Markdown("Upload a video to see its information")

                gr.Markdown("## ⚙️ Occlusion Settings")
                gr.Markdown("**Choose the type of occlusion to apply:**")
                occlusion_type = gr.Dropdown(
                    choices=[
                        ("Mixed (Recommended)", "mixed"),
                        ("Rectangle Blocks", "rectangle"),
                        ("Circular Blocks", "circle"),
                        ("Line Patterns", "line"),
                        ("Noise Patches", "noise"),
                        ("Blur Regions", "blur"),
                    ],
                    value="mixed",
                    label="Occlusion Type",
                )
                gr.Markdown(
                    "**Adjust intensity - Higher values create more challenging conditions:**"
                )
                intensity = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Occlusion Intensity",
                )

                gr.Markdown("---")
                process_btn = gr.Button("🚀 Apply Occlusion", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## 📤 Output Video")
                gr.Markdown("**Download this video for pose estimation testing**")
                video_output = gr.Video(label="Processed Video")
                gr.Markdown("**Processing Status:**")
                status_output = gr.Textbox(
                    label="Status Messages",
                    placeholder="Status messages will appear here...",
                    interactive=False,
                )

        with gr.Row():
            gr.Markdown(
                """
            ## 📋 Occlusion Types Explained

            - **Mixed**: Combines multiple occlusion types for maximum challenge
            - **Rectangle Blocks**: Random rectangular occlusions of various sizes
            - **Circular Blocks**: Random circular occlusions 
            - **Line Patterns**: Random line occlusions across the frame
            - **Noise Patches**: Random noise regions
            - **Blur Regions**: Selective blur effects on parts of the frame

            ## 🎯 Usage Tips

            - **Low Intensity (0.1‑0.3)**: Mild occlusions for basic testing
            - **Medium Intensity (0.4‑0.6)**: Moderate challenges
            - **High Intensity (0.7‑1.0)**: Severe occlusions for stress testing
            - **Mixed type** is recommended for comprehensive pose estimation evaluation
            """
            )

        video_input.change(fn=get_video_info_gradio, inputs=[video_input], outputs=[video_info_display])
        process_btn.click(
            fn=process_video_gradio,
            inputs=[video_input, occlusion_type, intensity],
            outputs=[video_output, status_output],
        )

        gr.Markdown(
            """
        ## 🎪 Example Usage

        1. Upload your MPI‑INF‑3DHP dataset video
        2. Select occlusion type (Mixed recommended)
        3. Adjust intensity based on desired difficulty
        4. Click **Apply Occlusion** to process
        5. Download the processed video for pose estimation testing
        """
        )

    return demo


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply occlusion effects to videos for pose estimation testing"
    )
    parser.add_argument("--input", "-i", type=str, help="Input video file path")
    parser.add_argument("--output", "-o", type=str, help="Output video file path")
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="mixed",
        choices=["rectangle", "circle", "line", "noise", "blur", "mixed"],
        help="Type of occlusion to apply",
    )
    parser.add_argument(
        "--intensity", type=float, default=0.3, help="Intensity of occlusion (0.1 to 1.0)"
    )
    parser.add_argument("--preview", action="store_true", help="Show preview while processing")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use command line interface instead of Gradio",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    # Decide whether we are running in CLI or launching Gradio
    has_processing_args = args.input or args.cli
    if not has_processing_args:
        print("🚀 Launching Gradio Web Interface…")
        print(f"📱 Opening on: http://localhost:{args.port}")
        if args.share:
            print("🌐 Creating public share link…")

        demo = create_gradio_interface()
        demo.launch(
            server_port=args.port,
            share=args.share,
            show_api=False,
            show_error=True,
            inbrowser=True,
        )
        return

    occluder = VideoOccluder()

    if not args.input:
        print("=== Video Occlusion Tool - Command Line Mode ===")
        print("💡 Tip: Run without --cli flag to use the web interface!")
        print("Supported formats:", occluder.supported_formats)

        while True:
            input_path = input("\nEnter the path to your input video file: ").strip().strip("\"")
            if os.path.exists(input_path):
                if occluder.validate_video_format(input_path):
                    break
                print("Unsupported video format. Please use one of:", occluder.supported_formats)
            else:
                print("File not found. Please check the path.")

        base_name = Path(input_path).stem
        output_path = (
            input(f"Enter output path (default: {base_name}_occluded.mp4): ").strip()
            or f"{base_name}_occluded.mp4"
        )

        print("\nAvailable occlusion types:")
        print("1. rectangle - Random rectangular blocks")
        print("2. circle     - Random circular blocks")
        print("3. line       - Random line occlusions")
        print("4. noise      - Random noise patches")
        print("5. blur       - Selective blur regions")
        print("6. mixed      - Combination of multiple types (recommended)")
        choice = input("Choose occlusion type (1-6, default: 6): ").strip()
        occlusion_type = {
            "1": "rectangle",
            "2": "circle",
            "3": "line",
            "4": "noise",
            "5": "blur",
            "6": "mixed",
        }.get(choice, "mixed")

        intensity_input = input("Enter occlusion intensity (0.1‑1.0, default: 0.3): ").strip()
        try:
            intensity = float(intensity_input) if intensity_input else 0.3
            intensity = max(0.1, min(1.0, intensity))
        except ValueError:
            intensity = 0.3

        preview = input("Show preview while processing? (y/N): ").strip().lower() == "y"
    else:
        input_path = args.input
        output_path = args.output or f"{Path(input_path).stem}_occluded.mp4"
        occlusion_type = args.type
        intensity = max(0.1, min(1.0, args.intensity))
        preview = args.preview

    success = occluder.process_video(input_path, output_path, occlusion_type, intensity, preview)
    if success:
        print(f"\n✅ Successfully created occluded video: {output_path}")
        print("This video is ready for pose estimation testing with MediaPipe and Martinez models.")
    else:
        print("\n❌ Failed to process video")


if __name__ == "__main__":
    main()
