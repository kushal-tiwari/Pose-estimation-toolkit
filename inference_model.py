# inference_model.py - FIXED VERSION
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import mediapipe as mp

# Absolute path relative to the *script*, not the launch CWD
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'model_checkpoint.pth')

# Import loaders
try:
    from mpi_inf_loader import get_mpi_inf_data, normalize_mpi_inf_data
except ImportError:
    print("Could not import mpi_inf_loader directly.")
    import sys
    sys.path.append('./data')
    try:
        from mpi_inf_loader import get_mpi_inf_data, normalize_mpi_inf_data
    except ImportError:
        print("Still couldn't import mpi_inf_loader. Please check file locations.")
        def get_mpi_inf_data(*args, **kwargs):
            raise NotImplementedError("mpi_inf_loader not available.")
        def normalize_mpi_inf_data(*args, **kwargs):
            raise NotImplementedError("mpi_inf_loader not available.")

class SimplePoseNet(nn.Module):
    """Simple neural network for 2D to 3D pose lifting."""
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super(SimplePoseNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def get_mpi_inf_joint_order():
    """The correct 28-joint order that matches your training data."""
    return [
        "pelvis",        # 0  ← ROOT JOINT (pelvis center)
        "spine1",        # 1  - lower spine
        "spine2",        # 2  - mid spine
        "spine3",        # 3  - upper spine (thorax)
        "neck_base",     # 4  - neck base  
        "neck",          # 5  - neck
        "head",          # 6  - head
        "head_top",      # 7  - head top
        "left_clavicle", # 8  - left clavicle
        "left_shoulder", # 9  - left shoulder
        "left_elbow",    # 10 - left elbow
        "left_wrist",    # 11 - left wrist
        "left_hand",     # 12 - left hand
        "left_fingers",  # 13 - left fingers
        "right_clavicle",# 14 - right clavicle
        "right_shoulder",# 15 - right shoulder
        "right_elbow",   # 16 - right elbow
        "right_wrist",   # 17 - right wrist
        "right_hand",    # 18 - right hand
        "right_fingers", # 19 - right fingers
        "left_hip",      # 20 - left hip
        "left_knee",     # 21 - left knee
        "left_ankle",    # 22 - left ankle
        "left_toe",      # 23 - left toe
        "right_hip",     # 24 - right hip
        "right_knee",    # 25 - right knee
        "right_ankle",   # 26 - right ankle
        "right_toe"      # 27 - right toe
    ]

def get_skeleton_connections_mpi_inf():
    return [
        # Spine
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        # Left arm
        (4, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        # Right arm
        (4, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
        # Left leg
        (0, 20), (20, 21), (21, 22), (22, 23),
        # Right leg
        (0, 24), (24, 25), (25, 26), (26, 27)
    ]

def plot_3d_pose(pose_3d, title="3D Pose", ax=None, center_on_root=True):
    """
    Enhanced 3D pose visualizer with colored limbs, consistent thickness, and centered root.
    
    Args:
        pose_3d: (J, 3) array of 3D joint coordinates
        title: plot title
        ax: Matplotlib 3D axis (optional)
        center_on_root: whether to subtract pelvis position for visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

    if center_on_root and len(pose_3d) > 0:
        root = pose_3d[0].copy()
        pose_3d_vis = pose_3d - root
    else:
        pose_3d_vis = pose_3d.copy()

    # Joint connections (MPI-INF-3DHP 28-joint skeleton)
    connections = get_skeleton_connections_mpi_inf()

    # Limb color map: torso, left, right, head
    color_map = {
        'torso': '#FF5733',     # reddish
        'left': '#33C1FF',      # blue
        'right': '#8DFF33',     # green
        'head': '#DA33FF'       # purple
    }

    def get_limb_color(i, j):
        left = [9,10,11,12,13,20,21,22,23]      # left arm + left leg
        right = [15,16,17,18,19,24,25,26,27]    # right arm + right leg
        head = [4,5,6,7]
        if i in left or j in left:
            return color_map['left']
        elif i in right or j in right:
            return color_map['right']
        elif i in head or j in head:
            return color_map['head']
        else:
            return color_map['torso']

    # Plot connections (bones)
    for i, j in connections:
        if i < len(pose_3d_vis) and j < len(pose_3d_vis):
            xs = [pose_3d_vis[i, 0], pose_3d_vis[j, 0]]
            ys = [pose_3d_vis[i, 1], pose_3d_vis[j, 1]]
            zs = [pose_3d_vis[i, 2], pose_3d_vis[j, 2]]
            ax.plot(xs, ys, zs, c=get_limb_color(i, j), linewidth=3)

    # Plot joints
    ax.scatter(pose_3d_vis[:, 0], pose_3d_vis[:, 1], pose_3d_vis[:, 2],
               c='black', s=20, alpha=1.0)

    # Centered axis limits
    max_range = np.array([
        pose_3d_vis[:, 0].max() - pose_3d_vis[:, 0].min(),
        pose_3d_vis[:, 1].max() - pose_3d_vis[:, 1].min(),
        pose_3d_vis[:, 2].max() - pose_3d_vis[:, 2].min()
    ]).max() / 2.0

    mid_x = (pose_3d_vis[:, 0].max() + pose_3d_vis[:, 0].min()) * 0.5
    mid_y = (pose_3d_vis[:, 1].max() + pose_3d_vis[:, 1].min()) * 0.5
    mid_z = (pose_3d_vis[:, 2].max() + pose_3d_vis[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-70)
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])

    return ax

def predict_3d_poses_from_2d(model, poses_2d_input, device):
    """Predict 3D poses from 2D poses using the trained model."""
    model.eval()
    model.to(device)
    
    print("DEBUG → poses_2d_input.shape:", poses_2d_input.shape)
    
    if poses_2d_input.shape[0] > 0:
        print("DEBUG → Joint 0 (pelvis):", poses_2d_input[0, 0])
        print("DEBUG → Joint 7 (head_top):", poses_2d_input[0, 7])
        print("DEBUG → Joint 16 (right_elbow):", poses_2d_input[0, 16])
        print("DEBUG → Joint 26 (right_ankle):", poses_2d_input[0, 26])
    
    # Convert to tensor and flatten
    poses_2d_tensor = torch.FloatTensor(poses_2d_input).to(device)
    poses_2d_flat = poses_2d_tensor.view(poses_2d_tensor.size(0), -1)
    
    with torch.no_grad():
        predictions_flat = model(poses_2d_flat)
    
    # Reshape predictions back to 3D pose format
    num_joints = poses_2d_input.shape[1]
    predictions_3d = predictions_flat.view(-1, num_joints, 3)
    
    return predictions_3d.cpu().numpy()

def load_trained_model(model_path, input_dim, output_dim, device):
    """Loads a trained SimplePoseNet model."""
    model = SimplePoseNet(input_dim, output_dim)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. "
                                "Please train the model first using test_main.py 'train' mode.")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    return model

import cv2
import numpy as np
import mediapipe as mp

def _mid(p1, p2):
    """Calculate midpoint between two points."""
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def estimate_2d_poses_from_video(video_path, expected_joints=28):
    """
    Extract 2D poses from video in CORRECT 28-joint MPI-INF-3DHP order.
    
    MPI-INF-3DHP 28-joint order:
    0: Pelvis (Hip center)
    1: Spine1 (Lower spine)
    2: Spine2 (Middle spine)
    3: Spine3/Thorax (Upper spine)
    4: Neck base
    5: Neck
    6: Head (Nose)
    7: Head top
    8: Left clavicle
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Left hand
    13: Left fingers
    14: Right clavicle
    15: Right shoulder
    16: Right elbow
    17: Right wrist
    18: Right hand
    19: Right fingers
    20: Left hip
    21: Left knee
    22: Left ankle
    23: Left foot
    24: Right hip
    25: Right knee
    26: Right ankle
    27: Right foot
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Could not open video"

    poses2d = []
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            coords = np.zeros((expected_joints, 2), dtype=np.float32)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                # Extract key MediaPipe landmarks (normalized coords * frame dimensions)
                nose = (lm[0].x * fw, lm[0].y * fh)
                left_shoulder = (lm[11].x * fw, lm[11].y * fh)
                right_shoulder = (lm[12].x * fw, lm[12].y * fh)
                left_elbow = (lm[13].x * fw, lm[13].y * fh)
                right_elbow = (lm[14].x * fw, lm[14].y * fh)
                left_wrist = (lm[15].x * fw, lm[15].y * fh)
                right_wrist = (lm[16].x * fw, lm[16].y * fh)
                left_hip = (lm[23].x * fw, lm[23].y * fh)
                right_hip = (lm[24].x * fw, lm[24].y * fh)
                left_knee = (lm[25].x * fw, lm[25].y * fh)
                right_knee = (lm[26].x * fw, lm[26].y * fh)
                left_ankle = (lm[27].x * fw, lm[27].y * fh)
                right_ankle = (lm[28].x * fw, lm[28].y * fh)
                left_heel = (lm[29].x * fw, lm[29].y * fh)
                right_heel = (lm[30].x * fw, lm[30].y * fh)
                left_foot_index = (lm[31].x * fw, lm[31].y * fh)
                right_foot_index = (lm[32].x * fw, lm[32].y * fh)
                
                # Hand landmarks
                left_pinky = (lm[17].x * fw, lm[17].y * fh)
                right_pinky = (lm[18].x * fw, lm[18].y * fh)
                left_index = (lm[19].x * fw, lm[19].y * fh)
                right_index = (lm[20].x * fw, lm[20].y * fh)
                left_thumb = (lm[21].x * fw, lm[21].y * fh)
                right_thumb = (lm[22].x * fw, lm[22].y * fh)
                
                # Calculate intermediate joints
                pelvis = _mid(left_hip, right_hip)
                thorax = _mid(left_shoulder, right_shoulder)
                
                # Spine joints (distribute along pelvis to thorax)
                spine1 = [(pelvis[0] * 2 + thorax[0] * 1) / 3, (pelvis[1] * 2 + thorax[1] * 1) / 3]
                spine2 = [(pelvis[0] * 1 + thorax[0] * 2) / 3, (pelvis[1] * 1 + thorax[1] * 2) / 3]
                spine3 = thorax
                
                # Neck and head calculations
                neck_base = [(thorax[0] + nose[0]) / 2, (thorax[1] + nose[1]) / 2]
                neck = [(neck_base[0] + nose[0]) / 2, (neck_base[1] + nose[1]) / 2]
                head_top = [nose[0], max(0, nose[1] - 30)]  # Estimate head top above nose
                
                # Clavicle joints (between neck base and shoulders)
                left_clavicle = [(neck_base[0] + left_shoulder[0]) / 2, (neck_base[1] + left_shoulder[1]) / 2]
                right_clavicle = [(neck_base[0] + right_shoulder[0]) / 2, (neck_base[1] + right_shoulder[1]) / 2]
                
                # Hand centers (average of available hand landmarks)
                left_hand = _mid(left_wrist, _mid(left_index, left_pinky))
                right_hand = _mid(right_wrist, _mid(right_index, right_pinky))
                
                # Finger tips (use index finger as representative)
                left_fingers = left_index
                right_fingers = right_index
                
                # Foot centers (between heel and toe)
                left_foot = _mid(left_heel, left_foot_index)
                right_foot = _mid(right_heel, right_foot_index)
                
                # Map to MPI-INF-3DHP 28-joint format - NO MORE DUPLICATES!
                coords[0]  = pelvis           # Pelvis
                coords[1]  = spine1           # Spine1
                coords[2]  = spine2           # Spine2  
                coords[3]  = spine3           # Spine3/Thorax
                coords[4]  = neck_base        # Neck base
                coords[5]  = neck             # Neck
                coords[6]  = nose             # Head (nose)
                coords[7]  = head_top         # Head top
                coords[8]  = left_clavicle    # Left clavicle
                coords[9]  = left_shoulder    # Left shoulder
                coords[10] = left_elbow       # Left elbow
                coords[11] = left_wrist       # Left wrist
                coords[12] = left_hand        # Left hand
                coords[13] = left_fingers     # Left fingers
                coords[14] = right_clavicle   # Right clavicle
                coords[15] = right_shoulder   # Right shoulder
                coords[16] = right_elbow      # Right elbow
                coords[17] = right_wrist      # Right wrist
                coords[18] = right_hand       # Right hand
                coords[19] = right_fingers    # Right fingers
                coords[20] = left_hip         # Left hip
                coords[21] = left_knee        # Left knee
                coords[22] = left_ankle       # Left ankle
                coords[23] = left_foot        # Left foot
                coords[24] = right_hip        # Right hip
                coords[25] = right_knee       # Right knee
                coords[26] = right_ankle      # Right ankle
                coords[27] = right_foot       # Right foot

                # Ensure all coordinates are valid (not NaN or inf)
                coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

            poses2d.append(coords)

    cap.release()
    return np.array(poses2d, dtype=np.float32), None

def validate_joint_mapping():
    """
    Validation function to ensure joint mapping is correct.
    Returns a dictionary mapping joint indices to their names.
    """
    joint_names = {
        0: "Pelvis",
        1: "Spine1", 
        2: "Spine2",
        3: "Spine3/Thorax",
        4: "Neck base",
        5: "Neck",
        6: "Head (Nose)",
        7: "Head top",
        8: "Left clavicle",
        9: "Left shoulder",
        10: "Left elbow",
        11: "Left wrist", 
        12: "Left hand",
        13: "Left fingers",
        14: "Right clavicle",
        15: "Right shoulder",
        16: "Right elbow",
        17: "Right wrist",
        18: "Right hand", 
        19: "Right fingers",
        20: "Left hip",
        21: "Left knee",
        22: "Left ankle",
        23: "Left foot",
        24: "Right hip", 
        25: "Right knee",
        26: "Right ankle",
        27: "Right foot"
    }
    
    print("MPI-INF-3DHP 28-Joint Mapping:")
    for idx, name in joint_names.items():
        print(f"Index {idx:2d}: {name}")
    
    return joint_names

# Additional utility function for debugging
def visualize_2d_pose(image, pose_2d, joint_names=None):
    """
    Visualize 2D pose on image for debugging purposes.
    """
    if joint_names is None:
        joint_names = validate_joint_mapping()
    
    img_vis = image.copy()
    
    # Draw joints
    for i, (x, y) in enumerate(pose_2d):
        if x > 0 and y > 0:  # Only draw valid joints
            cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(img_vis, str(i), (int(x), int(y-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw skeleton connections (basic connections)
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine
        (3, 4), (4, 5), (5, 6), (6, 7),  # Neck/Head
        (4, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),  # Left arm
        (4, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),  # Right arm
        (0, 20), (20, 21), (21, 22), (22, 23),  # Left leg
        (0, 24), (24, 25), (25, 26), (26, 27),  # Right leg
    ]
    
    for start_idx, end_idx in connections:
        start_point = pose_2d[start_idx]
        end_point = pose_2d[end_idx]
        
        if (start_point[0] > 0 and start_point[1] > 0 and 
            end_point[0] > 0 and end_point[1] > 0):
            cv2.line(img_vis, 
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (0, 0, 255), 2)
    
    return img_vis

if __name__ == "__main__":
    # Test the joint mapping
    validate_joint_mapping()

def normalize_2d_poses_for_model(poses_2d, image_width=None, image_height=None, root_joint_idx=0):
    """Normalize 2D poses for model input."""
    poses = poses_2d.copy().astype(np.float32)
    
    # If image dimensions provided, normalize to [-1, 1] range
    if image_width is not None and image_height is not None:
        poses[..., 0] = 2.0 * poses[..., 0] / image_width - 1.0   # X: [0, width] -> [-1, 1]
        poses[..., 1] = 2.0 * poses[..., 1] / image_height - 1.0  # Y: [0, height] -> [-1, 1]
    
    return poses

def denormalize_3d_poses_for_visualization(pred_3d, scale_factor=1.0):
    """
    FIXED: Apply correct scaling for visualization.
    
    Args:
        pred_3d: Predicted 3D poses from model
        scale_factor: Scale factor (use 1.0 to keep original scale)
    """
    return pred_3d * scale_factor

def process_video_to_3d_poses(video_path, model_path, output_path=None, device='cpu'):
    """
    Complete pipeline: Extract 2D from video → Predict 3D pose → Visualize 3D.
    Ensures consistency with training (root-relative + meters).
    """
    print("Step 1: Extracting 2D poses from video...")
    poses_2d, error = estimate_2d_poses_from_video(video_path, expected_joints=28)

    if error:
        print(f"Error extracting poses: {error}")
        return None

    print(f"Extracted 2D poses shape: {poses_2d.shape}")

    # Get video dimensions for normalization
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("Step 2: Normalizing 2D poses...")
    poses_2d_normalized = normalize_2d_poses_for_model(poses_2d, fw, fh, root_joint_idx=0)

    print("Step 3: Loading trained model...")
    num_joints = poses_2d.shape[1]
    input_dim = num_joints * 2
    output_dim = num_joints * 3

    print(f"Model dimensions: {input_dim} -> {output_dim}")
    model = load_trained_model(model_path, input_dim, output_dim, device)

    print("Step 4: Predicting 3D poses...")
    poses_3d_predicted = predict_3d_poses_from_2d(model, poses_2d_normalized, device)

    # Post-process: convert meters → mm and re-center from root-relative to global
    poses_3d_predicted *= 1000.0
    poses_3d_predicted += poses_3d_predicted[:, 0:1, :]  # add pelvis back

    # Debug output
    print(f"Raw 3D prediction (pelvis): {poses_3d_predicted[0, 0]}")
    print(f"Raw 3D prediction (head): {poses_3d_predicted[0, 7]}")
    print(f"Raw 3D prediction range: {poses_3d_predicted.min():.3f} to {poses_3d_predicted.max():.3f}")

    print("Step 5: Preparing for visualization...")
    poses_3d_vis = poses_3d_predicted  # already scaled and re-centered

    print("Step 6: Visualizing results...")
    if len(poses_3d_vis) > 0:
        fig = plt.figure(figsize=(15, 5))

        # Original 2D pose
        ax1 = fig.add_subplot(131)
        ax1.scatter(poses_2d[0, :, 0], poses_2d[0, :, 1], c='blue', s=50)
        ax1.set_title('Original 2D Pose (Frame 0)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.invert_yaxis()

        # Normalized 2D pose
        ax2 = fig.add_subplot(132)
        ax2.scatter(poses_2d_normalized[0, :, 0], poses_2d_normalized[0, :, 1], c='green', s=50)
        ax2.set_title('Normalized 2D Pose (Frame 0)')
        ax2.set_xlabel('X (normalized)')
        ax2.set_ylabel('Y (normalized)')

        # 3D pose prediction
        ax3 = fig.add_subplot(133, projection='3d')
        plot_3d_pose(poses_3d_vis[0], title='Predicted 3D Pose (Frame 0)', ax=ax3, center_on_root=True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")

        plt.show()

    print(f"Final 3D poses shape: {poses_3d_vis.shape}")
    return poses_3d_vis

# Utility functions
def inspect_dataset_format():
    """Utility function to inspect your converted dataset format."""
    print("Inspecting converted MPI-INF dataset format...")
    
    data_3d_path = 'data/data_3d_mpi_inf_3dhp.npz'
    data_2d_path = 'data/data_2d_mpi_inf_3dhp_gt.npz'
    
    if os.path.exists(data_3d_path):
        data_3d = np.load(data_3d_path)
        print(f"\n3D Data file keys: {list(data_3d.keys())}")
        
        if 'positions_3d' in data_3d:
            poses_3d = data_3d['positions_3d']
            print(f"3D poses shape: {poses_3d.shape}")
            print(f"Number of joints: {poses_3d.shape[1]}")
            print(f"Sample 3D pose (first frame):\n{poses_3d[0]}")
            
            if poses_3d.shape[1] > 0:
                root_joint = poses_3d[0, 0]
                print(f"Root joint (joint 0) position: {root_joint}")
                print(f"Is root near origin? {np.allclose(root_joint, 0, atol=10)}")
    else:
        print(f"3D data file not found: {data_3d_path}")
    
    if os.path.exists(data_2d_path):
        data_2d = np.load(data_2d_path)
        print(f"\n2D Data file keys: {list(data_2d.keys())}")
        
        if 'positions_2d' in data_2d:
            poses_2d = data_2d['positions_2d']
            print(f"2D poses shape: {poses_2d.shape}")
            print(f"Sample 2D pose (first frame):\n{poses_2d[0]}")
            
            if poses_2d.shape[1] > 0:
                root_joint_2d = poses_2d[0, 0]
                print(f"Root joint (joint 0) 2D position: {root_joint_2d}")
                print(f"Is 2D root near origin? {np.allclose(root_joint_2d, 0, atol=0.1)}")
    else:
        print(f"2D data file not found: {data_2d_path}")

def debug_joint_mapping():
    """Debug function to verify joint mapping."""
    print("=== MPI-INF 28-Joint Order Debug ===")
    joint_names = get_mpi_inf_joint_order()
    
    for i, name in enumerate(joint_names):
        print(f"Joint {i:2d}: {name}")
    
    print(f"\nTotal joints: {len(joint_names)}")
    print("✅ PELVIS is at index 0 (ROOT JOINT)")
    print("✅ Total count matches your 28-joint training data")