import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from mpi_inf_loader import load_mpi_inf_3dhp  # Import your MPI-INF loader

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Human Pose Estimation Training/Evaluation')
    
    # Dataset selection
    parser.add_argument('--dataset', 
                       default='mpi_inf_3dhp', 
                       choices=['mpi_inf_3dhp'],  # Only MPI-INF for now
                       help='Dataset to use for training/evaluation')
    
    # Training parameters
    parser.add_argument('--mode', 
                       default='train', 
                       choices=['train', 'eval', 'test', 'visualize'],
                       help='Mode: train, eval, test, or visualize')
    
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=32,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', 
                       type=float, 
                       default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--model_path', 
                       type=str, 
                       default='model_checkpoint.pth',
                       help='Path to save/load model')
    
    # Data parameters
    parser.add_argument('--train_split', 
                       type=float, 
                       default=0.8,
                       help='Training data split ratio')
    
    parser.add_argument('--max_samples', 
                       type=int, 
                       default=20000,
                       help='Maximum number of samples to use from dataset')
    
    parser.add_argument('--device', 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    # Visualization parameters
    parser.add_argument('--output_dir', 
                       type=str, 
                       default='pose_results',
                       help='Directory to save visualization results')
    
    parser.add_argument('--num_visualize', 
                       type=int, 
                       default=50,
                       help='Number of poses to visualize')
    
    return parser.parse_args()

def load_dataset(dataset_name, max_samples=20000):
    """Load the specified dataset with sample limit."""
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == 'mpi_inf_3dhp':
        positions_2d, positions_3d, subjects, actions = load_mpi_inf_3dhp()
        
        print(f"Full MPI-INF-3DHP dataset loaded: {positions_2d.shape[0]} samples")
        
        # Convert all arrays to numpy arrays to ensure consistent indexing
        positions_2d = np.array(positions_2d)
        positions_3d = np.array(positions_3d)
        subjects = np.array(subjects)
        actions = np.array(actions)
        
        # Limit to max_samples if dataset is larger
        if positions_2d.shape[0] > max_samples:
            print(f"Limiting dataset to {max_samples} samples...")
            # Use random sampling to get diverse data
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(positions_2d.shape[0], max_samples, replace=False)
            indices = np.sort(indices)  # Sort to maintain some order
            
            positions_2d = positions_2d[indices]
            positions_3d = positions_3d[indices]
            subjects = subjects[indices]
            actions = actions[indices]
        
        print(f"Final dataset size: {positions_2d.shape[0]} samples")
        print(f"Number of subjects: {len(np.unique(subjects))}")
        print(f"Number of actions: {len(np.unique(actions))}")
        
        # Step 3.4: Print shapes and check compatibility
        print("\n--- Data Shape Compatibility Check ---")
        print("2D poses shape:", positions_2d.shape)
        print("3D poses shape:", positions_3d.shape)
        
        # Check if shapes are compatible with Martinez model format
        # Expected format: [num_samples, num_joints, 2 or 3]
        if len(positions_2d.shape) != 3:
            print(f"Warning: 2D poses shape is {positions_2d.shape}, expected 3D array [samples, joints, 2]")
        if len(positions_3d.shape) != 3:
            print(f"Warning: 3D poses shape is {positions_3d.shape}, expected 3D array [samples, joints, 3]")
        
        if positions_2d.shape[-1] != 2:
            print(f"Warning: 2D poses last dimension is {positions_2d.shape[-1]}, expected 2")
        if positions_3d.shape[-1] != 3:
            print(f"Warning: 3D poses last dimension is {positions_3d.shape[-1]}, expected 3")
        
        # Auto-reshape if needed
        positions_2d, positions_3d = ensure_shape_compatibility(positions_2d, positions_3d)
        
        print("After shape adjustment:")
        print("2D poses shape:", positions_2d.shape)
        print("3D poses shape:", positions_3d.shape)
        print("--- Shape Check Complete ---\n")
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    return positions_2d, positions_3d, subjects, actions

def ensure_shape_compatibility(poses_2d, poses_3d):
    """Ensure data shapes are compatible with Martinez model format."""
    
    # Handle different possible input formats
    if len(poses_2d.shape) == 2:
        # If flattened [samples, joints*2], reshape to [samples, joints, 2]
        num_samples = poses_2d.shape[0]
        num_coords = poses_2d.shape[1]
        if num_coords % 2 == 0:
            num_joints = num_coords // 2
            poses_2d = poses_2d.reshape(num_samples, num_joints, 2)
            print(f"Reshaped 2D poses from {(num_samples, num_coords)} to {poses_2d.shape}")
    
    if len(poses_3d.shape) == 2:
        # If flattened [samples, joints*3], reshape to [samples, joints, 3]
        num_samples = poses_3d.shape[0]
        num_coords = poses_3d.shape[1]
        if num_coords % 3 == 0:
            num_joints = num_coords // 3
            poses_3d = poses_3d.reshape(num_samples, num_joints, 3)
            print(f"Reshaped 3D poses from {(num_samples, num_coords)} to {poses_3d.shape}")
    
    # Check if transpose is needed (sometimes data comes as [joints, coords, samples])
    if len(poses_2d.shape) == 3:
        if poses_2d.shape[2] != 2 and poses_2d.shape[0] == 2:
            # Likely [2, joints, samples] -> transpose to [samples, joints, 2]
            poses_2d = np.transpose(poses_2d, (2, 1, 0))
            print(f"Transposed 2D poses to {poses_2d.shape}")
        elif poses_2d.shape[1] == 2 and poses_2d.shape[2] != 2:
            # Likely [samples, 2, joints] -> transpose to [samples, joints, 2]
            poses_2d = np.transpose(poses_2d, (0, 2, 1))
            print(f"Transposed 2D poses to {poses_2d.shape}")
    
    if len(poses_3d.shape) == 3:
        if poses_3d.shape[2] != 3 and poses_3d.shape[0] == 3:
            # Likely [3, joints, samples] -> transpose to [samples, joints, 3]
            poses_3d = np.transpose(poses_3d, (2, 1, 0))
            print(f"Transposed 3D poses to {poses_3d.shape}")
        elif poses_3d.shape[1] == 3 and poses_3d.shape[2] != 3:
            # Likely [samples, 3, joints] -> transpose to [samples, joints, 3]
            poses_3d = np.transpose(poses_3d, (0, 2, 1))
            print(f"Transposed 3D poses to {poses_3d.shape}")
    
    return poses_2d, poses_3d

def create_data_loaders(positions_2d, positions_3d, subjects, actions, batch_size, train_split):
    """Create training and validation data loaders with metadata handling."""
    # Convert to tensors
    poses_2d_tensor = torch.FloatTensor(positions_2d)
    poses_3d_tensor = torch.FloatTensor(positions_3d)
    
    # Step 3.5: Handle metadata (subjects, actions) for MPI-INF-3DHP
    print(f"\n--- Training/Evaluation Pipeline Setup ---")
    print(f"Available subjects: {np.unique(subjects)}")
    print(f"Available actions: {np.unique(actions)}")
    
    # Option 1: Subject-based split (recommended to avoid data leakage)
    unique_subjects = np.unique(subjects)
    n_train_subjects = int(len(unique_subjects) * train_split)
    
    # Randomly select training subjects
    np.random.seed(42)  # For reproducibility
    train_subjects = np.random.choice(unique_subjects, n_train_subjects, replace=False)
    val_subjects = np.setdiff1d(unique_subjects, train_subjects)
    
    # Create subject-based indices
    train_indices = np.where(np.isin(subjects, train_subjects))[0]
    val_indices = np.where(np.isin(subjects, val_subjects))[0]
    
    print(f"Training subjects: {train_subjects}")
    print(f"Validation subjects: {val_subjects}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Convert to torch indices
    train_indices = torch.LongTensor(train_indices)
    val_indices = torch.LongTensor(val_indices)
    
    # Create datasets
    train_dataset = TensorDataset(poses_2d_tensor[train_indices], 
                                 poses_3d_tensor[train_indices])
    val_dataset = TensorDataset(poses_2d_tensor[val_indices], 
                               poses_3d_tensor[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Final training batches: {len(train_loader)}")
    print(f"Final validation batches: {len(val_loader)}")
    print("--- Pipeline Setup Complete ---\n")
    
    return train_loader, val_loader

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

def get_skeleton_connections():
    """Define skeleton connections for visualization."""
    # Standard human pose skeleton connections (adjust based on your joint ordering)
    connections = [
        # Head and torso
        (0, 1),   # nose to neck
        (1, 2),   # neck to right shoulder
        (1, 5),   # neck to left shoulder
        (2, 3),   # right shoulder to right elbow
        (3, 4),   # right elbow to right wrist
        (5, 6),   # left shoulder to left elbow
        (6, 7),   # left elbow to left wrist
        (1, 8),   # neck to mid hip
        (8, 9),   # mid hip to right hip
        (8, 12),  # mid hip to left hip
        (9, 10),  # right hip to right knee
        (10, 11), # right knee to right ankle
        (12, 13), # left hip to left knee
        (13, 14), # left knee to left ankle
    ]
    return connections

def plot_2d_pose(pose_2d, title="2D Pose", ax=None, color='blue'):
    """Plot a single 2D pose."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    connections = get_skeleton_connections()
    
    # Plot joints
    ax.scatter(pose_2d[:, 0], pose_2d[:, 1], c=color, s=50, alpha=0.8)
    
    # Plot skeleton connections
    for start_idx, end_idx in connections:
        if start_idx < len(pose_2d) and end_idx < len(pose_2d):
            x_coords = [pose_2d[start_idx, 0], pose_2d[end_idx, 0]]
            y_coords = [pose_2d[start_idx, 1], pose_2d[end_idx, 1]]
            ax.plot(x_coords, y_coords, c=color, linewidth=2, alpha=0.7)
    
    # Add joint numbers
    for i, (x, y) in enumerate(pose_2d):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax

def plot_3d_pose(pose_3d, title="3D Pose", ax=None, color='red'):
    """Plot a single 3D pose."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    connections = get_skeleton_connections()
    
    # Plot joints
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c=color, s=50, alpha=0.8)
    
    # Plot skeleton connections
    for start_idx, end_idx in connections:
        if start_idx < len(pose_3d) and end_idx < len(pose_3d):
            x_coords = [pose_3d[start_idx, 0], pose_3d[end_idx, 0]]
            y_coords = [pose_3d[start_idx, 1], pose_3d[end_idx, 1]]
            z_coords = [pose_3d[start_idx, 2], pose_3d[end_idx, 2]]
            ax.plot(x_coords, y_coords, z_coords, c=color, linewidth=2, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax

def create_comparison_image(pose_2d, pose_3d_gt, pose_3d_pred, save_path, pose_idx):
    """Create a comparison image showing 2D input, 3D ground truth, and 3D prediction."""
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 2D input
    ax1 = fig.add_subplot(131)
    plot_2d_pose(pose_2d, "2D Input", ax1, color='blue')
    
    # Plot 3D ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    plot_3d_pose(pose_3d_gt, "3D Ground Truth", ax2, color='green')
    
    # Plot 3D prediction
    ax3 = fig.add_subplot(133, projection='3d')
    plot_3d_pose(pose_3d_pred, "3D Prediction", ax3, color='red')
    
    plt.suptitle(f'Pose Comparison #{pose_idx}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison image: {save_path}")

def visualize_poses(model, val_loader, device, output_dir, num_visualize=50):
    """Generate pose visualization images."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '2d_poses'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '3d_gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '3d_pred'), exist_ok=True)
    
    model.eval()
    visualized = 0
    
    print(f"Generating {num_visualize} pose visualizations...")
    
    with torch.no_grad():
        for batch_idx, (batch_2d, batch_3d) in enumerate(val_loader):
            if visualized >= num_visualize:
                break
                
            batch_2d, batch_3d = batch_2d.to(device), batch_3d.to(device)
            
            # Get predictions
            batch_2d_flat = batch_2d.view(batch_2d.size(0), -1)
            predictions_flat = model(batch_2d_flat)
            
            # Reshape predictions back to 3D pose format
            num_joints = batch_3d.shape[1]
            predictions_3d = predictions_flat.view(-1, num_joints, 3)
            
            # Convert to numpy for visualization
            poses_2d = batch_2d.cpu().numpy()
            poses_3d_gt = batch_3d.cpu().numpy()
            poses_3d_pred = predictions_3d.cpu().numpy()
            
            # Visualize each pose in the batch
            for i in range(poses_2d.shape[0]):
                if visualized >= num_visualize:
                    break
                
                pose_2d = poses_2d[i]
                pose_3d_gt = poses_3d_gt[i]
                pose_3d_pred = poses_3d_pred[i]
                
                # Create comparison image
                comparison_path = os.path.join(output_dir, 'comparisons', f'pose_{visualized:04d}_comparison.png')
                create_comparison_image(pose_2d, pose_3d_gt, pose_3d_pred, comparison_path, visualized)
                
                # Save individual 2D pose
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                plot_2d_pose(pose_2d, f"2D Pose #{visualized}", ax, color='blue')
                plt.savefig(os.path.join(output_dir, '2d_poses', f'2d_pose_{visualized:04d}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save 3D ground truth
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                plot_3d_pose(pose_3d_gt, f"3D Ground Truth #{visualized}", ax, color='green')
                plt.savefig(os.path.join(output_dir, '3d_gt', f'3d_gt_{visualized:04d}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save 3D prediction
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                plot_3d_pose(pose_3d_pred, f"3D Prediction #{visualized}", ax, color='red')
                plt.savefig(os.path.join(output_dir, '3d_pred', f'3d_pred_{visualized:04d}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                visualized += 1
                
                if visualized % 10 == 0:
                    print(f"Generated {visualized}/{num_visualize} visualizations...")
    
    print(f"\n✅ Generated {visualized} pose visualizations!")
    print(f"📁 Results saved in: {os.path.abspath(output_dir)}")
    print(f"   - Comparison images: {output_dir}/comparisons/")
    print(f"   - 2D poses: {output_dir}/2d_poses/")
    print(f"   - 3D ground truth: {output_dir}/3d_gt/")
    print(f"   - 3D predictions: {output_dir}/3d_pred/")

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, model_path):
    """Train the model."""
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_2d, batch_3d in train_loader:
            batch_2d, batch_3d = batch_2d.to(device), batch_3d.to(device)
            
            # Flatten 2D poses for input
            batch_2d_flat = batch_2d.view(batch_2d.size(0), -1)
            batch_3d_flat = batch_3d.view(batch_3d.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(batch_2d_flat)
            loss = criterion(outputs, batch_3d_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_2d, batch_3d in val_loader:
                batch_2d, batch_3d = batch_2d.to(device), batch_3d.to(device)
                
                batch_2d_flat = batch_2d.view(batch_2d.size(0), -1)
                batch_3d_flat = batch_3d.view(batch_3d.size(0), -1)
                
                outputs = model(batch_2d_flat)
                loss = criterion(outputs, batch_3d_flat)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, model_path)
            print(f'Best model saved at epoch {epoch+1}')

def evaluate_model(model, val_loader, criterion, device, model_path):
    """Evaluate the trained model."""
    # Load best model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_2d, batch_3d in val_loader:
            batch_2d, batch_3d = batch_2d.to(device), batch_3d.to(device)
            
            batch_2d_flat = batch_2d.view(batch_2d.size(0), -1)
            batch_3d_flat = batch_3d.view(batch_3d.size(0), -1)
            
            outputs = model(batch_2d_flat)
            loss = criterion(outputs, batch_3d_flat)
            
            total_loss += loss.item() * batch_2d.size(0)
            total_samples += batch_2d.size(0)
    
    avg_loss = total_loss / total_samples
    print(f'Evaluation Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max samples: {args.max_samples}")
    if args.mode == 'visualize':
        print(f"  Output directory: {args.output_dir}")
        print(f"  Number to visualize: {args.num_visualize}")
    print("-" * 50)
    
    # Load dataset with sample limit
    positions_2d, positions_3d, subjects, actions = load_dataset(args.dataset, args.max_samples)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        positions_2d, positions_3d, subjects, actions, args.batch_size, args.train_split
    )
    
    # Initialize model
    input_dim = positions_2d.shape[1] * positions_2d.shape[2]  # flattened 2D poses
    output_dim = positions_3d.shape[1] * positions_3d.shape[2]  # flattened 3D poses
    
    model = SimplePoseNet(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"Model initialized:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)
    
    # Execute based on mode
    if args.mode == 'train':
        print("Starting training...")
        train_model(model, train_loader, val_loader, criterion, optimizer,
                   args.epochs, args.device, args.model_path)
        print("Training completed!")
        
        # After training, automatically generate some visualizations
        print("\nGenerating post-training visualizations...")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        visualize_poses(model, val_loader, args.device, args.output_dir, min(20, args.num_visualize))
        
    elif args.mode == 'eval':
        print("Starting evaluation...")
        evaluate_model(model, val_loader, criterion, args.device, args.model_path)
        print("Evaluation completed!")
        
    elif args.mode == 'visualize':
        print("Starting visualization...")
        # Load trained model
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        visualize_poses(model, val_loader, args.device, args.output_dir, args.num_visualize)
        print("Visualization completed!")
        
    else:  # test mode
        print("Test mode - using validation set for testing...")
        evaluate_model(model, val_loader, criterion, args.device, args.model_path)

if __name__ == '__main__':
    main()