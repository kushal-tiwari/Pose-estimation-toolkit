import os
import scipy.io as sio
import numpy as np
import h5py

def convert_mpi_inf_to_martinez(max_samples=20000):
    """
    Convert MPI-INF-3DHP dataset to Martinez format with limited samples
    Handles the specific MPI-INF-3DHP data structure correctly
    
    Args:
        max_samples (int): Maximum number of samples to convert (default: 20000)
    """
    # Ensure 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    DATA_DIR = 'data/mpi_inf_3dhp'
    SUBJECTS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    
    # Target joint count for Martinez format (28 joints)
    TARGET_JOINTS = 28
    
    poses_3d = []
    poses_2d = []
    total_frames_processed = 0
    
    print(f"Starting MPI-INF-3DHP dataset conversion (max {max_samples} samples)...")
    print(f"Target format: {TARGET_JOINTS} joints per pose")
    print(f"Expected input dimension: {TARGET_JOINTS * 2} (2D poses)")
    print(f"Expected output dimension: {TARGET_JOINTS * 3} (3D poses)")
    
    for subj in SUBJECTS:
        if total_frames_processed >= max_samples:
            print(f"Reached maximum sample limit of {max_samples}, stopping conversion.")
            break
            
        subj_dir = os.path.join(DATA_DIR, subj)
        
        if not os.path.exists(subj_dir):
            print(f"Subject directory {subj_dir} does not exist, skipping.")
            continue
        
        print(f"Processing subject: {subj}")
        
        # Get all sequence directories
        sequences = [d for d in os.listdir(subj_dir) 
                    if os.path.isdir(os.path.join(subj_dir, d))]
        
        if not sequences:
            print(f"No sequences found for subject {subj}")
            continue
        
        for seq in sequences:
            if total_frames_processed >= max_samples:
                break
                
            seq_dir = os.path.join(subj_dir, seq)
            annot_file = os.path.join(seq_dir, 'annot.mat')
            
            if not os.path.exists(annot_file):
                print(f"Annotation file {annot_file} not found, skipping.")
                continue
            
            try:
                print(f"  Loading {annot_file}")
                
                # Try different loading methods for MPI-INF-3DHP
                pose_3d, pose_2d = load_mpi_inf_annotations(annot_file)
                
                if pose_3d is None or pose_2d is None:
                    print(f"    Failed to load pose data, skipping.")
                    continue
                
                print(f"    Loaded 3D poses shape: {pose_3d.shape}")
                print(f"    Loaded 2D poses shape: {pose_2d.shape}")
                
                # Validate basic shapes
                if len(pose_3d.shape) != 3 or len(pose_2d.shape) != 3:
                    print(f"    Invalid pose data dimensions, skipping.")
                    continue
                
                if pose_3d.shape[2] != 3 or pose_2d.shape[2] != 2:
                    print(f"    Invalid coordinate dimensions, skipping.")
                    continue
                
                # Ensure same number of frames
                if pose_3d.shape[0] != pose_2d.shape[0]:
                    print(f"    Frame count mismatch: 3D={pose_3d.shape[0]}, 2D={pose_2d.shape[0]}")
                    min_frames = min(pose_3d.shape[0], pose_2d.shape[0])
                    pose_3d = pose_3d[:min_frames]
                    pose_2d = pose_2d[:min_frames]
                    print(f"    Truncated to {min_frames} frames")
                
                # Ensure exactly 28 joints
                pose_3d = ensure_joint_count(pose_3d, TARGET_JOINTS, 3)
                pose_2d = ensure_joint_count(pose_2d, TARGET_JOINTS, 2)
                
                if pose_3d is None or pose_2d is None:
                    print(f"    Failed to adjust to {TARGET_JOINTS} joints, skipping.")
                    continue
                
                # Check if adding this sequence would exceed the limit
                frames_in_seq = pose_3d.shape[0]
                if total_frames_processed + frames_in_seq > max_samples:
                    frames_to_take = max_samples - total_frames_processed
                    pose_3d = pose_3d[:frames_to_take]
                    pose_2d = pose_2d[:frames_to_take]
                    print(f"    Taking only {frames_to_take} frames to reach sample limit")
                
                # Validate data quality
                if not validate_pose_data(pose_3d, pose_2d):
                    print(f"    Data validation failed, skipping.")
                    continue
                
                poses_3d.append(pose_3d)
                poses_2d.append(pose_2d)
                
                total_frames_processed += pose_3d.shape[0]
                print(f"    ✓ Successfully loaded: {pose_3d.shape[0]} frames (total: {total_frames_processed})")
                
                if total_frames_processed >= max_samples:
                    print(f"    Reached sample limit, stopping at {total_frames_processed} frames")
                    break
                
            except Exception as e:
                print(f"    Error loading {annot_file}: {str(e)}")
                continue
        
        if total_frames_processed >= max_samples:
            break
    
    if not poses_3d or not poses_2d:
        print("ERROR: No poses were successfully loaded!")
        print("Please check:")
        print("1. Dataset path is correct")
        print("2. .mat files exist and contain pose data")
        print("3. MPI-INF-3DHP dataset format")
        return False
    
    try:
        # Concatenate all poses
        print(f"\nConcatenating pose data (total frames: {total_frames_processed})...")
        poses_3d_combined = np.concatenate(poses_3d, axis=0)
        poses_2d_combined = np.concatenate(poses_2d, axis=0)
        
        print(f"Final 3D poses shape: {poses_3d_combined.shape}")
        print(f"Final 2D poses shape: {poses_2d_combined.shape}")
        print(f"Total samples processed: {poses_3d_combined.shape[0]}")
        
        # Final validation
        assert poses_3d_combined.shape[1] == TARGET_JOINTS, f"3D joint count mismatch: {poses_3d_combined.shape[1]} != {TARGET_JOINTS}"
        assert poses_2d_combined.shape[1] == TARGET_JOINTS, f"2D joint count mismatch: {poses_2d_combined.shape[1]} != {TARGET_JOINTS}"
        assert poses_3d_combined.shape[2] == 3, f"3D coordinate count mismatch: {poses_3d_combined.shape[2]} != 3"
        assert poses_2d_combined.shape[2] == 2, f"2D coordinate count mismatch: {poses_2d_combined.shape[2]} != 2"
        
        # Save in Martinez format
        output_3d = 'data/data_3d_mpi_inf_3dhp.npz'
        output_2d = 'data/data_2d_mpi_inf_3dhp_gt.npz'
        
        # Create Martinez-compatible format
        data_3d = {
            'positions_3d': poses_3d_combined,
            'frame_count': poses_3d_combined.shape[0],
            'joint_count': poses_3d_combined.shape[1],
            'coord_count': poses_3d_combined.shape[2],
            'sample_limit': max_samples,
            'input_dim': TARGET_JOINTS * 2,
            'output_dim': TARGET_JOINTS * 3
        }
        
        data_2d = {
            'positions_2d': poses_2d_combined,
            'frame_count': poses_2d_combined.shape[0],
            'joint_count': poses_2d_combined.shape[1],
            'coord_count': poses_2d_combined.shape[2],
            'sample_limit': max_samples,
            'input_dim': TARGET_JOINTS * 2,
            'output_dim': TARGET_JOINTS * 3
        }
        
        np.savez_compressed(output_3d, **data_3d)
        np.savez_compressed(output_2d, **data_2d)
        
        print(f"\nSuccessfully saved:")
        print(f"- 3D poses: {output_3d}")
        print(f"- 2D poses: {output_2d}")
        
        # Verify saved files
        print("\nVerifying saved files...")
        test_3d = np.load(output_3d)
        test_2d = np.load(output_2d)
        
        print(f"3D file keys: {list(test_3d.keys())}")
        print(f"2D file keys: {list(test_2d.keys())}")
        print(f"3D positions shape: {test_3d['positions_3d'].shape}")
        print(f"2D positions shape: {test_2d['positions_2d'].shape}")
        print(f"Input dimension: {test_2d['input_dim']}")
        print(f"Output dimension: {test_3d['output_dim']}")
        
        # Additional validation for model compatibility
        flattened_2d = test_2d['positions_2d'].reshape(-1, TARGET_JOINTS * 2)
        flattened_3d = test_3d['positions_3d'].reshape(-1, TARGET_JOINTS * 3)
        
        print(f"\nModel compatibility check:")
        print(f"Input tensor shape (flattened 2D): {flattened_2d.shape}")
        print(f"Output tensor shape (flattened 3D): {flattened_3d.shape}")
        print(f"Expected input dim: 56, Actual: {flattened_2d.shape[1]}")
        print(f"Expected output dim: 84, Actual: {flattened_3d.shape[1]}")
        
        if flattened_2d.shape[1] == 56 and flattened_3d.shape[1] == 84:
            print("✓ Model dimension compatibility confirmed!")
        else:
            print("✗ Model dimension compatibility failed!")
            return False
        
        print(f"\n✓ Conversion completed successfully with {poses_3d_combined.shape[0]} samples!")
        print(f"✓ Format: {TARGET_JOINTS} joints per pose")
        print(f"✓ Input dimension: {TARGET_JOINTS * 2} (56)")
        print(f"✓ Output dimension: {TARGET_JOINTS * 3} (84)")
        
        return True
        
    except Exception as e:
        print(f"ERROR during concatenation/saving: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_mpi_inf_annotations(annot_file):
    """
    Load MPI-INF-3DHP annotations with multiple fallback methods
    
    Args:
        annot_file: Path to annotation .mat file
    
    Returns:
        tuple: (pose_3d, pose_2d) or (None, None) if failed
    """
    try:
        # Method 1: Standard scipy.io.loadmat
        print(f"    Trying standard loadmat...")
        annot = sio.loadmat(annot_file)
        
        print(f"    Available keys: {[k for k in annot.keys() if not k.startswith('__')]}")
        
        # Try to extract pose data from different possible keys
        pose_3d, pose_2d = extract_poses_from_mat(annot)
        
        if pose_3d is not None and pose_2d is not None:
            return pose_3d, pose_2d
            
    except Exception as e:
        print(f"    Standard loadmat failed: {str(e)}")
    
    try:
        # Method 2: Try with struct_as_record=False
        print(f"    Trying loadmat with struct_as_record=False...")
        annot = sio.loadmat(annot_file, struct_as_record=False, squeeze_me=True)
        
        pose_3d, pose_2d = extract_poses_from_mat(annot)
        
        if pose_3d is not None and pose_2d is not None:
            return pose_3d, pose_2d
            
    except Exception as e:
        print(f"    Alternative loadmat failed: {str(e)}")
    
    try:
        # Method 3: Try h5py for v7.3 MAT files
        print(f"    Trying h5py...")
        with h5py.File(annot_file, 'r') as f:
            print(f"    h5py keys: {list(f.keys())}")
            
            pose_3d, pose_2d = extract_poses_from_h5py(f)
            
            if pose_3d is not None and pose_2d is not None:
                return pose_3d, pose_2d
                
    except Exception as e:
        print(f"    h5py failed: {str(e)}")
    
    return None, None

def extract_poses_from_mat(annot):
    """
    Extract pose data from loaded MAT file
    
    Args:
        annot: Loaded MAT file dictionary
    
    Returns:
        tuple: (pose_3d, pose_2d) or (None, None) if failed
    """
    print(f"    Extracting poses from MAT data...")
    
    # Look for different possible key combinations
    possible_3d_keys = ['annot3', 'univ_annot3', 'poses_3d', 'pose3d']
    possible_2d_keys = ['annot2', 'poses_2d', 'pose2d']
    
    pose_3d_data = None
    pose_2d_data = None
    
    # Find 3D pose data
    for key in possible_3d_keys:
        if key in annot:
            pose_3d_data = annot[key]
            print(f"    Found 3D data in key: {key}, shape: {pose_3d_data.shape}")
            break
    
    # Find 2D pose data
    for key in possible_2d_keys:
        if key in annot:
            pose_2d_data = annot[key]
            print(f"    Found 2D data in key: {key}, shape: {pose_2d_data.shape}")
            break
    
    if pose_3d_data is None or pose_2d_data is None:
        print(f"    Could not find pose data in expected keys")
        return None, None
    
    # Process the data based on MPI-INF-3DHP structure
    try:
        # MPI-INF-3DHP often stores data as cell arrays or structured arrays
        pose_3d = process_mpi_inf_data(pose_3d_data, 3, "3D")
        pose_2d = process_mpi_inf_data(pose_2d_data, 2, "2D")
        
        return pose_3d, pose_2d
        
    except Exception as e:
        print(f"    Error processing pose data: {str(e)}")
        return None, None

def extract_poses_from_h5py(f):
    """
    Extract pose data from h5py file object
    
    Args:
        f: h5py file object
    
    Returns:
        tuple: (pose_3d, pose_2d) or (None, None) if failed
    """
    print(f"    Extracting poses from HDF5 data...")
    
    pose_3d_data = None
    pose_2d_data = None
    
    # Look for pose data in h5py structure
    if 'annot3' in f:
        pose_3d_data = np.array(f['annot3'])
    elif 'univ_annot3' in f:
        pose_3d_data = np.array(f['univ_annot3'])
    
    if 'annot2' in f:
        pose_2d_data = np.array(f['annot2'])
    
    if pose_3d_data is None or pose_2d_data is None:
        return None, None
    
    try:
        pose_3d = process_mpi_inf_data(pose_3d_data, 3, "3D")
        pose_2d = process_mpi_inf_data(pose_2d_data, 2, "2D")
        
        return pose_3d, pose_2d
        
    except Exception as e:
        print(f"    Error processing h5py pose data: {str(e)}")
        return None, None

def process_mpi_inf_data(data, expected_coords, data_type):
    """
    Process MPI-INF-3DHP specific data structure
    
    Args:
        data: Raw data from MPI-INF-3DHP
        expected_coords: Expected coordinate count (2 or 3)
        data_type: "2D" or "3D" for logging
    
    Returns:
        Processed pose data in shape (frames, joints, coords) or None if failed
    """
    print(f"    Processing {data_type} MPI-INF data, shape: {data.shape}")
    
    try:
        # Handle different MPI-INF-3DHP data structures
        
        # Case 1: Data is a cell array or object array
        if data.dtype == 'object':
            print(f"    Processing object array...")
            
            # Try to extract actual pose sequences from cell array
            poses_list = []
            
            if data.shape == (14, 1):  # Common MPI-INF structure
                # Each cell might contain pose data for different cameras/sequences
                for i in range(data.shape[0]):
                    cell_data = data[i, 0]
                    if cell_data is not None and hasattr(cell_data, 'shape'):
                        if len(cell_data.shape) >= 2:
                            processed = process_pose_sequence(cell_data, expected_coords)
                            if processed is not None:
                                poses_list.append(processed)
            
            elif data.shape[0] == 1:  # Single sequence
                cell_data = data[0]
                if hasattr(cell_data, 'shape') and len(cell_data.shape) >= 2:
                    processed = process_pose_sequence(cell_data, expected_coords)
                    if processed is not None:
                        poses_list.append(processed)
            
            if poses_list:
                # Concatenate all sequences
                result = np.concatenate(poses_list, axis=0)
                print(f"    ✓ Processed {data_type} object array: {result.shape}")
                return result
        
        # Case 2: Regular numeric data
        elif np.issubdtype(data.dtype, np.number):
            print(f"    Processing numeric array...")
            result = process_pose_sequence(data, expected_coords)
            if result is not None:
                print(f"    ✓ Processed {data_type} numeric array: {result.shape}")
                return result
        
        print(f"    Failed to process {data_type} data")
        return None
        
    except Exception as e:
        print(f"    Error processing {data_type} MPI-INF data: {str(e)}")
        return None

def process_pose_sequence(data, expected_coords):
    """
    Process a single pose sequence to standard format
    
    Args:
        data: Pose sequence data
        expected_coords: Expected coordinate count (2 or 3)
    
    Returns:
        Processed data in shape (frames, joints, coords) or None if failed
    """
    try:
        if len(data.shape) == 2:
            # Could be (frames, joints*coords) or (joints*coords, frames)
            if data.shape[1] % expected_coords == 0:
                # (frames, joints*coords)
                n_joints = data.shape[1] // expected_coords
                data = data.reshape(data.shape[0], n_joints, expected_coords)
            elif data.shape[0] % expected_coords == 0:
                # (joints*coords, frames)
                n_joints = data.shape[0] // expected_coords
                data = data.T.reshape(data.shape[1], n_joints, expected_coords)
            else:
                return None
        
        elif len(data.shape) == 3:
            # Determine the correct arrangement
            if data.shape[2] == expected_coords:
                # (frames, joints, coords) - correct format
                pass
            elif data.shape[1] == expected_coords:
                # (joints, coords, frames)
                data = data.transpose(2, 0, 1)
            elif data.shape[0] == expected_coords:
                # (coords, joints, frames)
                data = data.transpose(2, 1, 0)
            else:
                return None
        
        else:
            return None
        
        # Validate final shape
        if len(data.shape) == 3 and data.shape[2] == expected_coords and data.shape[0] > 0:
            return data
        
        return None
        
    except Exception:
        return None

def ensure_joint_count(pose_data, target_joints, coords):
    """
    Ensure pose data has exactly the target number of joints
    """
    current_joints = pose_data.shape[1]
    
    if current_joints == target_joints:
        return pose_data
    
    print(f"      Adjusting joint count from {current_joints} to {target_joints}")
    
    if current_joints > target_joints:
        # Truncate to target joints
        return pose_data[:, :target_joints, :]
    else:
        # Pad with zeros
        frames, _, coords_actual = pose_data.shape
        padding_needed = target_joints - current_joints
        padding = np.zeros((frames, padding_needed, coords_actual))
        return np.concatenate([pose_data, padding], axis=1)

def validate_pose_data(pose_3d, pose_2d):
    """
    Validate pose data for common issues
    """
    try:
        # Check for NaN or infinite values
        if np.any(np.isnan(pose_3d)) or np.any(np.isnan(pose_2d)):
            print("      Warning: NaN values detected")
            return False
        
        if np.any(np.isinf(pose_3d)) or np.any(np.isinf(pose_2d)):
            print("      Warning: Infinite values detected")
            return False
        
        # Check for reasonable ranges
        if np.max(np.abs(pose_3d)) > 10000 or np.max(np.abs(pose_2d)) > 10000:
            print("      Warning: Very large values detected")
        
        return True
        
    except Exception as e:
        print(f"      Error during validation: {str(e)}")
        return False

def verify_dataset_structure():
    """
    Verify the MPI-INF-3DHP dataset structure
    """
    DATA_DIR = 'data/mpi_inf_3dhp'
    
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory {DATA_DIR} does not exist!")
        return False
    
    print(f"Dataset directory: {DATA_DIR}")
    print("Directory structure:")
    
    annot_files_found = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        
        if 'annot.mat' in files:
            annot_files_found += 1
        
        for file in files[:5]:
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    print(f"\nFound {annot_files_found} annotation files")
    
    if annot_files_found == 0:
        print("WARNING: No annotation files found!")
        return False
    
    return True

if __name__ == "__main__":
    print("MPI-INF-3DHP to Martinez Format Converter (Fixed for MPI-INF Structure)")
    print("=" * 75)
    
    SAMPLE_LIMIT = 20000
    print(f"Sample limit: {SAMPLE_LIMIT}")
    print(f"Target format: 28 joints per pose")
    print(f"Input dimension: 56 (28 joints × 2D)")
    print(f"Output dimension: 84 (28 joints × 3D)")
    
    if verify_dataset_structure():
        success = convert_mpi_inf_to_martinez(max_samples=SAMPLE_LIMIT)
        
        if success:
            print(f"\n" + "="*75)
            print("CONVERSION SUCCESSFUL!")
            print("="*75)
            print("✓ MPI-INF-3DHP dataset converted successfully")
            print("✓ Format: 28 joints per pose")
            print("✓ Dimensions: 56→84 (compatible with your model)")
        else:
            print(f"\n" + "="*75)
            print("CONVERSION FAILED!")
            print("="*75)
    else:
        print("Dataset verification failed!")