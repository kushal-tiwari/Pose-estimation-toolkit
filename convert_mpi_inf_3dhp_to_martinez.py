import os
import scipy.io as sio
import numpy as np

def convert_mpi_inf_to_martinez():
    """
    Convert MPI-INF-3DHP dataset to Martinez format
    """
    # Ensure 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    DATA_DIR = 'data/mpi_inf_3dhp'
    SUBJECTS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    
    poses_3d = []
    poses_2d = []
    
    print("Starting MPI-INF-3DHP dataset conversion...")
    
    for subj in SUBJECTS:
        subj_dir = os.path.join(DATA_DIR, subj)
        
        if not os.path.exists(subj_dir):
            print(f"Subject directory {subj_dir} does not exist, skipping.")
            continue
        
        print(f"Processing subject: {subj}")
        
        # Get all sequence directories (should be numbered directories like Seq1, Seq2, etc.)
        sequences = [d for d in os.listdir(subj_dir) 
                    if os.path.isdir(os.path.join(subj_dir, d))]
        
        if not sequences:
            print(f"No sequences found for subject {subj}")
            continue
        
        for seq in sequences:
            seq_dir = os.path.join(subj_dir, seq)
            annot_file = os.path.join(seq_dir, 'annot.mat')
            
            if not os.path.exists(annot_file):
                print(f"Annotation file {annot_file} not found, skipping.")
                continue
            
            try:
                print(f"  Loading {annot_file}")
                annot = sio.loadmat(annot_file)
                
                # Print available keys for debugging
                print(f"    Available keys: {list(annot.keys())}")
                
                # Common key variations in MPI-INF-3DHP
                pose_3d_key = None
                pose_2d_key = None
                
                # Check for different possible key names
                for key in annot.keys():
                    if '3d' in key.lower() or 'annot3' in key.lower():
                        pose_3d_key = key
                    elif '2d' in key.lower() or 'annot2' in key.lower():
                        pose_2d_key = key
                
                if pose_3d_key is None or pose_2d_key is None:
                    print(f"    Required 3D/2D pose keys not found in {annot_file}")
                    print(f"    Available keys: {[k for k in annot.keys() if not k.startswith('__')]}")
                    continue
                
                pose_3d = annot[pose_3d_key]
                pose_2d = annot[pose_2d_key]
                
                print(f"    3D poses shape: {pose_3d.shape}")
                print(f"    2D poses shape: {pose_2d.shape}")
                
                # Validate data shapes
                if len(pose_3d.shape) < 2 or len(pose_2d.shape) < 2:
                    print(f"    Invalid pose data shape, skipping.")
                    continue
                
                # Handle different data arrangements
                # MPI-INF-3DHP might have data in different formats
                # Common formats: (frames, joints, coords) or (joints, coords, frames)
                
                # Ensure consistent format: (frames, joints, coords)
                if pose_3d.shape[-1] == 3:  # Last dimension is coordinates
                    if len(pose_3d.shape) == 3:
                        # Already in correct format (frames, joints, 3)
                        pass
                    elif len(pose_3d.shape) == 2:
                        # Reshape if needed (joints*3, frames) -> (frames, joints, 3)
                        if pose_3d.shape[0] % 3 == 0:
                            n_joints = pose_3d.shape[0] // 3
                            pose_3d = pose_3d.T.reshape(-1, n_joints, 3)
                else:
                    # Data might be in (joints, coords, frames) format
                    if len(pose_3d.shape) == 3 and pose_3d.shape[1] == 3:
                        pose_3d = pose_3d.transpose(2, 0, 1)  # (frames, joints, 3)
                
                # Similar handling for 2D poses
                if pose_2d.shape[-1] == 2:  # Last dimension is coordinates
                    if len(pose_2d.shape) == 3:
                        # Already in correct format (frames, joints, 2)
                        pass
                    elif len(pose_2d.shape) == 2:
                        # Reshape if needed
                        if pose_2d.shape[0] % 2 == 0:
                            n_joints = pose_2d.shape[0] // 2
                            pose_2d = pose_2d.T.reshape(-1, n_joints, 2)
                else:
                    # Data might be in (joints, coords, frames) format
                    if len(pose_2d.shape) == 3 and pose_2d.shape[1] == 2:
                        pose_2d = pose_2d.transpose(2, 0, 1)  # (frames, joints, 2)
                
                # Validate final shapes
                if pose_3d.shape[0] != pose_2d.shape[0]:
                    print(f"    Mismatch in frame count: 3D={pose_3d.shape[0]}, 2D={pose_2d.shape[0]}")
                    min_frames = min(pose_3d.shape[0], pose_2d.shape[0])
                    pose_3d = pose_3d[:min_frames]
                    pose_2d = pose_2d[:min_frames]
                    print(f"    Truncated to {min_frames} frames")
                
                poses_3d.append(pose_3d)
                poses_2d.append(pose_2d)
                
                print(f"    Successfully loaded: {pose_3d.shape[0]} frames")
                
            except Exception as e:
                print(f"    Error loading {annot_file}: {str(e)}")
                continue
    
    if not poses_3d or not poses_2d:
        print("ERROR: No poses were successfully loaded!")
        print("Please check:")
        print("1. Dataset path is correct")
        print("2. .mat files exist and contain pose data")
        print("3. Key names in .mat files")
        return False
    
    try:
        # Concatenate all poses
        print("\nConcatenating all pose data...")
        poses_3d_combined = np.concatenate(poses_3d, axis=0)
        poses_2d_combined = np.concatenate(poses_2d, axis=0)
        
        print(f"Total 3D poses shape: {poses_3d_combined.shape}")
        print(f"Total 2D poses shape: {poses_2d_combined.shape}")
        
        # Save in Martinez format
        # Martinez format expects specific structure
        output_3d = 'data/data_3d_mpi_inf_3dhp.npz'
        output_2d = 'data/data_2d_mpi_inf_3dhp_gt.npz'
        
        # For Martinez format, we might need to structure data differently
        # Create a dictionary similar to Human3.6M format
        data_3d = {
            'positions_3d': poses_3d_combined,
            # Add metadata if needed
            'frame_count': poses_3d_combined.shape[0],
            'joint_count': poses_3d_combined.shape[1]
        }
        
        data_2d = {
            'positions_2d': poses_2d_combined,
            'frame_count': poses_2d_combined.shape[0],
            'joint_count': poses_2d_combined.shape[1]
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
        
        print("\n✓ Conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during concatenation/saving: {str(e)}")
        return False

def verify_dataset_structure():
    """
    Verify the MPI-INF-3DHP dataset structure
    """
    DATA_DIR = 'data/mpi_inf_3dhp'
    
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory {DATA_DIR} does not exist!")
        print("Please ensure you have downloaded and extracted the MPI-INF-3DHP dataset.")
        return False
    
    print(f"Dataset directory: {DATA_DIR}")
    print("Directory structure:")
    
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return True

if __name__ == "__main__":
    print("MPI-INF-3DHP to Martinez Format Converter")
    print("=" * 50)
    
    # First verify dataset structure
    if verify_dataset_structure():
        # Then attempt conversion
        success = convert_mpi_inf_to_martinez()
        
        if success:
            print("\nNext steps:")
            print("1. Verify the saved .npz files in the 'data' directory")
            print("2. Check joint ordering matches Martinez expectations")
            print("3. Modify Martinez code to load these files for training/evaluation")
            print("4. You may need to adjust joint mapping between MPI-INF and H36M formats")
        else:
            print("\nConversion failed. Please check the error messages above.")
    else:
        print("\nDataset verification failed. Please check your dataset path.")