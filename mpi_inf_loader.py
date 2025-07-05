import numpy as np
import os

def load_mpi_inf_3dhp(data_dir='data', max_samples=20000):
    data_3d_path = os.path.join(data_dir, 'data_3d_mpi_inf_3dhp.npz')
    data_2d_path = os.path.join(data_dir, 'data_2d_mpi_inf_3dhp_gt.npz')

    if not os.path.exists(data_3d_path) or not os.path.exists(data_2d_path):
        raise FileNotFoundError("NPZ files not found in provided directory.")

    print(f"Loading MPI-INF-3DHP data from {data_dir} (max {max_samples} samples)...")

    data_3d = np.load(data_3d_path, allow_pickle=True)
    data_2d = np.load(data_2d_path, allow_pickle=True)

    if 'sample_limit' in data_3d.keys():
        print(f"Data was pre-limited during conversion to {data_3d['sample_limit']} samples")
    
    positions_3d_full = data_3d['positions_3d']
    positions_2d_full = data_2d['positions_2d']

    # Fix: handle object arrays of shape (N, 1)
    if positions_3d_full.dtype == object and positions_3d_full.shape[1] == 1:
        print("Converting 3D data from object array to stacked NumPy array")
        positions_3d_full = np.vstack(positions_3d_full[:, 0])

    if positions_2d_full.dtype == object and positions_2d_full.shape[1] == 1:
        print("Converting 2D data from object array to stacked NumPy array")
        positions_2d_full = np.vstack(positions_2d_full[:, 0])

    print(f"Full dataset shapes - 3D: {positions_3d_full.shape}, 2D: {positions_2d_full.shape}")

    if positions_3d_full.shape[0] > max_samples:
        print(f"Limiting dataset from {positions_3d_full.shape[0]} to {max_samples} samples")
        np.random.seed(42)
        indices = np.random.choice(positions_3d_full.shape[0], size=max_samples, replace=False)
        indices = np.sort(indices)
        positions_3d_full = positions_3d_full[indices]
        positions_2d_full = positions_2d_full[indices]
        print(f"Sampled {max_samples} random frames from the dataset")
    else:
        print(f"Dataset size ({positions_3d_full.shape[0]}) is within limit ({max_samples})")

    # ✅ FIXED: Changed from 17 to 28 joints
    expected_joints = 28
    expected_3d_coords_per_joint = 3
    expected_2d_coords_per_joint = 2

    positions_3d_all_frames, positions_2d_all_frames = [], []
    subjects_all_frames, actions_all_frames = [], []

    if len(positions_3d_full.shape) == 3 and len(positions_2d_full.shape) == 3:
        print("Data is already in proper 3D format")
        current_joints_3d = positions_3d_full.shape[1]
        current_joints_2d = positions_2d_full.shape[1]

        print(f"Current joint count: 3D={current_joints_3d}, 2D={current_joints_2d}")

        # ✅ FIXED: Now handles both cases - take first 28 joints OR pad if fewer
        if current_joints_3d >= expected_joints and current_joints_2d >= expected_joints:
            positions_3d_final = positions_3d_full[:, :expected_joints, :]
            positions_2d_final = positions_2d_full[:, :expected_joints, :]
            print(f"Using first {expected_joints} joints")
        elif current_joints_3d < expected_joints or current_joints_2d < expected_joints:
            print(f"Warning: Dataset has fewer joints than expected ({expected_joints})")
            print(f"Padding with zeros to reach {expected_joints} joints")
            
            # Pad 3D data
            if current_joints_3d < expected_joints:
                pad_joints_3d = expected_joints - current_joints_3d
                pad_shape_3d = (positions_3d_full.shape[0], pad_joints_3d, 3)
                padding_3d = np.zeros(pad_shape_3d)
                positions_3d_final = np.concatenate([positions_3d_full, padding_3d], axis=1)
            else:
                positions_3d_final = positions_3d_full[:, :expected_joints, :]
            
            # Pad 2D data
            if current_joints_2d < expected_joints:
                pad_joints_2d = expected_joints - current_joints_2d
                pad_shape_2d = (positions_2d_full.shape[0], pad_joints_2d, 2)
                padding_2d = np.zeros(pad_shape_2d)
                positions_2d_final = np.concatenate([positions_2d_full, padding_2d], axis=1)
            else:
                positions_2d_final = positions_2d_full[:, :expected_joints, :]
        else:
            positions_3d_final = positions_3d_full
            positions_2d_final = positions_2d_full

        num_samples = positions_3d_final.shape[0]
        subjects_all_frames = list(range(num_samples // 1000 + 1)) * (num_samples // (num_samples // 1000 + 1) + 1)
        subjects_all_frames = subjects_all_frames[:num_samples]
        actions_all_frames = ['unknown'] * num_samples

    else:
        print("Data needs format processing...")
        if len(positions_3d_full.shape) == 2:
            num_samples = positions_3d_full.shape[0]
            coords_per_sample_3d = positions_3d_full.shape[1]
            coords_per_sample_2d = positions_2d_full.shape[1]

            if coords_per_sample_3d % 3 == 0 and coords_per_sample_2d % 2 == 0:
                joints_3d = coords_per_sample_3d // 3
                joints_2d = coords_per_sample_2d // 2

                print(f"Inferred joints - 3D: {joints_3d}, 2D: {joints_2d}")

                positions_3d_reshaped = positions_3d_full.reshape(num_samples, joints_3d, 3)
                positions_2d_reshaped = positions_2d_full.reshape(num_samples, joints_2d, 2)

                # ✅ FIXED: Handle padding/truncation for 28 joints
                if joints_3d >= expected_joints and joints_2d >= expected_joints:
                    positions_3d_final = positions_3d_reshaped[:, :expected_joints, :]
                    positions_2d_final = positions_2d_reshaped[:, :expected_joints, :]
                elif joints_3d < expected_joints or joints_2d < expected_joints:
                    print(f"Padding reshaped data to {expected_joints} joints")
                    
                    # Pad 3D
                    if joints_3d < expected_joints:
                        pad_joints_3d = expected_joints - joints_3d
                        pad_shape_3d = (num_samples, pad_joints_3d, 3)
                        padding_3d = np.zeros(pad_shape_3d)
                        positions_3d_final = np.concatenate([positions_3d_reshaped, padding_3d], axis=1)
                    else:
                        positions_3d_final = positions_3d_reshaped[:, :expected_joints, :]
                    
                    # Pad 2D
                    if joints_2d < expected_joints:
                        pad_joints_2d = expected_joints - joints_2d
                        pad_shape_2d = (num_samples, pad_joints_2d, 2)
                        padding_2d = np.zeros(pad_shape_2d)
                        positions_2d_final = np.concatenate([positions_2d_reshaped, padding_2d], axis=1)
                    else:
                        positions_2d_final = positions_2d_reshaped[:, :expected_joints, :]
                else:
                    positions_3d_final = positions_3d_reshaped
                    positions_2d_final = positions_2d_reshaped

                subjects_all_frames = list(range(num_samples // 1000 + 1)) * (num_samples // (num_samples // 1000 + 1) + 1)
                subjects_all_frames = subjects_all_frames[:num_samples]
                actions_all_frames = ['unknown'] * num_samples
            else:
                raise ValueError(f"Unable to properly reshape data: 3D shape {positions_3d_full.shape}, 2D shape {positions_2d_full.shape}")
        else:
            raise ValueError(f"Unexpected data format: 3D shape {positions_3d_full.shape}, 2D shape {positions_2d_full.shape}")

    subjects = np.array(subjects_all_frames, dtype=int)
    actions = actions_all_frames

    if positions_3d_final.shape[0] != positions_2d_final.shape[0]:
        min_samples = min(positions_3d_final.shape[0], positions_2d_final.shape[0])
        positions_3d_final = positions_3d_final[:min_samples]
        positions_2d_final = positions_2d_final[:min_samples]
        subjects = subjects[:min_samples]
        actions = actions[:min_samples]
        print(f"Adjusted to {min_samples} samples due to shape mismatch")

    print(f"Final loaded samples: {len(positions_3d_final)}")
    print(f"Final 3D shape: {positions_3d_final.shape}")
    print(f"Final 2D shape: {positions_2d_final.shape}")
    print(f"Subjects range: {np.min(subjects)} to {np.max(subjects)}")
    
    # ✅ VERIFICATION: Check dimensions match expected
    actual_input_dim = positions_2d_final.shape[1] * positions_2d_final.shape[2]
    actual_output_dim = positions_3d_final.shape[1] * positions_3d_final.shape[2]
    print(f"\n🔍 DIMENSION CHECK:")
    print(f"   Expected input dim: 56 (28 joints × 2)")
    print(f"   Actual input dim: {actual_input_dim} ({positions_2d_final.shape[1]} joints × {positions_2d_final.shape[2]})")
    print(f"   Expected output dim: 84 (28 joints × 3)")
    print(f"   Actual output dim: {actual_output_dim} ({positions_3d_final.shape[1]} joints × {positions_3d_final.shape[2]})")
    
    if actual_input_dim == 56 and actual_output_dim == 84:
        print("   ✅ Dimensions are CORRECT!")
    else:
        print(f"   ❌ Dimensions are INCORRECT!")

    return positions_2d_final, positions_3d_final, subjects, actions


def prepare_mpi_inf_for_training(positions_2d, positions_3d, subjects, actions,
                                 train_subjects=None, test_subjects=None):
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)

    if train_subjects is None or test_subjects is None:
        split_idx = int(0.8 * n_subjects)
        train_subjects = unique_subjects[:split_idx] if train_subjects is None else train_subjects
        test_subjects = unique_subjects[split_idx:] if test_subjects is None else test_subjects

    print(f"Train subjects: {train_subjects}")
    print(f"Test subjects: {test_subjects}")

    train_mask = np.isin(subjects, train_subjects)
    test_mask = np.isin(subjects, test_subjects)

    data = {
        'train': {
            'positions_2d': positions_2d[train_mask],
            'positions_3d': positions_3d[train_mask],
            'subjects': subjects[train_mask],
            'actions': [actions[i] for i in np.where(train_mask)[0]]
        },
        'test': {
            'positions_2d': positions_2d[test_mask],
            'positions_3d': positions_3d[test_mask],
            'subjects': subjects[test_mask],
            'actions': [actions[i] for i in np.where(test_mask)[0]]
        }
    }

    print(f"Training samples: {len(data['train']['positions_2d'])}")
    print(f"Test samples: {len(data['test']['positions_2d'])}")

    return data


def normalize_mpi_inf_data(positions_2d, positions_3d, image_width=2048, image_height=2048):
    positions_2d_norm = positions_2d.copy().astype(np.float32)
    positions_2d_norm[:, :, 0] = 2.0 * positions_2d_norm[:, :, 0] / image_width - 1.0
    positions_2d_norm[:, :, 1] = 2.0 * positions_2d_norm[:, :, 1] / image_height - 1.0

    positions_3d_norm = positions_3d.copy().astype(np.float32)
    root_joint = 0
    positions_3d_norm = positions_3d_norm - positions_3d_norm[:, root_joint:root_joint+1, :]
    positions_3d_norm = positions_3d_norm / 1000.0

    return positions_2d_norm, positions_3d_norm


def get_mpi_inf_data(data_dir='data', normalize=True, max_samples=20000):
    positions_2d, positions_3d, subjects, actions = load_mpi_inf_3dhp(data_dir, max_samples)

    if normalize:
        positions_2d, positions_3d = normalize_mpi_inf_data(positions_2d, positions_3d)

    data = prepare_mpi_inf_for_training(positions_2d, positions_3d, subjects, actions)

    return data


if __name__ == "__main__":
    try:
        SAMPLE_LIMIT = 20000
        print(f"Loading MPI-INF-3DHP data with sample limit: {SAMPLE_LIMIT}")
        data = get_mpi_inf_data('data', max_samples=SAMPLE_LIMIT)
        print("\n=== Data Loading Successful ===")
        print(f"Train 2D shape: {data['train']['positions_2d'].shape}")
        print(f"Train 3D shape: {data['train']['positions_3d'].shape}")
        print(f"Test 2D shape: {data['test']['positions_2d'].shape}")
        print(f"Test 3D shape: {data['test']['positions_3d'].shape}")

        print("\nData quality check:")
        print(f"Train 2D NaNs: {np.isnan(data['train']['positions_2d']).sum()}")
        print(f"Train 3D NaNs: {np.isnan(data['train']['positions_3d']).sum()}")
        print(f"\nTotal samples loaded: {len(data['train']['positions_2d']) + len(data['test']['positions_2d'])}")
        print(f"Sample limit was: {SAMPLE_LIMIT}")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure your .npz files are in the 'data' directory and their structure matches the script's expectations.")