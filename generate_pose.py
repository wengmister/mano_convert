"""
MANO Hand Pose Generator - Proper implementation with LBS
Uses only numpy data, no chumpy at runtime
"""

import numpy as np
import trimesh

# Load the clean MANO data
mano_data = np.load('MANO_RIGHT_clean.npy', allow_pickle=True).item()

print("MANO data loaded successfully!")
print(f"Vertices: {mano_data['v_template'].shape}")
print(f"Joints: {mano_data['J'].shape}")
print(f"Weights: {mano_data['weights'].shape}")
print(f"Kinematic tree: {mano_data['kintree_table'].shape}")

def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle to rotation matrix"""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-7:
        return np.eye(3)
    
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

def generate_hand(pose_params=None, shape_params=None, use_pca=True):
    """
    Generate MANO hand with proper skeletal animation
    
    Args:
        pose_params: Either 6 PCA coeffs (if use_pca=True) or 45 axis-angle params
        shape_params: 10-dim shape parameters
        use_pca: If True, use PCA pose space (easier to control)
    
    Returns:
        trimesh.Trimesh
    """
    # Defaults
    if shape_params is None:
        shape_params = np.zeros(10)
    
    # Get MANO components
    v_template = mano_data['v_template']
    shapedirs = mano_data['shapedirs']
    posedirs = mano_data['posedirs']
    J_regressor = mano_data['J_regressor']
    weights = mano_data['weights']
    kintree_table = mano_data['kintree_table']
    faces = mano_data['f']
    
    # Apply shape blend shapes
    v_shaped = v_template + np.einsum('vij,j->vi', shapedirs, shape_params)
    
    # Get joint locations from shaped vertices
    J = np.dot(J_regressor, v_shaped)
    
    # Handle pose parameters
    if use_pca and pose_params is not None:
        # Convert PCA to full pose
        hands_mean = mano_data['hands_mean']
        hands_components = mano_data['hands_components']
        n_comps = min(len(pose_params), hands_components.shape[0])
        full_pose = hands_mean + np.dot(hands_components[:n_comps].T, pose_params[:n_comps])
    elif pose_params is not None:
        full_pose = pose_params
    else:
        full_pose = np.zeros(45)
    
    # Reshape to 15 joints x 3
    pose_rodrigues = full_pose.reshape(15, 3)
    
    # Add root orientation (typically zeros for hand-only)
    root_rot = np.zeros(3)
    
    # Convert all poses to rotation matrices
    rot_mats = []
    rot_mats.append(axis_angle_to_matrix(root_rot))  # Root
    for i in range(15):
        rot_mats.append(axis_angle_to_matrix(pose_rodrigues[i]))
    rot_mats = np.array(rot_mats)
    
    # Apply pose blend shapes
    # Compute pose feature (rotation matrices minus identity)
    # Only use hand joints (not root)
    pose_feature = (rot_mats[1:] - np.eye(3)).ravel()
    # posedirs is (778, 3, 135) where 135 = 15 joints * 9 (flattened 3x3 rot mat)
    # We only have 15*9 = 135 dimensions
    pose_blend = np.einsum('ijk,k->ij', posedirs[:, :, :len(pose_feature)], pose_feature)
    v_posed = v_shaped + pose_blend
    
    # === Forward Kinematics ===
    num_joints = len(J)
    G = np.zeros((num_joints, 4, 4))
    
    # Root joint
    G[0] = np.eye(4)
    G[0, :3, :3] = rot_mats[0]
    
    # Build global transformations using kinematic tree
    for i in range(1, num_joints):
        parent = kintree_table[0, i]
        
        # Local rotation at this joint
        local_rot = np.eye(4)
        local_rot[:3, :3] = rot_mats[i]
        
        # Translation from parent to child joint
        local_trans = np.eye(4)
        local_trans[:3, 3] = J[i] - J[parent]
        
        # Global transform = parent_global * translation * rotation
        G[i] = G[parent].dot(local_trans).dot(local_rot)
    
    # Apply inverse bind pose (subtract rest joint positions)
    G_final = np.zeros_like(G)
    for i in range(num_joints):
        inv_bind = np.eye(4)
        inv_bind[:3, 3] = -J[i]
        G_final[i] = G[i].dot(inv_bind)
    
    # === Linear Blend Skinning ===
    # Transform each vertex by weighted combination of joint transforms
    T = np.tensordot(weights, G_final, axes=([1], [0]))
    
    # Apply transformations to vertices
    homogeneous = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
    v_final = np.einsum('vij,vj->vi', T, homogeneous)[:, :3]
    
    return trimesh.Trimesh(vertices=v_final, faces=faces, process=False)


# ========================================
# Generate example poses using PCA
# ========================================

print("\n" + "="*60)
print("Generating hand poses using PCA coefficients...")
print("="*60 + "\n")

# Test different PCA combinations
pca_examples = {
    'flat': [0, 0, 0, 0, 0, 0],
    'pca1_small': [1, 0, 0, 0, 0, 0],
    'pca1_med': [2, 0, 0, 0, 0, 0],
    'pca1_large': [3, 0, 0, 0, 0, 0],
    'pca1_neg': [-2, 0, 0, 0, 0, 0],
    'pca2_pos': [0, 2, 0, 0, 0, 0],
    'pca2_neg': [0, -2, 0, 0, 0, 0],
    'pca3_pos': [0, 0, 2, 0, 0, 0],
    'pca4_pos': [0, 0, 0, 2, 0, 0],
    'pca5_pos': [0, 0, 0, 0, 2, 0],
    'pca6_pos': [0, 0, 0, 0, 0, 2],
    'combined1': [1, 1, 0, 0, 0, 0],
    'combined2': [1, -1, 0, 0, 0, 0],
    'combined3': [1, 0, 1, 0, 0, 0],
}

for name, pca in pca_examples.items():
    print(f"Generating {name}...")
    mesh = generate_hand(pose_params=np.array(pca), use_pca=True)
    mesh.export(f'hand_{name}.stl')
    print(f"  ✓ Saved hand_{name}.stl")

# Different hand shapes
print("\nGenerating size variations...")
for scale in [-2, 0, 2]:
    shape = np.zeros(10)
    shape[0] = scale
    mesh = generate_hand(shape_params=shape)
    mesh.export(f'hand_shape_{scale:+d}.stl')
    print(f"  ✓ Saved hand_shape_{scale:+d}.stl")

print("\n" + "="*60)
print("DONE! Generated STL files.")
print("Check the files - they should show different poses now!")
print("="*60)

# Show one example
print("\nOpening viewer for 'pca1_pos' example...")
mesh = generate_hand(pose_params=np.array([3, 0, 0, 0, 0, 0]), use_pca=True)
mesh.show()