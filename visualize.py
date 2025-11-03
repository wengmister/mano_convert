# visualize.py
import numpy as np
import trimesh

mano_data = np.load('MANO_RIGHT_clean.npy', allow_pickle=True).item()

mesh = trimesh.Trimesh(
    vertices=mano_data['v_template'],
    faces=mano_data['f']
)
mesh.show()