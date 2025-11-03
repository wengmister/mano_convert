import pickle
import numpy as np


import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Convert MANO pkl to clean numpy format.')
    parser.add_argument('--type', choices=['left', 'right'], required=True, help="Type of MANO model: 'left' or 'right'")
    args = parser.parse_args()

    mano_type = args.type.lower()
    if mano_type == 'left':
        pkl_path = 'models/MANO_LEFT.pkl'
        npy_path = 'MANO_LEFT_clean.npy'
    elif mano_type == 'right':
        pkl_path = 'models/MANO_RIGHT.pkl'
        npy_path = 'MANO_RIGHT_clean.npy'
    else:
        print("Invalid type. Use 'left' or 'right'.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract all data as pure numpy
    clean_data = {
        'v_template': np.array(data['v_template']),
        'shapedirs': np.array(data['shapedirs']),
        'J_regressor': np.array(data['J_regressor'].toarray()),
        'kintree_table': np.array(data['kintree_table']),
        'weights': np.array(data['weights']),
        'posedirs': np.array(data['posedirs']),
        'f': np.array(data['f']),
        'hands_components': np.array(data['hands_components']),
        'hands_mean': np.array(data['hands_mean'])
    }

    # Save as clean numpy format
    np.save(npy_path, clean_data)
    print(f"Converted successfully! Saved to {npy_path}")

if __name__ == '__main__':
    main()
