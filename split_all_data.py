import os
import numpy as np
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

orig_data_dir = os.path.join(os.getenv('ORIGINAL_DATA_DIR'))
data_types = ['CurveVel_A',
 'CurveFault_A',
 'Style_A',
 'FlatVel_B',
 'FlatFault_B',
 'CurveVel_B',
 'Style_B',
 'CurveFault_B',
 'FlatVel_A',
 'FlatFault_A']
save_dir = os.getenv('SPLIT_DATA_DIR')

def parse_common_fault_files(d):
    fault_files = os.listdir(d)

    fault_files_x = [x for x in fault_files if 'seis' in x]
    x_ids = set([x.split('.')[0][4:] for x in fault_files_x])
    fault_files_x = sorted(fault_files_x)

    fault_files_y = [x for x in fault_files if 'vel' in x]
    y_ids = set([y.split('.')[0][3:] for y in fault_files_y])
    fault_files_y = sorted(fault_files_y)

    common_ids = x_ids.intersection(y_ids)

    fault_files_x = [os.path.join(d,x) for x in fault_files_x if x.split('.')[0][4:] in common_ids]
    fault_files_y = [os.path.join(d,y) for y in fault_files_y if y.split('.')[0][3:] in common_ids]

    return fault_files_x, fault_files_y

def parse_reg_files(d):
    x_dir = os.path.join(d,'data')
    reg_files_x = [(x, x.split('.')[0][4:]) for x in os.listdir(x_dir)]
    reg_files_x = sorted(reg_files_x, key=lambda x: int(x[1]))
    reg_files_x = [os.path.join(x_dir,f[0]) for f in reg_files_x]

    y_dir = os.path.join(d,'model')
    reg_files_y = [(x, x.split('.')[0][5:]) for x in os.listdir(y_dir)]
    reg_files_y = sorted(reg_files_y, key=lambda x: int(x[1]))
    reg_files_y = [os.path.join(y_dir,f[0]) for f in reg_files_y]

    return reg_files_x, reg_files_y

# Let's split all the data into chunks, identified by the current filename prefixed by the data type, and save them in a directory
import gc


def extract_numerical(filename):
    return int(''.join(filter(str.isdigit, filename)))

def create_data_id(data_type, id):
    # Create a unique identifier for the data based on the data type and file name
    return f"{data_type}_{id}"

def save_data_chunks(save_dir, data_types, root_dir):
    os.makedirs(save_dir, exist_ok=True)
    for data_type in data_types:
        print(f"Loading data from {data_type}")
        ex_dir = os.path.join(root_dir, data_type)
        if 'Fault' in data_type:
            exs_x, exs_y = parse_common_fault_files(ex_dir)
        else:
            exs_x, exs_y = parse_reg_files(ex_dir)

        for i in range(len(exs_x)):
            assert extract_numerical(exs_x[i]) == extract_numerical(exs_y[i]), f"File names do not match: {exs_x[i]} and {exs_y[i]}"
        # Load the data
        
        for i, (ex_x, ex_y) in enumerate(zip(exs_x, exs_y)):
            print(f"Loading {i} out of {len(exs_x)}")
            # Concatenate the data
            x = np.load(ex_x)
            y = np.load(ex_y)
            gc.collect()

            # Split the data into 500 chunks
            chunk_size = 1
            num_chunks = x.shape[0] // chunk_size

            # Save each chunk as a separate .npy file
            print(f"Splitting data into {num_chunks} chunks of size {chunk_size}")
            for i in tqdm(range(num_chunks)):
                chunk_x = x[i*chunk_size:(i+1)*chunk_size].squeeze()
                chunk_y = y[i*chunk_size:(i+1)*chunk_size].squeeze()
                assert chunk_x.shape == (5, 1000, 70)
                assert chunk_y.shape == (70, 70)

                # Create a directory for this chunk if it doesn't exist
                chunk_id = f'{data_type}/{extract_numerical(ex_x)}/{i}'
                chunk_dir = os.path.join(save_dir, chunk_id)
                os.makedirs(chunk_dir, exist_ok=True)

                # Save the chunk data
                np.save(os.path.join(chunk_dir, 'x.npy'), chunk_x)
                np.save(os.path.join(chunk_dir, 'y.npy'), chunk_y)
                
if __name__ == "__main__":
    save_data_chunks(save_dir, data_types, orig_data_dir)