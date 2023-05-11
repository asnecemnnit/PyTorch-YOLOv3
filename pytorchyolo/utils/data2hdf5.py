import os
import numpy as np
import pickle
from pathlib import Path
import mmcv
import cv2
import h5py


#This code ist based on https://blade6570.github.io/soumyatripathy/hdf5_blog.html
#and https://stackoverflow.com/questions/66631284/convert-a-folder-comprising-jpeg-images-to-hdf5/66641176#66641176

def search_files(parent_folder, suffix):
    paths = list(filter(lambda x: x.suffix==suffix , Path(parent_folder).rglob("**/*")))
    num_files = len(paths)
    assert num_files > 0, "No samples found in %s" % parent_folder
    return paths, num_files


def read_hdf5(hdf5_file, img_path):
    hf = h5py.File(hdf5_file, 'r') # open a hdf5 file
    img_bytes = np.array(hf[str(img_path)]) # read image in bytes
    img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb') # convert to array
    return img

if __name__ == '__main__':
     
    # Create results directory
    #hdf5_path = Path("./data/nuscenes/")
    #hdf5_path.mkdir(parents=True, exist_ok=True)
    #hdf5_path = hdf5_path.as_posix()
    
    
    # Look into the pkl files to see how the dataset is structured:
    opened_pickle = open("data/nuscenes/nuscenes_infos_temporal_train.pkl", "rb")
    loaded_train_pickle = pickle.load(opened_pickle)
    opened_pickle = open("data/nuscenes/nuscenes_infos_temporal_val.pkl", "rb")
    loaded_val_pickle = pickle.load(opened_pickle)   
    
    
    # get number of samples and a list of filepaths in data folder
    data_folder = "data"
    sample_paths, num_samples = search_files(data_folder, '.jpg')
    

    save_path = './data/nuscenes/data.hdf5'
    
    index =1
    progress_frequency = 50
    with h5py.File(save_path, 'w') as hf:
        
        for img_path in sample_paths[:30]: # save only the first 30 images for testing
            
            binary_img = open(str(img_path), 'rb').read()
            binary_img_np = np.asarray(binary_img)
            
            dset = hf.create_dataset(str(img_path), data=binary_img_np)
            if index % progress_frequency == 0:
                print("[%d/%d]" % (index, num_samples))
            index = index + 1

    print('hdf5 file size: %d bytes'%os.path.getsize(save_path))
    

    hdf5_file ='./data/nuscenes/data.hdf5'
    
    # read data from hf5 file and check size
    for img_path in sample_paths[:30]:
        img = read_hdf5(hdf5_file, img_path)
        print ('img.shape', img.shape)    
    
    cv2.imwrite(('./test.jpg'),img)
    
 
                 
