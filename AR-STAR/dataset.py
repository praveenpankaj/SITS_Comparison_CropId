#This is the Customary DataLoader class
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split

class SITSData(Dataset):
    def __init__(self, mode='train', transform=None):
        self.patch_list = []
        self.gt_list = []
        self.root = '/root/input/grapes-dataset' 
        self.mode = mode

        # this is a customary way of loading the data. Modify as requried.
        for path in os.listdir(self.root):
            cp = os.path.join(self.root, path)  # Full path

            if mode == 'train':
                if path[-1] in ['4']:
                    g = cp+'/'+'2022'+'/'
                    self.getfiles(g,'Grapes')
               
                if path[-1] in ['5']:
                    ng = cp+'/'+'2022'+'/'
                    self.getfiles(ng,'Non Grapes')    

            if mode == 'test':

                if path[-1] in ['2']:          
                    g = cp+'/'+'2022'+'/'
                    self.getfiles(g,'Grapes')
                if path[-1] in ['3']:
                    ng = cp + '/' + '2022' +'/'
                    self.getfiles(ng,'Non Grapes')
    
    # custom help function to load the data using file paths
    def getfiles(self, path, wanted):
    
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)  # Get full path
            
            if not os.path.isdir(cls_path):  # Ensure it's a directory
                continue
    
            if cls == wanted:
                for i,patch in enumerate(os.listdir(cls_path)):
                    # print(cls_path,patch)
                    self.patch_list.append(os.path.join(cls_path, patch))
                    self.gt_list.append(1 if wanted == "Grapes" else 0)  # Assign label based on class

                    if  i == 4892 and self.mode == 'test' and wanted == 'Non Grapes':
                        break

                    if  i == 100 and self.mode == 'train' and wanted == 'Grapes':
                        break

                    if  i == 2000 and self.mode == 'train' and wanted == 'Non Grapes':
                        break
                    
    def __len__(self):
        # Return the number of samples in the training dataset
        return len(self.patch_list)

    # pre-processing the images
    def reshape_sits_data(self,input_array):
   
        # Step 1: Convert each element from numpy.void to a numpy array of shape (280,)
        parsed_data = np.array(input_array.tolist(), dtype=np.float32)  # Convert all elements to float32

        # Step 2: Reshape from (24, 24, 280) → (24, 24, 28, 10)  [time, bands] 
        reshaped_data = parsed_data.reshape(24, 24, 10, 28)
    
        # Step 3: Rearrange to (28, 10, 24, 24)  [final shape] - in accordance with STAR input dim
        return np.transpose(reshaped_data, (3, 2, 0, 1))
    
    def __getitem__(self, idx):
        # Get a sample from the training data
        t = np.load(self.patch_list[idx])
        t = self.reshape_sits_data(t)
        t = t/10000.0                             #scaling the Satellite images.
        t = torch.tensor(t, dtype=torch.float32)  # Convert to tensor
        # Check gt_list[idx] and create the appropriate label tensor
        if self.gt_list[idx] == 1:
            label = torch.ones((24, 24), dtype=torch.float32)  # Tensor of shape (24,24) with all ones
        else:
            label = torch.zeros((24, 24), dtype=torch.float32)  # Tensor of shape (24,24) with all zeros
    
        return t, label