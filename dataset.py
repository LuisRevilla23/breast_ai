import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
import skvideo.io
import scipy.io as sio
import cv2
import torchvision.transforms as transforms
import torchio as tio


class BreastDataset(Dataset):
    """ Dataset for lung 3D segmentation """

    def __init__(self, meta_data, root_dir, cache_data=False, transform=None, resize_to=(128, 128),out_channels=1):
        df = pd.read_csv(meta_data)
        self.wsi_list = df.wsi.to_list()
        self.root_dir = root_dir
        self.transform = transform
        self.resize_to = resize_to
        self.cache_data = cache_data
        self.imgs_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'video', '*.mp4')) for i in range(len(self.wsi_list))])
        self.gt_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'mask', '*.mat')) for i in range(len(self.wsi_list))])
        #self.imgs_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'frames', '*.png')) for i in range(len(self.wsi_list))])
        #self.gt_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'frames-mask', '*.png')) for i in range(len(self.wsi_list))])
       
        self.out_channels=out_channels
        self.total_frames = 0

        if self.cache_data:
            print("Caching data...")
            dataset_imgs = []
            dataset_gt = []

            self.dataset_imgs = [skvideo.io.vread(data) for data in self.imgs_path]
            self.dataset_gt = [sio.loadmat(gt)['labels'] for gt in self.gt_path]

            self.dataset_imgs = dataset_imgs.copy()
            self.dataset_gt = dataset_gt.copy()

    def __len__(self):
        return len(self.imgs_path)
    
    def load_video_as_gray(self, video_path):
        video = skvideo.io.vread(video_path)
        video_gray = np.zeros((video.shape[0], video.shape[1], video.shape[2]), dtype=np.uint8)
        for i in range(video.shape[0]):
            video_gray[i, :, :] = np.dot(video[i, :, :, :], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return np.expand_dims(video_gray, axis=1)  # Add a channel dimension for consistency with the RGB images


    def __getitem__(self, idx):
        if self.cache_data:
            image = self.dataset_imgs[idx]
            mask = self.dataset_gt[idx]
        else:
            image = self.load_video_as_gray(self.imgs_path[idx])
            mask = sio.loadmat(self.gt_path[idx])['labels']
            
        mask=mask[1:,:,:]

        # Ensure frames are in the same dimension
        image = np.transpose(image, (1, 0, 2, 3))  # Convert (T, 1, H, W)
        mask = np.transpose(mask, (2, 0, 1))       # Convert (T, H, W)
        
        # resize frames to 128x128
        image = np.array([cv2.resize(frame, self.resize_to) for frame in image[0]])
        mask = np.array([cv2.resize(frame, self.resize_to,interpolation = cv2.INTER_NEAREST) for frame in mask])

        mask[mask>3]=0
        mask[mask==3]=2

        resize = tio.transforms.Resample((4,1,1))
        intensity=tio.RescaleIntensity((0, 1))
        image=np.expand_dims(image,axis=0)
        mask=np.expand_dims(mask,axis=0)

        #subject = tio.Subject(image=tio.Image(tensor=image), label=tio.Image(tensor=mask))
        subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )
        subject = resize(subject)
        subject=intensity(subject)
        image= subject['image']['data']
        mask = subject['mask']['data']
        mask=torch.squeeze(mask)
        frame_indices = [i for i in range(mask.shape[0]) if torch.any(mask[i] > 0)]
        if frame_indices[0]+64 < mask.shape[0]:
            mask = mask[frame_indices[0]:frame_indices[0]+64]
            image = image[:,frame_indices[0]:frame_indices[0]+64]
        else:
            mask = mask[frame_indices[0]:]
            image = image[:,frame_indices[0]:]
            pad_size =64 - mask.shape[0]
            mask = np.pad(mask, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            image = np.pad(image[0], ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            image = np.expand_dims(image, axis=0)
            mask=torch.Tensor(mask)
            image=torch.Tensor(image)
        

        if self.transform:
            mask=torch.unsqueeze(mask,dim=0)
            # Crear un objeto 'Subject' que contenga la imagen y la máscara
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )

            subject = self.transform(subject)
            image = subject['image']['data']
            mask = subject['mask']['data']
            mask=torch.squeeze(mask)
        
        mask=mask.long()
        #mask=mask.float()
        #mask = torch.nn.functional.one_hot(mask,num_classes=self.out_channels).permute(3,0,1,2).float()

        return image, mask


class BreastDataset2D(Dataset):
    """
    Dataset that loads ALL .png images from each patient's folder
    as separate samples for binary classification.
    """

    def __init__(self, csv_file, data_dir, transform=None, resize_to=(224, 224)):
        """
        Args:
            csv_file (str): Path to a CSV containing at least:
                            - "Patient_ID"
                            - "SOC_binario" (with values "maligno" or "benigno")
            data_dir (str): Base directory where 'benign'/'malign' folders exist,
                            each containing patient subfolders with images.
            transform (callable, optional): Optional PyTorch-style transform
                                            to apply to each image tensor.
            resize_to (tuple): (width, height) size to resize images, e.g. (224, 224).
        """
        self.transform = transform
        self.resize_to = resize_to

        # Read the CSV with patients & label info
        df = pd.read_csv(csv_file)

        # We'll collect (image_path, label) tuples here
        self.samples = []

        # Go through each patient row in the CSV
        for i in range(len(df)):
            row = df.iloc[i]
            patient_id = row["Patient_ID"]
            label_str = row["SOC_binario"]  # "maligno" or "benigno"
            label = 1 if label_str == "maligno" else 0

            # Build the path to the patient's folder
            # e.g. data_dir/malign/<patient_id>/cropped_p2/*.png
            if label == 0:
                patient_id_str = str(patient_id)  # Convert to string
                folder_path = os.path.join(data_dir, "benign", patient_id_str, "cropped_p2")
            else:
                patient_id_str = str(patient_id)  # Convert to string
                folder_path = os.path.join(data_dir, "malign", patient_id_str, "cropped_p2")

            # Collect ALL .png files in that folder
            image_files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg"))
            if not image_files:
                print(f"Warning: no .png or .jpg files found for patient {patient_id} at {folder_path}")
                continue

            # Each .png is a separate training sample
            for image_path in image_files:
                self.samples.append((image_path, label, patient_id))  # Incluir patient_id

        print(f"[BreastDataset2D] Found {len(self.samples)} total images across all patients.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]

        # Read the image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        image = cv2.resize(image, self.resize_to)  # [H, W]

        image_tensor = torch.tensor(image).unsqueeze(0).float() / 255.0  # shape [1, H, W]

        # Repeat along channel dimension => shape [3, H, W]
        image_tensor = image_tensor.repeat(3, 1, 1)

        # Apply any additional transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label).long(), patient_id
    

class BreastDataset2DMulticlass(Dataset):
    """
    Dataset that loads ALL .png images from each patient's folder
    as separate samples for multiclass classification.
    """

    def __init__(self, csv_file, data_dir, transform=None, resize_to=(224, 224)):
        """
        Args:
            csv_file (str): Path to a CSV containing at least:
                            - "Patient_ID"
                            - "SOC_binario" (with values "maligno" or "benigno")
                            - "Diagnosis" (with values 'No follow up', 'Follow up', 'Biopsy')
            data_dir (str): Base directory where 'benign'/'malign' folders exist,
                            each containing patient subfolders with images.
            transform (callable, optional): Optional PyTorch-style transform
                                            to apply to each image tensor.
            resize_to (tuple): (width, height) size to resize images, e.g. (224, 224).
        """
        self.transform = transform
        self.resize_to = resize_to

        # Mapping for multiclass labels
        self.diagnosis_mapping = {
            'No follow up': 0,
            'Follow up': 1,
            'Biopsy': 2
        }

        # Read the CSV with patient & label info
        df = pd.read_csv(csv_file)

        # We'll collect (image_path, label, patient_id) tuples here
        self.samples = []

        # Go through each patient row in the CSV
        for i in range(len(df)):
            row = df.iloc[i]
            patient_id = row["Patient_ID"]
            soc_binario = row["SOC_binario"]  # "maligno" or "benigno"
            diagnosis = row["BIRADS_CAT"]

            if diagnosis not in self.diagnosis_mapping:
                raise ValueError(f"Unexpected diagnosis: {diagnosis}")

            label = self.diagnosis_mapping[diagnosis]

            # Build the path to the patient's folder
            if soc_binario == "benigno":
                folder_path = os.path.join(data_dir, "benign", str(patient_id), "cropped_p2")
            elif soc_binario == "maligno":
                folder_path = os.path.join(data_dir, "malign", str(patient_id), "cropped_p2")
            else:
                raise ValueError(f"Unexpected SOC_binario value: {soc_binario}")

            # Collect ALL .png/.jpg files in that folder
            image_files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg"))
            if not image_files:
                print(f"Warning: no .png or .jpg files found for patient {patient_id} at {folder_path}")
                continue

            # Each .png is a separate training sample
            for image_path in image_files:
                self.samples.append((image_path, label, patient_id))  # Include patient_id

        print(f"[BreastDatasetMulticlass] Found {len(self.samples)} total images across all patients.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]

        # Read the image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        image = cv2.resize(image, self.resize_to)  # [H, W]
        image_tensor = torch.tensor(image).unsqueeze(0).float() / 255.0  # shape [1, H, W]

        # Repeat along channel dimension => shape [3, H, W]
        image_tensor = image_tensor.repeat(3, 1, 1)

        # Apply any additional transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label).long(), patient_id
