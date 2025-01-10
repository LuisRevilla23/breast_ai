from .dataset import BreastDataset2D, BreastDataset2DMulticlass
import torch
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from collections import Counter

class BreastDataModule(L.LightningDataModule):
    """DataModule para manejar los datasets de entrenamiento, validación y prueba"""

    def __init__(self, batch_size, workers, train_file, test_file, data_dir, resize_to=(224, 224)):
        super().__init__()  # IMPORTANT: call parent constructor

        self.batch_size = batch_size
        self.workers = workers
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.resize_to = resize_to

        # Optionally specify if data should be prepared once vs. per node
        # prepare_data_per_node can be changed if you have specific needs:
        # self.prepare_data_per_node = True

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        """Use this if you need to download data, tokenize, etc. 
           It's called only from a single process in the cluster."""
        pass

    def setup(self, stage=None):
        """
        Lightning calls setup() in these situations:
          - once per process, per stage (train/val/test), 
          - or at Trainer init if you pass `setup_on_init=True`.
        """
        self.train_dataset = BreastDataset2D(
            csv_file=self.train_file,
            data_dir=self.data_dir,
            transform=self.train_transforms,
            resize_to=self.resize_to,
        )


        labels = [label for _, label, _ in self.train_dataset.samples]
        print("Distribución de clases antes del balanceo:", Counter(labels))

        self.test_dataset = BreastDataset2D(
            csv_file=self.test_file,
            data_dir=self.data_dir,
            resize_to=self.resize_to,
        )
        # If you had a separate val dataset, you'd define self.val_dataset

        self.val_dataset = BreastDataset2D(
            csv_file=self.train_file,
            data_dir=self.data_dir,
            resize_to=self.resize_to,
        )

    def train_dataloader(self):
        # Contar ejemplos de cada clase
        class_counts = [0, 0]  # Inicializar contadores para benigno y maligno
        for _, label, _ in self.train_dataset.samples:
            class_counts[label] += 1

        # Calcular pesos para cada clase
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[label] for _, label, _ in self.train_dataset.samples]

        # Crear el sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )


class BreastDataModuleMulticlass(L.LightningDataModule):
    """DataModule para manejar los datasets de entrenamiento, validación y prueba"""

    def __init__(self, batch_size, workers, train_file, test_file, data_dir, resize_to=(224, 224)):
        super().__init__()  # IMPORTANT: call parent constructor

        self.batch_size = batch_size
        self.workers = workers
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.resize_to = resize_to

        # Optionally specify if data should be prepared once vs. per node
        # prepare_data_per_node can be changed if you have specific needs:
        # self.prepare_data_per_node = True

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        """Use this if you need to download data, tokenize, etc. 
           It's called only from a single process in the cluster."""
        pass

    def setup(self, stage=None):
        """
        Lightning calls setup() in these situations:
          - once per process, per stage (train/val/test), 
          - or at Trainer init if you pass `setup_on_init=True`.
        """
        self.train_dataset = BreastDataset2DMulticlass(
            csv_file=self.train_file,
            data_dir=self.data_dir,
            transform=self.train_transforms,
            resize_to=self.resize_to,
        )


        labels = [label for _, label, _ in self.train_dataset.samples]
        print("Distribución de clases antes del balanceo:", Counter(labels))

        self.test_dataset = BreastDataset2DMulticlass(
            csv_file=self.test_file,
            data_dir=self.data_dir,
            resize_to=self.resize_to,
        )
        # If you had a separate val dataset, you'd define self.val_dataset

        self.val_dataset = BreastDataset2DMulticlass(
            csv_file=self.train_file,
            data_dir=self.data_dir,
            resize_to=self.resize_to,
        )

    def train_dataloader(self):
        # Contar ejemplos de cada clase
        class_counts = [0, 0, 0]  # Inicializar contadores para benigno y maligno
        for _, label, _ in self.train_dataset.samples:
            class_counts[label] += 1

        # Calcular pesos inversos
        #   O bien la forma “clásica”:
        #       class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        #   O la forma escalada para que la suma sea más consistente, etc.
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

        # Crear sample_weights para cada muestra
        sample_weights = [class_weights[label] for _, label, _ in self.train_dataset.samples]


        # Crear el sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # sampler=sampler,
            shuffle = True,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
