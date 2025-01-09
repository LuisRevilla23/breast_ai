import argparse
import yaml
from addict import Dict
import torch
import random
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb  # Import wandb
from model.model_lightning import MyModel  # Update import with folder name
from utils.data_datamodule import BreastDataModule  # Updated import
import os
import lightning as L

def main(config_file):
    # Cargar configuración desde archivo YAML
    conf = Dict(yaml.safe_load(open(config_file, "r")))
    print(f"Configuración cargada: {conf}")

    # Establecer semillas aleatorias para reproducibilidad
    random_seed = conf.train_par.random_seed if conf.train_par.random_seed != 'default' else 2022
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Crear directorio para resultados
    results_path = os.path.join(conf.train_par.results_path, conf.dataset.experiment)
    os.makedirs(results_path, exist_ok=True)
    conf.train_par.results_model_filename = os.path.join(results_path, 'best_model.pt')

    # Inicializar DataModule
    data_module = BreastDataModule(
        batch_size=conf.train_par.batch_size,
        workers=conf.train_par.workers,
        train_file=conf.dataset.train_dir,
        test_file=conf.dataset.test_dir,
        data_dir=conf.dataset.data_dir,
        resize_to=(224, 224),
    )

    # Inicializar modelo Lightning
    model = MyModel(model_opts=conf.model_opts, train_par=conf.train_par)

    # Configurar Wandb Logger
    wandb_logger = WandbLogger(
        project="Breast",
        config=conf,
        name=f"{conf.wandb.prefix_name}_{conf.dataset.experiment}"
    )

    # Configurar callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=conf.train_par.patience,
        mode="min",
        verbose=True
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=results_path,
        filename='best_model',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    # Inicializar Trainer de Lightning
    trainer = L.Trainer(
        max_epochs=conf.train_par.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Entrenar el modelo
    trainer.fit(model=model, datamodule=data_module)

    # Evaluar el modelo en el conjunto de prueba (opcional)
    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script de entrenamiento para clasificación binaria')
    parser.add_argument('--config-file', type=str, default='./default_config_train.yaml', help='Ruta al archivo de configuración YAML')
    args = parser.parse_args()

    main(args.config_file)
