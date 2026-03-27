import os
import pandas as pd
import gc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from PPEPModel import PPEPModel
from DatasetList import EFPDataset, ESDataset, ConcatenatedDataset
import torch
from torch.utils.data import DataLoader
import config as cfg

# config and paths
config = cfg.load_config()
DATA_DIR = config["DATA_DIR"]
EFP_TRAIN = config["EFP_TRAIN"]
EFP_TEST = config["EFP_TEST"]
EFP_VAL = config["EFP_VAL"]
ES_TRAIN = config["ES_TRAIN"]
ES_TEST = config["ES_TEST"]
ES_VAL = config["ES_VAL"]
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
efp_train_path = os.path.join(parent_dir, DATA_DIR, EFP_TRAIN)
es_train_path = os.path.join(parent_dir, DATA_DIR, ES_TRAIN)
efp_test_path = os.path.join(parent_dir, DATA_DIR, EFP_TEST)
es_test_path = os.path.join(parent_dir, DATA_DIR, ES_TEST)
efp_val_path = os.path.join(parent_dir, DATA_DIR, EFP_VAL)
es_val_path = os.path.join(parent_dir, DATA_DIR, ES_VAL)

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    efp_train_d = pd.read_csv(efp_train_path)
    es_train_d = pd.read_csv(es_train_path)
    efp_val_d = pd.read_csv(efp_val_path)
    es_val_d = pd.read_csv(es_val_path)
    efp_test_d = pd.read_csv(efp_test_path)
    es_test_d = pd.read_csv(es_test_path)

    # Work through loop for different epochs and different embedders
    epochs = [1] # should be [100]
    for epoch in epochs:
        # Establish DataLoaders for training, validation, and testing
        efp_train_ds = EFPDataset(efp_train_d)
        es_train_ds = ESDataset(es_train_d)
        train_ds = ConcatenatedDataset(efp_train_ds, es_train_ds)
        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=16)
        efp_val_ds = EFPDataset(efp_val_d)
        es_val_ds = ESDataset(es_val_d)
        val_ds = ConcatenatedDataset(efp_val_ds, es_val_ds)
        val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=16)
        efp_test_ds = EFPDataset(efp_test_d)
        es_test_ds = ESDataset(es_test_d)
        test_ds = ConcatenatedDataset(efp_test_ds, es_test_ds)
        test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=16)

        # Establish callbacks and logger
        lr_monitor = LearningRateMonitor(logging_interval='step')
        spearman_callback = ModelCheckpoint(save_top_k=1, monitor="val_spearman", mode="max")
        logger = CSVLogger(save_dir="logs", name=f"ppep_model_EFP_ES_{epoch}")
        precision = 32

        # Build model and trainer
        model = PPEPModel()
        trainer = pl.Trainer(
            log_every_n_steps=5,
            max_epochs=int(epoch), 
            accelerator="gpu",
            callbacks=[lr_monitor, spearman_callback],
            precision=precision,
            strategy=DDPStrategy(find_unused_parameters=True),
            logger=logger
        )
        trainer.fit(model=model,train_dataloaders=[train_dl], val_dataloaders = [val_dl])
        trainer.test(model=model,dataloaders=[test_dl])

        # delete models and free GPU after use
        model.cpu()
        del model, trainer, train_ds, train_dl, val_ds, val_dl, test_ds, test_dl, logger, lr_monitor, spearman_callback
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()