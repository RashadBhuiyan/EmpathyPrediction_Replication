# MODEL UPDATES (TODO):
# all of the following model updates are for unique model variants, not cumulative updates to the same model
# 1. cosine-similarity is replaced with 3-layer MLP with input dimension of 768*2, and layer dimensions 768, 384, 192, followed by a sigmoid
# 2. storyB only classifier replaces with 2-layer MLP with input dimension of 768 and layer dimensions of 384 and 192, followed by a sigmoid
# 3. classifier with 2nd embeddings (ie. demographics or place and why) replaces with a 4-layer MLP with layer dimensions of 1536, 768, 384, 192, followed by a sigmoid --> CANNOT DO AS THERE IS NO DEMOGRAPHIC INFORMATION PROVIDED AS PART OF THE DATASET
# 4. experiment s with e5 (embedding size of 1024), MLP input dimension is changes to 1024*2 and layer dimensions are 1024,512,192 followed by a sigmoid

import gc
import os
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import SpearmanCorrCoef, F1Score, PearsonCorrCoef, Precision, Recall, MeanSquaredError, Accuracy
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from DatasetList import EFPDataset
import config as cfg

# config and paths
config = cfg.load_config()
DATA_DIR = config["DATA_DIR"]
EFP_TRAIN = config["EFP_TRAIN"]
EFP_TEST = config["EFP_TEST"]
EFP_VAL = config["EFP_VAL"]
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
train_path = os.path.join(parent_dir, DATA_DIR, EFP_TRAIN)
test_path = os.path.join(parent_dir, DATA_DIR, EFP_TEST)
val_path = os.path.join(parent_dir, DATA_DIR, EFP_VAL)

torch.set_float32_matmul_precision('high')

# Define 3-Layer MLP architecture for PPEP model
class EmpathyMLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dims=None):
        super(EmpathyMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [768, 384, 192]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ContextModel(pl.LightningModule):
    def __init__(self, model="SBERT", pooling="CLS", input_dim=1536, hidden_dims=None):
        super(ContextModel, self).__init__()
        self.base_model = model
        self.pooling = pooling
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [768, 384, 192]

        # Load pre-trained model weights and initialize corresponding tokenizer.
        if self.base_model == "SBERT":
            self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            self.tokenizer = self.model.tokenizer
        else:
            self.model = SentenceTransformer("intfloat/e5-large-v2")
            self.tokenizer = self.model.tokenizer

        self.learning_rate = 1e-6
        
        # Initialize MLP for empathy prediction
        self.mlp = EmpathyMLP(input_dim=self.input_dim, hidden_dims=self.hidden_dims)

        # load evaluation metrics
        self.f1_score = F1Score(task="binary")
        self.spearman = SpearmanCorrCoef()
        self.pearson = PearsonCorrCoef()
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.mse = MeanSquaredError()
        self.accuracy = Accuracy(task="binary")

    def forward(self, story):
        story = self.tokenizer(story,padding=True,truncation=True,return_tensors="pt")
        for k in story:
            story[k] = story[k].to(self.device)
        
        embedding = self.model(story)
        if self.pooling == "MEAN":
            sentence_representation = embedding.sentence_embedding
        else:
            sentence_representation = self.cls_pooling(embedding.token_embeddings)

        return sentence_representation

    def cls_pooling(self, token_embeddings):
        return token_embeddings[:,0]

    def training_step(self, batch, batch_idx):
        # get batch information
        batch = batch[0]
        place_A = batch[0]
        why_A = batch[1]
        story_A = batch[2]
        place_B = batch[3]
        why_B = batch[4]
        story_B = batch[5]
        empathy_score = batch[6]

        # combine the two parts to make two full stories
        full_A = place_A + "[SEP]" + why_A + "[SEP]" + story_A
        full_B = place_B + "[SEP]" + why_B + "[SEP]" + story_B

        # use embeddings for MLP
        storyA_emb = self(full_A)
        storyB_emb = self(full_B)
        
        # Concatenate embeddings and pass through MLP
        concatenated = torch.cat([storyA_emb, storyB_emb], dim=1)
        mlp_output = self.mlp(concatenated).squeeze(1)
        
        # calculate loss
        self.mse = self.mse.to(self.device)
        loss = 0
        loss += self.mse(mlp_output, empathy_score)
        self.log("train_loss", loss)
        return loss

    def eval_step(self,batch,batch_idx,prefix):
        # get batch information
        batch = batch[0]
        place_A = batch[0]
        why_A = batch[1]
        story_A = batch[2]
        place_B = batch[3]
        why_B = batch[4]
        story_B = batch[5]
        empathy_score = batch[6]

        # combine the two parts to make two full stories
        full_A = place_A + "[SEP]" + why_A + "[SEP]" + story_A
        full_B = place_B + "[SEP]" + why_B + "[SEP]" + story_B

        # use embeddings for MLP
        storyA_emb = self(full_A)
        storyB_emb = self(full_B)
        
        # Concatenate embeddings and pass through MLP
        concatenated = torch.cat([storyA_emb, storyB_emb], dim=1)
        mlp_output = self.mlp(concatenated).squeeze(1)

        # Normalize empathy_score to [0, 1] range (binarize at median/mean for classification)
        threshold = 0.5  # Median of [0, 1]
        mlp_binary = (mlp_output >= threshold).int()
        empathy_binary = (empathy_score >= threshold).int()

        # evaluation metric with binarized values
        self.precision = self.precision.to(self.device)
        self.recall = self.recall.to(self.device)
        self.accuracy = self.accuracy.to(self.device)
        self.f1_score = self.f1_score.to(self.device)
        self.precision.update(mlp_binary, empathy_binary)
        self.recall.update(mlp_binary, empathy_binary)
        self.accuracy.update(mlp_binary, empathy_binary)
        self.f1_score.update(mlp_binary, empathy_binary)

        # calculate evaluation metrics        
        self.spearman = self.spearman.to(self.device)
        self.pearson = self.pearson.to(self.device)
        self.spearman.update(mlp_output.float(), empathy_score.float())
        self.pearson.update(mlp_output.float(), empathy_score.float())

        
        # calculate loss
        self.mse = self.mse.to(self.device)
        loss = self.mse(mlp_output, empathy_score)
        self.log(prefix+"_loss", loss)
        return loss
    
    def on_eval_end(self,prefix):
        # Compute metrics at the end of the epoch and log them
        f1 = self.f1_score.compute()
        spearman = self.spearman.compute()
        pearson = self.pearson.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        accuracy = self.accuracy.compute()
        mse = self.mse.compute()

        self.log(prefix+"_f1", f1, sync_dist=True)
        self.log(prefix+"_spearman", spearman, sync_dist=True)
        self.log(prefix+"_pearson", pearson, sync_dist=True)
        self.log(prefix+"_precision", precision, sync_dist=True)
        self.log(prefix+"_recall", recall, sync_dist=True)
        self.log(prefix+"_accuracy", accuracy, sync_dist=True)
        self.log(prefix+"_mse", mse, sync_dist=True)
        
        # Reset metrics for next epoch
        self.f1_score.reset()
        self.spearman.reset()
        self.pearson.reset()
        self.precision.reset()
        self.recall.reset()
        self.accuracy.reset()
        self.mse.reset()

    def on_validation_epoch_start(self):
        # Reset metrics at the start of validation epoch
        self.f1_score.reset()
        self.spearman.reset()
        self.pearson.reset()
        self.precision.reset()
        self.recall.reset()
        self.accuracy.reset()
        self.mse.reset()

    def on_test_epoch_start(self):
        # Reset metrics at the start of test epoch
        self.f1_score.reset()
        self.spearman.reset()
        self.pearson.reset()
        self.precision.reset()
        self.recall.reset()
        self.accuracy.reset()
        self.mse.reset()

    def on_validation_epoch_end(self):
        self.on_eval_end(prefix="val")

    def on_test_epoch_end(self):
        self.on_eval_end(prefix="eval")

    def validation_step(self, batch, batch_idx):
        r = self.eval_step(batch,batch_idx,prefix="val")
        return r

    def test_step(self,batch,batch_idx):
        r = self.eval_step(batch,batch_idx,prefix="eval")
        return r

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.5 * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
if __name__ == '__main__':
    train_d = pd.read_csv(train_path)
    val_d = pd.read_csv(val_path)
    test_d = pd.read_csv(test_path)

    # Work through loop for different epochs and different embedders
    epochs = [100] # should be [100]
    for epoch in epochs:
        # Establish DataLoaders for training, validation, and testing
        train_ds = EFPDataset(train_d)
        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=16)
        val_ds = EFPDataset(val_d)
        val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=16)
        test_ds = EFPDataset(test_d)
        test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=16)

        # Establish callbacks and logger
        lr_monitor = LearningRateMonitor(logging_interval='step')
        spearman_callback = ModelCheckpoint(save_top_k=1, monitor="val_spearman", mode="max")
        logger = CSVLogger(save_dir="logs", name=f"context_model_{epoch}")
        precision = 32

        # Build model and trainer
        model = ContextModel()
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