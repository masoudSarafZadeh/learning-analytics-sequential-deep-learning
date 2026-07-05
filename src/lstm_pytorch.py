import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence
from torchmetrics.functional import accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# Import from utils file
from utils import OULADataModule

N_CLASSES = 1

class SequenceModel(nn.Module):
    def __init__(self, n_features, n_classes=N_CLASSES, n_hidden=256, n_layeres=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layeres,
            batch_first=True,
            dropout=0.2
        )
        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x, lengths):
        if x.shape[0] == 0:
            return torch.empty(0, self.classifier.out_features, device=x.device)

        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, _) = self.lstm(packed_input)
        out = hidden[-1]
        out = self.classifier(out)
        return torch.sigmoid(out)

class OULADPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.model = SequenceModel(n_features=n_features, n_classes=n_classes)
        self.criterion = nn.BCELoss()

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def _common_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        lengths = batch["lengths"]

        outputs = self(sequences, lengths) 
        labels = labels.float().unsqueeze(1) 
        
        loss = self.criterion(outputs, labels)
        predictions = (outputs > 0.5).long()
        step_accuracy = accuracy(predictions, labels, task="binary")
        return loss, step_accuracy

    def training_step(self, batch, batch_idx):
        loss, step_accuracy = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, logger=True, on_step=True, on_epoch=True)
        self.log('train_accuracy', step_accuracy, logger=True, on_step=True, on_epoch=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        loss, step_accuracy = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, logger=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', step_accuracy, logger=True, on_step=False, on_epoch=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def test_step(self, batch, batch_idx):
        loss, step_accuracy = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, logger=True, on_step=False, on_epoch=True)
        self.log('test_accuracy', step_accuracy, logger=True, on_step=False, on_epoch=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    from preprocess import df_info_filtered, final_filtered
    
    print("Splitting data and generating sequences...")
    
    # 2. Train/Val/Test Split
    train_df, temp_df = train_test_split(df_info_filtered, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_list = train_df['id_student'].unique().tolist()
    val_list = val_df['id_student'].unique().tolist()
    test_list = test_df['id_student'].unique().tolist()

    all_train = final_filtered[final_filtered['id_student'].isin(train_list)].copy()
    all_val = final_filtered[final_filtered['id_student'].isin(val_list)].copy()
    all_test = final_filtered[final_filtered['id_student'].isin(test_list)].copy()

    print("Scaling features...")
    exclude_cols = ['id_student', 'week_from_start', 'final_result']
    cols_to_scale = [c for c in all_train.columns if c not in exclude_cols]
    scaler = StandardScaler()
    scaler.fit(all_train[cols_to_scale])
    all_train[cols_to_scale] = scaler.transform(all_train[cols_to_scale])
    all_val[cols_to_scale] = scaler.transform(all_val[cols_to_scale])
    all_test[cols_to_scale] = scaler.transform(all_test[cols_to_scale])

    all_train = all_train.fillna(0.0)
    all_val = all_val.fillna(0.0)
    all_test = all_test.fillna(0.0)

    # 3. Feature Definitions
    FEATURE_COLUMNS = all_train.columns.tolist()[1:-1]
    SPARSE_FEATURE_NAMES = ['score_CMA', 'score_TMA']

    MISSING_VALUE_PLACEHOLDER = 0.0

    NUM_DAILY_FEATURES = len([f for f in FEATURE_COLUMNS if f not in SPARSE_FEATURE_NAMES])
    NUM_SPARSE_FEATURES = len(SPARSE_FEATURE_NAMES)
    LSTM_INPUT_SIZE = NUM_DAILY_FEATURES + (NUM_SPARSE_FEATURES * 2) 

    # 4. Sequence Generation
    train_sequences = []
    val_sequences = []
    test_sequences = []

    for id_student, group in all_train.groupby('id_student'):
        sequence_features = group[FEATURE_COLUMNS]
        label = train_df[train_df.id_student==id_student].iloc[0].final_result
        train_sequences.append((sequence_features, label))

    for id_student, group in all_val.groupby('id_student'):
        sequence_features = group[FEATURE_COLUMNS]
        label = val_df[val_df.id_student==id_student].iloc[0].final_result
        val_sequences.append((sequence_features, label))

    for id_student, group in all_test.groupby('id_student'):
        sequence_features = group[FEATURE_COLUMNS]
        label = test_df[test_df.id_student==id_student].iloc[0].final_result
        test_sequences.append((sequence_features, label))

    # 5. PyTorch Lightning Configuration
    N_EPOCHS = 10
    BATCH_SIZE = 64

    # Initialize DataModule
    data_module = OULADataModule(
        train_sequences, val_sequences, test_sequences, BATCH_SIZE,
        FEATURE_COLUMNS, SPARSE_FEATURE_NAMES, MISSING_VALUE_PLACEHOLDER
    )
    data_module.setup()

    # Initialize Model
    model = OULADPredictor(n_features=LSTM_INPUT_SIZE, n_classes=N_CLASSES)

    # Callbacks and Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    progress_bar = TQDMProgressBar(refresh_rate=30)
    logger = TensorBoardLogger("lightning_logs", name="OULAD")

    # 6. Train the Model
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar],
        max_epochs=N_EPOCHS,
        enable_progress_bar=True
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)

    # 7. Test the Model
    print("Testing the best model...")
    test_results = trainer.test(ckpt_path="best", datamodule=data_module)
    
    print("\nTest Results:")
    print(test_results)
