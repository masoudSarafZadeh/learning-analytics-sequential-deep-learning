import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Import from utils
from utils import OULADataModule

N_CLASSES = 1

# ==========================================
# Neural Network Architectures
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention_weights_layer = nn.Linear(d_model, 1)

    def forward(self, transformer_output: torch.Tensor, mask: torch.Tensor):
        raw_attention_scores = self.attention_weights_layer(transformer_output).squeeze(-1)
        raw_attention_scores = raw_attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = torch.softmax(raw_attention_scores, dim=1)
        pooled_output = torch.sum(transformer_output * attention_weights.unsqueeze(-1), dim=1)
        return pooled_output, attention_weights

class TransformerSequenceModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int, d_model: int = 64, nhead: int = 4, 
                 num_encoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1, max_seq_len: int = 43):
        super().__init__()
        self.input_embedding = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attention_pooling = AttentionPooling(d_model)
        self.classifier_head = nn.Linear(d_model, n_classes) 
        self.n_classes = n_classes 

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        
        current_batch_max_seq_len = x.size(1)
        mask = torch.arange(current_batch_max_seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        transformer_output = self.transformer_encoder(src=x, src_key_padding_mask=mask)
        pooled_output, attention_weights = self.attention_pooling(transformer_output, mask)
        logits = self.classifier_head(pooled_output)

        return logits, attention_weights

class OULADPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int = N_CLASSES, d_model: int = 64, nhead: int = 4, 
                 num_encoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1, max_seq_len: int = 43):
        super().__init__()
        self.save_hyperparameters() 
        self.model = TransformerSequenceModel(
            n_features=n_features, n_classes=n_classes, d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, 
            dropout=dropout, max_seq_len=max_seq_len
        )
        self.criterion = nn.BCEWithLogitsLoss()
            
    def forward(self, x, lengths):
        return self.model(x, lengths)
            
    def _common_step(self, batch, batch_idx):
        sequences = batch["sequence"] 
        labels = batch["label"]     
        lengths = batch["lengths"]  

        labels = labels.float().unsqueeze(1) 
        outputs, _ = self(sequences, lengths) 
        
        loss = self.criterion(outputs, labels)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).long() 
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
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, threshold=0.0001, patience=4, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    from preprocess import df_info_allterm, merged_vle_aca_info_filtered 
    
    print("Splitting data and generating sequences...")
    
    # 1. Train/Val/Test Split
    train_df, temp_df = train_test_split(df_info_allterm, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_list = train_df['id_student'].unique().tolist()
    val_list = val_df['id_student'].unique().tolist()
    test_list = test_df['id_student'].unique().tolist()

    all_train = merged_vle_aca_info_filtered[merged_vle_aca_info_filtered['id_student'].isin(train_list)].copy()
    all_val = merged_vle_aca_info_filtered[merged_vle_aca_info_filtered['id_student'].isin(val_list)].copy()
    all_test = merged_vle_aca_info_filtered[merged_vle_aca_info_filtered['id_student'].isin(test_list)].copy()

    print("Scaling features...")
    exclude_cols = ['id_student', 'week_from_start', 'final_result']
    cols_to_scale = [c for c in all_train.columns if c not in exclude_cols]

    scaler = StandardScaler()

    all_train[cols_to_scale] = scaler.fit_transform(all_train[cols_to_scale])
    all_val[cols_to_scale] = scaler.transform(all_val[cols_to_scale])
    all_test[cols_to_scale] = scaler.transform(all_test[cols_to_scale])

    MISSING_VALUE_PLACEHOLDER = 0.0
    all_train = all_train.fillna(MISSING_VALUE_PLACEHOLDER)
    all_val = all_val.fillna(MISSING_VALUE_PLACEHOLDER)
    all_test = all_test.fillna(MISSING_VALUE_PLACEHOLDER)
    # ==========================================

    # 2. Sequence Definitions
    FEATURE_COLUMNS = all_train.columns.tolist()[1:-1]
    SPARSE_FEATURE_NAMES = ['pure_score']
    NUM_DAILY_FEATURES = len([f for f in FEATURE_COLUMNS if f not in SPARSE_FEATURE_NAMES])
    NUM_SPARSE_FEATURES = len(SPARSE_FEATURE_NAMES)
    FEATURE_DIM = NUM_DAILY_FEATURES + (NUM_SPARSE_FEATURES * 2) 

    train_sequences, val_sequences, test_sequences = [], [], []

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

    print(f"Loaded {len(train_sequences)} train sequences, {len(val_sequences)} val sequences, {len(test_sequences)} test sequences.")

    # 3. Model Configuration
    MAX_SEQUENCE_LENGTH = 43 
    N_EPOCHS = 50
    BATCH_SIZE = 64

    data_module = OULADataModule(
        train_sequences, val_sequences, test_sequences, BATCH_SIZE,
        FEATURE_COLUMNS, SPARSE_FEATURE_NAMES, MISSING_VALUE_PLACEHOLDER
    )
    data_module.setup()

    model = OULADPredictor(n_features=FEATURE_DIM, n_classes=N_CLASSES, max_seq_len=MAX_SEQUENCE_LENGTH)

    # 4. Callbacks and Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="best_transformer_checkpoint",
        save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=12, verbose=True, mode="min"
    )
    progress_bar = TQDMProgressBar(refresh_rate=30)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger("lightning_logs", name="OULAD_Transformer_v2")

    # 5. Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar, lr_monitor], 
        max_epochs=N_EPOCHS,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Complete. Running validation and test ---")

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path}")
        best_model = OULADPredictor.load_from_checkpoint(best_model_path)
        trainer.test(best_model, datamodule=data_module)
    else:
        print("No best model checkpoint found to test.")
