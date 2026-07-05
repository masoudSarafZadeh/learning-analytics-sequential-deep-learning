import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl

class OULADataset(Dataset):
    def __init__(self, sequences, feature_columns, sparse_feature_names, missing_value_placeholder):
        self.sequences = sequences
        self.feature_columns = feature_columns
        self.sparse_feature_names = sparse_feature_names
        self.missing_value_placeholder = missing_value_placeholder
        self.dense_feature_names = [f for f in feature_columns if f not in sparse_feature_names]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        if sequence.empty:
            num_final_features = len(self.dense_feature_names) + (len(self.sparse_feature_names) * 2)
            return torch.empty(0, num_final_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        processed_sequence_data = []
        for _, row in sequence.iterrows():
            daily_features = row[self.dense_feature_names].tolist()
            sparse_features_and_indicators = []
            
            for sparse_col in self.sparse_feature_names:
                val = row[sparse_col]
                if pd.isna(val) or val == self.missing_value_placeholder: 
                    sparse_features_and_indicators.extend([0.0, 0.0])
                else:
                    sparse_features_and_indicators.extend([float(val), 1.0]) 
            
            processed_sequence_data.append(daily_features + sparse_features_and_indicators)

        sequence_tensor = torch.tensor(processed_sequence_data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sequence_tensor, label_tensor 

def custom_collate_fn(batch):
    sequences, labels = zip(*batch)
    original_lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
    sorted_lengths, sorted_indices = original_lengths.sort(descending=True)
    
    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_labels = torch.stack([labels[i] for i in sorted_indices])

    padded_sequences = nn.utils.rnn.pad_sequence(sorted_sequences, batch_first=True, padding_value=0.0)

    return {
        'sequence': padded_sequences,
        'label': sorted_labels,
        'lengths': sorted_lengths,
        'original_indices': sorted_indices
    }

class OULADataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, val_sequences, test_sequences, batch_size, feature_columns, sparse_feature_names, missing_value_placeholder):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.sparse_feature_names = sparse_feature_names
        self.missing_value_placeholder = missing_value_placeholder
        print("OULADataModule initialized.")

    def setup(self, stage=None):
        self.train_dataset = OULADataset(self.train_sequences, self.feature_columns, self.sparse_feature_names, self.missing_value_placeholder)
        self.val_dataset = OULADataset(self.val_sequences, self.feature_columns, self.sparse_feature_names, self.missing_value_placeholder)
        self.test_dataset = OULADataset(self.test_sequences, self.feature_columns, self.sparse_feature_names, self.missing_value_placeholder)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False, collate_fn=custom_collate_fn)
