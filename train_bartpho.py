import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, BartphoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import os
import gc
from tqdm import tqdm
import time
import logging
import warnings
warnings.filterwarnings("ignore")


# Khởi tạo logger
logger = logging.getLogger("training")
logger.setLevel(logging.INFO)
# Lưu file log
handler = logging.FileHandler("training.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# Define the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiacriticRestorationDataset(Dataset):
    def __init__(self, file_path_without_diacritics, file_path_with_diacritics, tokenizer, max_length=1024):
        self.pairs = self.read_data(file_path_without_diacritics, file_path_with_diacritics)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def read_data(self, file_path_without_diacritics, file_path_with_diacritics):
        pairs = []
        with open(file_path_without_diacritics, 'r', encoding='utf-8') as file_without_diacritics, open(file_path_with_diacritics, 'r', encoding='utf-8') as file_with_diacritics:
            for line_without_diacritics, line_with_diacritics in zip(file_without_diacritics, file_with_diacritics):
                sentence_without_diacritics = line_without_diacritics.strip()
                sentence_with_diacritics = line_with_diacritics.strip()
                pairs.append((sentence_without_diacritics, sentence_with_diacritics))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        encoding = self.tokenizer(pair[0], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        label = self.tokenizer(pair[1], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')['input_ids'].squeeze()
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'labels': label.squeeze()}

# Example usage
tokenizer = BartphoTokenizer.from_pretrained("vinai/bartpho-syllable")
model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")
model.to(device)

# Train data
train_without_diacritics_path = "/kaggle/input/accent-vietnamese/train.txt"
train_with_diacritics_path = "/kaggle/input/accent-vietnamese/train_Label.txt"
train_dataset = DiacriticRestorationDataset(train_without_diacritics_path, train_with_diacritics_path, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# Validation data
val_without_diacritics_path = "/kaggle/input/accent-vietnamese/valid.txt"
val_with_diacritics_path = "/kaggle/input/accent-vietnamese/valid_Label.txt"
val_dataset = DiacriticRestorationDataset(val_without_diacritics_path, val_with_diacritics_path, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Training loop
num_epochs = 10
save_dir = "./checkpoint"
os.makedirs(save_dir, exist_ok=True)


best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_batches = 0
    average_loss = 0
    average_val_loss = 0
    with tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as progress_bar:
        # Loop qua các batches trong train_dataloader
        for batch_idx, batch in enumerate(progress_bar):
        # for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            # print(f'Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item()}')
            progress_bar.set_postfix({'Loss': f'{loss.item():.10f}'})

            del input_ids
            del attention_mask
            del labels
            del output
            gc.collect()
            torch.cuda.empty_cache()

        average_loss = total_loss / total_batches
        
        if average_loss < best_loss:
            best_loss = average_loss

            # Save the model as the new best model
            best_model_save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved at {best_model_save_path}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        total_val_batches = 0

        for val_batch in val_dataloader:  # Assuming you have a DataLoader for the validation set
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['labels'].to(device)

            output = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
            val_loss = output.loss

            total_val_loss += val_loss.item()
            total_val_batches += 1

        average_val_loss = total_val_loss / total_val_batches
    print(f"Epoch {epoch + 1}/{num_epochs},Train Loss: {average_loss}, Validation Loss: {average_val_loss}")
    logger.info(f'Epoch {epoch+1}, Train_Loss: {average_loss:.10f}, Val_Loss: {average_val_loss:.10f}')
handler.close()