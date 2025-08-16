import torch

from torch.utils.data import DataLoader

from models import AlterEgo
from training import train
from datasets import PreprocessedDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
model = AlterEgo().to(device)

dataset = PreprocessedDataset(only_tiny=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 3

for t in range(epochs):
    avg_loss = train(dataloader, model, device)
    print(f"Epoch {t+1}, Average loss: {avg_loss}\n-------------------------------")

print("Done!")