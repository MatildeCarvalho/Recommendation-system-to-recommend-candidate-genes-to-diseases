import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import sqlite3
from data_processing import process_data



class RecSysModel(nn.Module):
    def __init__(self, n_diseases, n_genes, n_factors=16):  # Adicionando n_hidden para o tamanho da camada intermediária
        super().__init__()
        self.diseases_embed = nn.Embedding(n_diseases, n_factors)
        self.genes_embed = nn.Embedding(n_genes, n_factors)
        self.out = nn.Linear(n_factors*2, 1)  # Alterado para prever um valor contínuo

    def forward(self, diseases, genes):
        diseases_embeds = self.diseases_embed(diseases)
        genes_embeds = self.genes_embed(genes)
        output = torch.cat([diseases_embeds, genes_embeds], dim=1)
        output = self.out(output)
        output = torch.sigmoid(output)  # Aplicar a ativação sigmoid
        return output.squeeze()

    def rmse(self, output, target):
        return torch.sqrt(nn.MSELoss()(output, target)) # Alteração aqui


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    rmse_values = []

    for batch_idx, (diseases, genes, ei) in enumerate(train_loader):
        diseases, genes, ei = diseases.to(device), genes.to(device), ei.to(device)

        optimizer.zero_grad()
        output = model(diseases, genes)
        output = output.view(-1, 1)

        rating = ei.view(len(ei), -1).to(torch.float32).detach()

        loss = criterion(output, rating)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Print da perda a cada lote
        #print(f'Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item()}')

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for val_batch_idx, (val_diseases, val_genes, val_ei) in enumerate(val_loader):
            val_diseases, val_genes, val_ei = val_diseases.to(device), val_genes.to(device), val_ei.to(device)

            val_output = model(val_diseases, val_genes)
            val_output = val_output.view(-1, 1)

            val_rating = val_ei.view(len(val_ei), -1).to(torch.float32).detach()

            val_loss = criterion(val_output, val_rating)
            total_loss += val_loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss
##
