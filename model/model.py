import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
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

def train(model, optimizer, criterion, train_loader, device, verbose=True):
    total_loss = 0
    model_output_list, rating_list = [], []  # Para acumular RMSE para cada lote

    model.train()

    for batch_idx, batch_train in enumerate(train_loader):
        diseases, genes, ei = batch_train['diseases'].to(device), batch_train['genes'].to(device), batch_train['ei'].to(device)

        optimizer.zero_grad()
        output = model(diseases, genes).view(-1, 1)  # Certifique-se de que o modelo retorna no formato certo
        rating = ei.to(torch.float32).detach().view(len(ei), -1)  # Detach e reshape apenas uma vez

        loss = criterion(output, rating)  # Calcular a perda
        total_loss += loss.sum().item()  # Use item() para extrair valor escalar

        loss.backward()  # Executar backpropagation
        optimizer.step()  # Atualizar pesos

        if verbose and (batch_idx % 500 == 0):  # Mostra logs a cada 500 batches
            print(f'Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item()}')

        # Usar o próprio output em vez de `sum` e `item` para maior precisão
        model_output_list.append(output.mean().cpu().item())
        rating_list.append(rating.mean().cpu().item())

    # Calcular média de perda
    avg_loss = total_loss / len(train_loader)

    # Calcular RMSE usando sklearn
    rmse = metrics.mean_squared_error(rating_list, model_output_list, squared=False)

    return avg_loss, rmse


def validate(model, criterion, val_loader, device, verbose=True):
    model.eval()
    total_loss = 0

    model_output_list, rating_list = [], []

    with torch.no_grad():
        for val_batch_idx, batch_val in enumerate(val_loader):
            val_diseases, val_genes, val_ei = batch_val['diseases'].to(device), batch_val['genes'].to(device), batch_val['ei'].to(device)

            val_output = model(val_diseases, val_genes).view(-1, 1)  # Certifique-se de que o modelo retorna no formato certo
            rating = val_ei.to(torch.float32).detach().view(len(val_ei), -1)  # Detach e reshape apenas uma vez

            val_rating = val_ei.view(len(val_ei), -1).to(torch.float32).detach()

            val_loss = criterion(val_output, val_rating)
            total_loss += val_loss.sum().item()

            if verbose and (val_batch_idx % 500 == 0):  # Mostra logs a cada 500 batches
                print(f'Batch {val_batch_idx + 1}/{len(val_loader)} - Loss: {val_loss.item()}')

        # Usar o próprio output em vez de `sum` e `item` para maior precisão
        model_output_list.append(val_output.mean().cpu().item())
        rating_list.append(rating.mean().cpu().item())

   # Calcular média de perda
    avg_loss = total_loss / len(val_loader)

    # Calcular RMSE usando sklearn
    rmse = metrics.mean_squared_error(rating_list, model_output_list, squared=False)

    return avg_loss, rmse

