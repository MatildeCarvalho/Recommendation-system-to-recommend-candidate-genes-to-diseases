from model import RecSysModel, train, validate
from data_processing import process_data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from evaluate import calculate_rmse, calculate_precision_recall, build_user_est_true

import numpy as np

if __name__ == "__main__":
    # Processamento dos dados e criação dos datasets
    train_dataset, val_dataset, test_dataset, lbl_diseases, lbl_genes = process_data("DATA\data\DiseaseGene.db")

    # Verificar se há disponibilidade de GPU e definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Carregamento dos dados utilizando DataLoader
    batch_size = 6
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Enviar os dados para a GPU, se disponível
    train_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in train_loader]
    val_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in val_loader]
    test_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in test_loader]


    model = RecSysModel(n_diseases=len(lbl_diseases.classes_), 
                        n_genes=len(lbl_genes.classes_), 
                        n_factors=16).to(device)


    # Definição das variáveis necessárias
    #criterion = nn.MSELoss()  # Defina o critério de perda aqui
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Defina o otimizador aqui
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    loss_func = nn.MSELoss() # Defina o critério de perda aqui

    # Defina o número máximo de épocas e o limite para o "early stopping"
    epochs = 7
    patience = 1  # Limite para parar se a validação não melhorar após 'patience' épocas
    best_val_loss = float('inf')  # Inicialize a melhor perda de validação como infinito
    counter = 0  # Contador para controlar o "early stopping"

    train_losses = []
    val_losses = []
    train_rmse = []
    val_rmse = []

    for epoch_i in range(epochs):
        train_loss = train(model, optimizer, loss_func, train_loader, device)
        val_loss = validate(model, loss_func, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Calculando RMSE
        model.eval()
        with torch.no_grad():
            train_rmse_values = []
            val_rmse_values = []
            
            for batch_idx, (diseases, genes, ei) in enumerate(train_loader):
                diseases, genes, ei = diseases.to(device), genes.to(device), ei.to(device)
                output = model(diseases, genes)
                output = output.view(-1, 1) 
                rating = ei.view(len(ei), -1).to(torch.float32).detach()
                train_rmse_values.append(model.rmse(output, rating).item())

            for val_batch_idx, (val_diseases, val_genes, val_ei) in enumerate(val_loader):
                val_diseases, val_genes, val_ei = val_diseases.to(device), val_genes.to(device), val_ei.to(device)
                val_output = model(val_diseases, val_genes)
                val_output = val_output.view(-1, 1)
                val_rating = val_ei.view(len(val_ei), -1).to(torch.float32).detach()
                val_rmse_values.append(model.rmse(val_output, val_rating).item())
        
        train_rmse.append(np.mean(train_rmse_values))
        val_rmse.append(np.mean(val_rmse_values))

        # Liberar a memória da GPU
        torch.cuda.empty_cache()
        
        print(f"Epoch [{epoch_i + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train RMSE: {train_rmse[-1]:.4f}, Validation RMSE: {val_rmse[-1]:.4f}")

        # "Early stopping" - Verifica se a perda de validação melhorou
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reseta o contador se a perda de validação melhorou
        else:
            counter += 1  # Incrementa o contador se a perda de validação não melhorou

        if counter >= patience:
            print(f'Early stopping at epoch {epoch_i + 1} as there is no improvement in validation loss.')
            break  # Sai do loop se não houver melhoria por 'patience' épocas

        # Chama o scheduler após cada época
        scheduler.step()

    # Plotando o gráfico de Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('DATA\\TABELAS\\graficos\\loss.png')
    plt.show()

    # Plotando o gráfico de RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Train and Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('DATA\\TABELAS\\graficos\\rsme.png')
    plt.show()

    # Salvar o modelo em um arquivo
    caminho_arquivo = 'DATA\model\model.pth'  # Nome do arquivo onde o modelo será salvo
    torch.save(model.state_dict(), caminho_arquivo)


    rms = calculate_rmse(model, test_loader)
    print(f"Root Mean Squared Error (RMSE): {rms}")

    precisions, recalls = calculate_precision_recall(model, test_loader, k=5, threshold=0.973373)
    print(f"Precisions: {precisions}")
    print(f"Recalls: {recalls}")

    k_values = range(1, 6)  # Valores de k de 1 a 5

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, precisions, label='Precision', marker='o')
    plt.plot(k_values, recalls, label='Recall', marker='o')
    plt.xticks(k_values)
    #plt.title('Precision and Recall vs. k')
    plt.xlabel('k')
    plt.ylabel('Metric Value')
    plt.legend()
    #plt.grid(True)
    plt.savefig('DATA\TABELAS\graficos\precissionrecall.png')  # Salva o gráfico como um arquivo PNG
    plt.show()

    # Exemplo de chamada da função
    user_est_true = build_user_est_true(model, test_loader, lbl_diseases, lbl_genes, verbose=False, csv_filename='DATA\\data\\results.csv')

##
