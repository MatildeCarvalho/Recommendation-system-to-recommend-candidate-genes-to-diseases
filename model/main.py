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
from evaluate import EvaluateModel
import joblib


import numpy as np

if __name__ == "__main__":
    # Processamento dos dados e criação dos datasets
    train_dataset, val_dataset, test_dataset, lbl_diseases, lbl_genes = process_data("data\DiseaseGene.db")


    # Verificar se há disponibilidade de GPU e definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Carregamento dos dados utilizando DataLoader
    batch_size = 6
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Enviar os dados para a GPU, se disponível
    #train_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in train_loader]
    #val_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in val_loader]
    #test_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in test_loader]


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
    train_rmses = []
    val_rmses = []

    for epoch_i in range(epochs):
        train_loss, train_rmse = train(model, optimizer, loss_func, train_loader, device, verbose=False)
        val_loss, val_rmse = validate(model, loss_func, val_loader, device, verbose=False)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        # Liberar a memória da GPU
        torch.cuda.empty_cache()
        
        print(f"Epoch [{epoch_i + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")

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
    plt.savefig('model\\storage\\images\\loss.png')
    plt.show()

    # Plotando o gráfico de RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(val_rmses, label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Train and Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('model\\storage\\images\\rsme.png')
    plt.show()

    # Salvar o modelo em um arquivo
    caminho_arquivo = 'model\storage\model.pth'  # Nome do arquivo onde o modelo será salvo
    torch.save(model.state_dict(), caminho_arquivo)





    # Supondo que você já tenha importado as bibliotecas necessárias e criado seu modelo e conjunto de dados de teste

    # Crie uma instância de EvaluateModel
    evaluator = EvaluateModel(model, test_loader, lbl_diseases, lbl_genes, device)

    # Calcular RMSE
    rms = evaluator.calculate_rmse()
    print(f"Root Mean Squared Error (RMSE): {rms}")

    # Calcular precisão e recall em k
    precisions, recalls = evaluator.calculate_precision_recall_at_k(k=5, threshold=0.973373)
    print(f"Precisions: {precisions}")
    print(f"Recalls: {recalls}")
 

    # Construir e salvar as estimativas do usuário
    user_estimations = evaluator.write_csv(csv_filename="data\\file_storage\\results.csv")


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
    plt.savefig('model\\storage\\images\\precissionrecall.png')  # Salva o gráfico como um arquivo PNG
    plt.show()

    joblib.dump(lbl_diseases, 'model\storage\lbl_diseases.pkl')
    joblib.dump(lbl_genes, 'model\storage\lbl_genes.pkl')

    # Suponha que você já tenha criado uma instância de EvaluateModel chamada evaluator

    raw_disease_id = "1234"

    # Obter as top 5 recomendações para essa doença
    top_k = evaluator.top_k_recommendations(raw_disease_id, k=5)

    # Imprimir as recomendações
    print("Top 5 recomendações para a doença", raw_disease_id)
    for i, (gene_id, predicted_rating) in enumerate(top_k):
        print(f"{i+1}. Gene: {gene_id}, Rating Previsto: {predicted_rating:.2f}")








