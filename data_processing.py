import pandas as pd
import sqlite3
import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import os
class DiseaseGeneDatase(Dataset):
    def __init__(self, diseases, genes, ei):
        self.diseases = diseases
        self.genes = genes
        self.ei = ei

    def __len__(self):
        return len(self.diseases)

    def __getitem__(self, idx):
        return {
            "diseases": torch.tensor(self.diseases[idx], dtype=torch.long),
            "genes": torch.tensor(self.genes[idx], dtype=torch.long),
            "ei": torch.tensor(self.ei[idx], dtype=torch.float),
        }

    
def load_data_from_sqlite(sqlite_file):
    with sqlite3.connect(sqlite_file) as conn:
        query = """SELECT mappingDOID, geneNID, EI FROM DiseaseGeneNetwork;"""
        df = pd.read_sql_query(query, conn)

    return df

def preprocess_data(df):
    df_grouped = df.groupby(['mappingDOID', 'geneNID'])['EI'].agg(['sum', 'mean']).reset_index()
    df_grouped.columns = ['mappingDOID', 'geneNID', 'sum_EI', 'mean_EI']
    
    df_grouped.loc[df_grouped['sum_EI'] == 0.0, 'mean_EI'] = 0.0

    df_grouped.dropna(inplace=True)
    print(df_grouped.head())
    print(df_grouped.info())

    return df_grouped

def split_data_by_doid(df):
    df = df.dropna(subset=['mappingDOID', 'geneNID', 'sum_EI', 'mean_EI'])
    unique_doids = df['mappingDOID'].unique()
    train_data = []
    val_data = []
    test_data = []

    for doid in unique_doids:
        doid_data = df[df['mappingDOID'] == doid]
        num_samples = len(doid_data)
        
        if num_samples == 1:
            train_data.append(doid_data.copy())  # Cópia do DataFrame - vai para o conjunto de treino
        elif num_samples == 2:
            train_data.append(doid_data.iloc[:1].copy())  # Primeira linha para o conjunto de treino
            test_data.append(doid_data.iloc[1:].copy())   # Segunda linha para o conjunto de teste
        elif num_samples == 3:
            train_data.append(doid_data.iloc[:2].copy())  # Duas primeiras linhas para o conjunto de treino
            test_data.append(doid_data.iloc[2:].copy())   # Terceira linha para o conjunto de teste
        else:
            train, remaining = train_test_split(doid_data, test_size=0.2, random_state=2)
            if len(remaining) == 1:  # Se sobrou apenas uma amostra
                train_data.append(train.copy())  # Adiciona ao conjunto de treino
                test_data.append(remaining.copy())   # Adiciona ao conjunto de teste
            else:
                val, test = train_test_split(remaining, test_size=0.5, random_state=2)
                train_data.append(train.copy())
                val_data.append(val.copy())
                test_data.append(test.copy())

    return train_data, val_data, test_data


def create_datasets(train_df, val_df, test_df):
    #print(train_df['mappingDOID'].values)
    #print(type(train_df['mappingDOID'].values))
    train_dataset = DiseaseGeneDatase(train_df['mappingDOID'].values, train_df['geneNID'].values, train_df['mean_EI'].values)
    # print(f'Tamanho do conjunto de dados de treinamento: {train_dataset.__len__()}')
    val_dataset = DiseaseGeneDatase(val_df['mappingDOID'].values, val_df['geneNID'].values, val_df['mean_EI'].values)
    test_dataset = DiseaseGeneDatase(test_df['mappingDOID'].values, test_df['geneNID'].values, test_df['mean_EI'].values)

    return train_dataset, val_dataset, test_dataset

def save_datasets(train_df, val_df, test_df, lbl_diseases, lbl_genes):
    # Ensure the directory exists
    output_dir = 'data\\file_storage'
    os.makedirs(output_dir, exist_ok=True)

    # Convertendo a coluna 'geneNID' para string e removendo o ponto decimal
    # Converta a coluna 'geneNID' para valores numéricos, e os não numéricos serão convertidos para NaN
    train_df['geneNID'] = pd.to_numeric(train_df['geneNID'], errors='coerce')
    test_df['geneNID'] = pd.to_numeric(test_df['geneNID'], errors='coerce')

    # Remover linhas com valores NaN na coluna 'geneNID' do dataframe de treino
    train_df.dropna(subset=['geneNID'], inplace=True)

    # Remover linhas com valores NaN na coluna 'geneNID' do dataframe de teste
    test_df.dropna(subset=['geneNID'], inplace=True)
    val_df.dropna(subset=['geneNID'], inplace=True)

    # Após lidar com os valores NaN, você pode converter a coluna 'geneNID' para string
    # Convertendo os valores de 'geneNID' para string e removendo o '.0'
    train_df['geneNID'] = train_df['geneNID'].astype(int).astype(str).str.replace('\.0', '', regex=True)
    test_df['geneNID'] = test_df['geneNID'].astype(int).astype(str).str.replace('\.0', '', regex=True)

    # decode 
    # Decodificação dos rótulos para valores originais usando os dados codificados
    train_df['mappingDOID'] = lbl_diseases.inverse_transform(train_df['mappingDOID'].astype(int))
    train_df['geneNID'] = lbl_genes.inverse_transform(train_df['geneNID'].astype(int))

    val_df['mappingDOID'] = lbl_diseases.inverse_transform(val_df['mappingDOID'].astype(int))
    val_df['geneNID'] = lbl_genes.inverse_transform(val_df['geneNID'].astype(int))
    
    test_df['mappingDOID'] = lbl_diseases.inverse_transform(test_df['mappingDOID'].astype(int))
    test_df['geneNID'] = lbl_genes.inverse_transform(test_df['geneNID'].astype(int))

    # Removendo a coluna '0' se existir
    if '0' in train_df.columns:
        train_df.drop(columns=['0'], inplace=True)
    if '0' in test_df.columns:
        test_df.drop(columns=['0'], inplace=True)

    # Salvando apenas as colunas necessárias
    train_df[['mappingDOID', 'geneNID', 'sum_EI', 'mean_EI']].to_csv('data\\file_storage\\train.csv', index=False)
    val_df[['mappingDOID', 'geneNID', 'sum_EI', 'mean_EI']].to_csv('data\\file_storage\\validation.csv', index=False)
    test_df[['mappingDOID', 'geneNID', 'sum_EI', 'mean_EI']].to_csv('data\\file_storage\\test.csv', index=False)

def process_data(sqlite_file):
    df = load_data_from_sqlite(sqlite_file)

    # Codificação dos rótulos de classes
    lbl_diseases = preprocessing.LabelEncoder()
    lbl_genes = preprocessing.LabelEncoder()
    df['mappingDOID'] = lbl_diseases.fit_transform(df['mappingDOID'])
    df['geneNID'] = lbl_genes.fit_transform(df['geneNID'])

    df_processed = preprocess_data(df)
    train_data_list, val_data_list, test_data_list = split_data_by_doid(df_processed)
    
    # Convertendo as listas de DataFrames para DataFrames únicos concatenando-os
    train_df = pd.concat(train_data_list)
    val_df = pd.concat(val_data_list)
    test_df = pd.concat(test_data_list)

    train_dataset, val_dataset, test_dataset = create_datasets(train_df, val_df, test_df)
    save_datasets(train_df, val_df, test_df, lbl_diseases, lbl_genes)
    
    return train_dataset, val_dataset, test_dataset, lbl_diseases, lbl_genes


# if __name__ == "__main__":
    # Processamento dos dados e criação dos datasets
    # train_dataset, val_dataset, test_dataset, lbl_diseases, lbl_genes = process_data("DATA\data\DiseaseGene.db")

    # Verificação do primeiro exemplo no dataset de treino
    example = train_dataset[0]
    print("Example from the training dataset:")
    print("Diseases:", example['diseases'])
    print("Genes:", example['genes'])
    print("EI:", example['ei'])

    # Verificar se há disponibilidade de GPU e definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Carregamento dos dados utilizando DataLoader
    batch_size = 6
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Enviar os dados para o dispositivo (GPU ou CPU)
    train_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in train_loader]
    val_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in val_loader]
    test_loader = [(batch['diseases'].to(device), batch['genes'].to(device), batch['ei'].to(device)) for batch in test_loader]

    # Carregar dados novamente para inspeção das dimensões e tipos de dados
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Verificar um batch de dados de treinamento para inspecionar dimensões e tipos de dados
    for batch_idx, batch in enumerate(train_loader):
        diseases, genes, ei = batch['diseases'], batch['genes'], batch['ei']
        
        # Exibir as dimensões dos dados
        print("\nDimensions of the training batch", batch_idx, ":")
        print("Diseases shape:", diseases.shape)
        print("Genes shape:", genes.shape)
        print("EI shape:", ei.shape)
        
        # Verificar os tipos de dados
        print("Diseases type:", diseases.dtype)
        print("Genes type:", genes.dtype)
        print("EI type:", ei.dtype)
        
        # Passar os dados para o modelo (substitua isso pela parte do seu código onde o modelo é treinado)
        #output = model(diseases, genes)
