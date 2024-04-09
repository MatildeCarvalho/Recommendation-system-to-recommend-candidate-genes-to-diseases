import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic, accuracy, SVD
from collections import defaultdict
import time

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        if len(user_ratings) >= k:  
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision_at_k = sum(prec for prec in precisions.values()) / len(precisions)
    recall_at_k = sum(rec for rec in recalls.values()) / len(recalls)

    return precision_at_k, recall_at_k

# Carregar os dados de treinamento e teste
train_data = pd.read_csv('train_filtered.csv')
validation_data = pd.read_csv('data\\validation.csv')
test_data = pd.read_csv('data\\test.csv')


# Convertendo as colunas de 'mappingDOID' e 'geneNID' em conjuntos (sets)
train_diseases_set = set(train_data['mappingDOID'])
validation_diseases_set = set(validation_data['mappingDOID'])

train_genes_set = set(train_data['geneNID'])
validation_genes_set = set(validation_data['geneNID'])

# Verificar se todas as doenças em train estão em validation
if train_diseases_set.issubset(validation_diseases_set):
    print("Todas as doenças em 'train' estão presentes em 'validation'.")
else:
    print("Algumas doenças em 'train' não estão presentes em 'validation'.")

# Verificar se todos os genes em train estão em validation
if train_genes_set.issubset(validation_genes_set):
    print("Todos os genes em 'train' estão presentes em 'validation'.")
else:
    print("Alguns genes em 'train' não estão presentes em 'validation'.")




start_time = time.time()
reader = Reader(rating_scale=(min(train_data['mean_EI']), max(train_data['mean_EI'])))

# Carregar os dados no formato Surprise Dataset
data = Dataset.load_from_df(train_data[['mappingDOID', 'geneNID', 'mean_EI']], reader)

trainset = data.build_full_trainset()
validationset = Dataset.load_from_df(validation_data[['mappingDOID', 'geneNID', 'mean_EI']], reader).build_full_trainset().build_testset()

# Definir os parâmetros que deseja ajustar
ks = [1, 2, 3, 4, 5, 10, 15, 20]
sim_options = {'name': ['msd', 'pearson']}

# Inicializar as variáveis para armazenar os melhores parâmetros e a melhor RMSE
best_rmse = float('inf')
best_params = {}



# Loop para testar diferentes valores de k
for k in ks:
    # Loop para testar diferentes métricas de similaridade
    for sim_name in sim_options['name']:
        # Inicializar o modelo com os parâmetros atuais
        algo = KNNBasic(k=k, sim_options={'name': sim_name, 'user_based': True, "shrinkage": 0})
        
        # Treinar o modelo
        algo.fit(trainset)
        
        # Fazer previsões no conjunto de validação
        predictions = algo.test(validationset)
        
        # Calcular RMSE
        rmse = accuracy.rmse(predictions)
        
        # Atualizar os melhores parâmetros se o RMSE atual for melhor
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'k': k, 'sim_options': {'name': sim_name, 'user_based': True}}

# Treinar o modelo com os melhores parâmetros encontrados usando o conjunto de treinamento completo
best_model = KNNBasic(k=best_params['k'], sim_options=best_params['sim_options'])
best_model.fit(trainset)
print("Melhores parâmetros encontrados:", best_params)

# Fazer previsões no conjunto de teste
testset = list(zip(test_data['mappingDOID'], test_data['geneNID'], test_data['mean_EI']))
predictions = best_model.test(testset)

# Avaliar o desempenho do modelo KNN no conjunto de teste
rmse = accuracy.rmse(predictions)
print(f'Root Mean Squared Error (RMSE) no conjunto de teste: {rmse}')

# Fazer previsões no conjunto de teste
predictions_knn = best_model.test(testset)

# Calcular precisão e recall para os primeiros k itens
k_values = [1, 2, 3, 4, 5]

precision_knn = []
recall_knn = []

for k in k_values:
    precision, recall = precision_recall_at_k(predictions_knn, k=k, threshold=0.973373)
    precision_knn.append(precision)
    recall_knn.append(recall)
    print(f'KNN - Precision@{k}: {precision}')
    print(f'KNN - Recall@{k}: {recall}')
    print('---')

# Avaliar o desempenho do modelo KNN
rmse_knn = accuracy.rmse(predictions_knn)
print(f'KNN - Root Mean Squared Error (RMSE): {rmse_knn}')

end_time = time.time()

# Calcular o tempo de execução
execution_time = end_time - start_time
print("Tempo de execução:", execution_time, "segundos")