from sklearn.metrics import mean_squared_error
import torch
from collections import defaultdict
import csv
import pandas as pd

def calculate_rmse(model, test_loader):
    model_output_list = []
    target_rating_list = []

    model.eval()

    with torch.no_grad():
        for i, batched_data in enumerate(test_loader): 
            diseases, genes, ei = batched_data
            model_output = model(diseases, genes)
            model_output = model_output.view(-1, 1)

            model_output_list.append(model_output.sum().item() / len(diseases))
            target_rating = ei
            
            target_rating_list.append(target_rating.sum().item() / len(diseases))

    rms = mean_squared_error(target_rating_list, model_output_list, squared=False)
    return rms

def calculate_precision_recall(model, test_loader, k=5, threshold=0.95):
    user_est_true = defaultdict(list)

    with torch.no_grad():
        for i, batched_data in enumerate(test_loader): 
            users = batched_data[0]
            movies = batched_data[1]
            ratings = batched_data[2]
            
            model_output = model(users, movies)

            for i in range(len(users)):
                user_id = users[i].item()
                movie_id = movies[i].item() 
                pred_rating = model_output[i].item()
                true_rating = ratings[i].item()
                
                user_est_true[user_id].append((pred_rating, true_rating))

        precisions_list = []
        recalls_list = []

        for k in range(1, k+1):  # Valores de K de 1 a 5
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

            precisions_list.append(precision_at_k)
            recalls_list.append(recall_at_k)

        return precisions_list, recalls_list


def build_user_est_true(model, test_loader, lbl_diseases=None, lbl_genes=None, verbose=False, csv_filename=None):
    user_est_true = defaultdict(list)
    csv_writer = None

    if csv_filename:
        csvfile = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['mappingDOID', 'geneNID', 'Predicted Rating', 'True Rating'])

    with torch.no_grad():
        for i, batched_data in enumerate(test_loader): 
            users = batched_data[0]
            movies = batched_data[1]
            ratings = batched_data[2]
            
            model_output = model(users, movies)

            for i in range(len(users)):
                user_id = users[i].item()
                movie_id = movies[i].item() 
                pred_rating = model_output[i].item()
                true_rating = ratings[i].item()


                if verbose:
                    print(f"{user_id}, {movie_id}, {pred_rating}, {true_rating}")  # Print se verbose for True
                
                if csv_writer:  # Adiciona no CSV se verbose for True
                    csv_writer.writerow([user_id, movie_id, pred_rating, true_rating])
                
                user_est_true[user_id].append((movie_id, pred_rating, true_rating))

    # Salvar em um arquivo CSV se o nome do arquivo for fornecido
    if csv_writer:
        csvfile.close()
        data = pd.read_csv(csv_filename)
        if lbl_diseases:
            data['mappingDOID'] = lbl_diseases.inverse_transform(data['mappingDOID'].astype(int))
        if lbl_genes:
            data['geneNID'] = lbl_genes.inverse_transform(data['geneNID'].astype(int))
        data.to_csv(csv_filename, index=False)

    return user_est_true










