import torch
from collections import defaultdict
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error

class EvaluateModel:
    def __init__(self, model, test_loader, lbl_diseases=None, lbl_genes=None):
        self.model = model
        self.test_loader = test_loader
        self.lbl_diseases = lbl_diseases
        self.lbl_genes = lbl_genes

        self.model_output_list = []
        self.target_rating_list = []
        self.user_ratings = defaultdict(list)

        self.evaluate()

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            for i, batched_data in enumerate(self.test_loader): 
                diseases, genes, ei = batched_data
                model_output = self.model(diseases, genes)
                model_output = model_output.view(-1, 1)

                users = batched_data[0]
                movies = batched_data[1]
                ratings = batched_data[2]

                for i in range(len(users)):
                    user_id = users[i].item()
                    movie_id = movies[i].item() 
                    pred_rating = model_output[i].item()
                    true_rating = ratings[i].item()

                    self.model_output_list.append(pred_rating)
                    self.target_rating_list.append(true_rating)
                    self.user_ratings[user_id].append((movie_id, pred_rating, true_rating))

    def calculate_rmse(self, squared=True):
        rms = mean_squared_error(self.target_rating_list, self.model_output_list, squared=True)
        return rms
    
    def calculate_precision_recall_at_k(self, k=5, threshold=0.973373):
        precisions_list = []
        recalls_list = []

        for k in range(1, k+1):  # Valores de K de 1 a 5
            precisions = dict()
            recalls = dict()

            for uid, user_ratings in self.user_ratings.items():
                if len(user_ratings) >= k:
                    user_ratings.sort(key=lambda x: x[1], reverse=True)

                    n_rel = sum((true_r >= threshold) for (_, _, true_r) in user_ratings)
                    n_rec_k = sum((est >= threshold) for (_, est, _) in user_ratings[:k])
                    n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (_, est, true_r) in user_ratings[:k])

                    precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
                    recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

            precision_at_k = sum(prec for prec in precisions.values()) / len(precisions)
            recall_at_k = sum(rec for rec in recalls.values()) / len(recalls)

            precisions_list.append(precision_at_k)
            recalls_list.append(recall_at_k)

        return precisions_list, recalls_list
    
    def write_csv(self, csv_filename=None):
        if not self.user_ratings:
            self.evaluate()

        user_est_true = self.user_ratings
        csv_writer = None

        if csv_filename:
            with open(csv_filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['mappingDOID', 'geneNID', 'Predicted Rating', 'True Rating'])

                for user_id, ratings in user_est_true.items():
                    for rating in ratings:
                        csv_writer.writerow([user_id, rating[0], rating[1], rating[2]])

            if self.lbl_diseases or self.lbl_genes:
                data = pd.read_csv(csv_filename)
                if self.lbl_diseases:
                    data['mappingDOID'] = self.lbl_diseases.inverse_transform(data['mappingDOID'].astype(int))
                if self.lbl_genes:
                    data['geneNID'] = self.lbl_genes.inverse_transform(data['geneNID'].astype(int))
                data.to_csv(csv_filename, index=False)

        print(f"The file '{csv_filename}' was written successfully")


    
