# import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree
from torch.utils.data import DataLoader

import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import seaborn as sns

# Read the useful datasets
test_data = pd.read_csv("data/test_dataset.csv")
item_dataset = pd.read_csv("data/item_dataset.csv")

class MovieLensNet(nn.Module):
    def __init__(self, num_movies, num_users, num_genres_encoded,
                 embedding_size, hidden_dim):
        super(MovieLensNet, self).__init__()
        self.num_movies = num_movies
        self.num_users = num_users
        self.num_genres_encoded = num_genres_encoded
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2 + num_genres_encoded, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, movie_id, user_id, genre_id):
        genre_id = torch.unsqueeze(genre_id, dim=2)
        if genre_id.size() != (movie_id.size(0), self.num_genres_encoded, 1):
            raise ValueError(f"Expected genre_id to have size ({movie_id.size(0)}, {self.num_genres_encoded}, 1)")
        movie_emb = self.movie_embedding(movie_id)
        user_emb = self.user_embedding(user_id)
        movie_emb = torch.unsqueeze(movie_emb, dim=2)
        user_emb = torch.unsqueeze(user_emb, dim=2)
        x = torch.cat([movie_emb, user_emb, genre_id.float()], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, data, movies, genres_encoded, mlb, max_genre_count, num_users):
        self.data = data
        self.movies = movies
        self.genres_encoded = genres_encoded
        self.mlb = mlb
        self.max_genre_count = max_genre_count
        self.num_users = num_users

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        movie_id = torch.tensor(row["item_id"], dtype=torch.long)
        user_id = torch.tensor(row["user_id"], dtype=torch.long)
        if user_id.min() < 0 or user_id.max() > self.num_users:
            print('self.num_users = ', self.num_users)
            raise ValueError(f"Invalid user ID: {user_id}")
        movie_genres = self.movies.loc[self.movies['movie_id'] == row['item_id'], 'genres'].iloc[0]
        genre_indices = []
        for genre in movie_genres.split('|'):
            if genre in self.mlb.classes_:
                genre_indices.append(np.where(self.mlb.classes_ == genre)[0][0])
        if len(genre_indices) == 0:
            genre_indices.append(0)
        genre_id = torch.tensor(genre_indices, dtype=torch.long)
        genre_id = torch.flatten(genre_id)[:self.max_genre_count]
        genre_pad = torch.zeros(self.max_genre_count - genre_id.shape[0], dtype=torch.long)
        genre_id = torch.cat([genre_id, genre_pad])
        rating = torch.tensor(row["rating"], dtype=torch.float)
        return {"movie_id": movie_id, "user_id": user_id, "genre_id": genre_id, "rating": rating}

vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
movie_genres = vectorizer.fit_transform(item_dataset['genres'])
mlb = MultiLabelBinarizer(sparse_output=True)
genres_encoded = mlb.fit_transform(list(vectorizer.vocabulary_.keys()))
genres_encoded = genres_encoded.astype(np.float32)
num_genres = 19
num_users = 943
num_movies = 1683

test_dataset = MovieLensDataset(test_data, item_dataset, genres_encoded, mlb, max_genre_count=num_genres, num_users=num_users)

# Evaluate the model on the test set
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
criterion = nn.MSELoss()

model = MovieLensNet(num_movies, num_users, num_genres, embedding_size=32, hidden_dim=64)

state_dict = torch.load('../models/model_parameters.pth')
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    total_loss = 0.0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Collect the predicted ratings and true ratings
    all_predicted_ratings = []
    all_true_ratings = []

    for i, batch in tqdm(enumerate(test_loader)):
        movie_id = batch['movie_id']
        user_id = batch['user_id']
        genre_id = batch['genre_id']
        rating = batch['rating']
        output = model(movie_id, user_id, genre_id)
        loss = criterion(output, rating)
        total_loss += loss.item()
        predicted_labels = (output >= 2).float()  # Threshold at 4 or higher
        tp += ((predicted_labels == 1) & (rating >= 3.5)).sum().item()
        fp += ((predicted_labels == 1) & (rating < 3.5)).sum().item()
        tn += ((predicted_labels == 0) & (rating < 3.5)).sum().item()
        fn += ((predicted_labels == 0) & (rating >= 3.5)).sum().item()

        all_predicted_ratings.extend(output.squeeze().tolist())
        all_true_ratings.extend(rating.squeeze().tolist())

    # Plot distribution of true and predicted ratings
    plt.hist(all_true_ratings, bins=np.arange(0, 6, 0.5), alpha=0.5, label='True ratings')
    plt.hist(all_predicted_ratings, bins=np.arange(0, 6, 0.5), alpha=0.5, label='Predicted ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    # Plot distribution of prediction errors
    errors = np.array(all_true_ratings) - np.array(all_predicted_ratings)
    plt.hist(errors, bins=np.arange(-3, 4, 0.5))
    plt.xlabel('Prediction error')
    plt.ylabel('Count')
    plt.show()

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix((np.array(all_true_ratings) >= 3).astype(int), (np.array(all_predicted_ratings) >= 3).astype(int))
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    plt.show()

    avg_loss = total_loss / len(test_loader)
    print('Test RMSE: %.3f' % np.sqrt(avg_loss))
    precision = tp / (tp + fp+0.000001)*100
    recall = tp / (tp + fn+0.000001)*100
    f1 = 2 * precision * recall / (precision + recall+0.000001)
    print('Test precision: %.3f' % precision + ' %')
    print('Test recall: %.3f' % recall + ' %')
    print('Test F1 score: %.3f' % f1 + ' %')
    # Compute the optimal threshold
    thresholds = np.arange(2, 5.0, 0.1)
    f1_scores = []
    for threshold in thresholds:
        predicted_labels = (torch.Tensor(all_predicted_ratings) >= threshold).float()
        tp = ((predicted_labels == 1) & (torch.Tensor(all_true_ratings) >= 3.5)).sum().item()
        fp = ((predicted_labels == 1) & (torch.Tensor(all_true_ratings) < 3.5)).sum().item()
        tn = ((predicted_labels == 0) & (torch.Tensor(all_true_ratings) < 3.5)).sum().item()
        fn = ((predicted_labels == 0) & (torch.Tensor(all_true_ratings) >= 3.5)).sum().item()
        precision = tp / (tp + fp+0.000001)
        recall = tp / (tp + fn+0.000001)
        f1 = 2 * precision * recall / (precision + recall+0.000001)
        f1_scores.append(f1)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    # print('Optimal threshold:', optimal_threshold)