
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(train_path="data/100x100.csv", test_path="data/100x100_2.csv"):
    return pd.read_csv(train_path), pd.read_csv(test_path)


def als(df, rank, iterations, regularization=0.1):
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    matrix = data_matrix.values
    num_users, num_items = matrix.shape
    X = np.ones((num_users, rank)) 
    Y = np.ones((num_items, rank)) 
    mask = matrix > 0

    for _ in range(iterations):
        for i in range(num_users):
            Y_i = Y[mask[i]]
            if Y_i.size == 0:
                continue
            A = Y_i.T @ Y_i + regularization * np.eye(rank)
            b = Y_i.T @ matrix[i, mask[i]]
            X[i] = np.linalg.lstsq(A, b, rcond=None)[0]

        for j in range(num_items):
            X_j = X[mask[:, j]]
            if X_j.size == 0:
                continue
            A = X_j.T @ X_j + regularization * np.eye(rank)
            b = X_j.T @ matrix[mask[:, j], j]
            Y[j] = np.linalg.lstsq(A, b, rcond=None)[0]

    return X, Y


def baseline_model(_lambda, df):
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    delta = df[['u_id', 'a_id', 'score']]
    avg_rating = df['score'].mean()
    num_users, num_anime = data_matrix.shape

    b_u = cp.Variable(num_users)
    b_i = cp.Variable(num_anime)

    user_id_to_index = {u_id: idx for idx, u_id in enumerate(data_matrix.index)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(data_matrix.columns)}

    R_ui = [(avg_rating + b_u[user_id_to_index[row['u_id']]] + b_i[anime_id_to_index[row['a_id']]]) for _, row in delta.iterrows()]
    training_error = [(R_ui[i] - delta.iloc[i]['score'])**2 for i in range(len(delta))]
    
    obj = cp.Minimize(cp.sum(training_error) + _lambda * (cp.sum_squares(b_u) + cp.sum_squares(b_i)))
    prob = cp.Problem(obj)
    prob.solve()

    R = np.zeros((num_users, num_anime))
    for u in range(num_users):
        for i in range(num_anime):
            R[u, i] = avg_rating + b_u.value[u] + b_i.value[i]

    return R


def nuclear_norm_model_df(df):
    user_list = sorted(df['u_id'].unique())
    anime_list = sorted(df['a_id'].unique())

    num_users = len(user_list)
    num_anime = len(anime_list)

    user_id_to_index = {u_id: idx for idx, u_id in enumerate(user_list)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(anime_list)}

    R = cp.Variable((num_users, num_anime))

    delta = df[['u_id', 'a_id', 'score']]
    constraints = []

    for idx in range(len(delta)):
        u = user_id_to_index[int(delta.at[idx, 'u_id'])]
        i = anime_id_to_index[int(delta.at[idx, 'a_id'])]
        score = delta.at[idx, 'score']
        constraints.append(R[u, i] == score)

    obj = cp.Minimize(cp.normNuc(R))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)

    if prob.status == cp.OPTIMAL:
        print("Optimization succeeded.")
        return R.value
    else:
        print(f"Optimization failed with status: {prob.status}")
        return None

#This takes in matrix input
def nuclear_norm_model_matrix(data_matrix):
    num_users, num_anime = data_matrix.shape

    R = cp.Variable((num_users, num_anime))

    constraints = []
    for i in range(num_users):
        for j in range(num_anime):
            if data_matrix[i, j] != 0:
                constraints.append(R[i, j] == data_matrix[i, j])

    objective = cp.Minimize(cp.normNuc(R))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if problem.status == cp.OPTIMAL:
        print("Optimization succeeded.")
        return R.value
    else:
        print(f"Optimization failed. Status: {problem.status}")
        return None

def spectral_regularization_model(_lambda, df):
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    delta = df[['u_id', 'a_id', 'score']]
    num_users, num_anime = data_matrix.shape

    R = cp.Variable((num_users, num_anime))
    user_id_to_index = {u_id: idx for idx, u_id in enumerate(data_matrix.index)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(data_matrix.columns)}

    training_error = [(R[user_id_to_index[row['u_id']], anime_id_to_index[row['a_id']]] - row['score']) ** 2 for _, row in delta.iterrows()]
    obj = cp.Minimize(0.5 * cp.sum(training_error) + _lambda * cp.norm(R, "nuc"))
    prob = cp.Problem(obj)
    prob.solve()

    return R.value


def recommend_anime(R, u_id, df, x=5):
    original_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    R_df = pd.DataFrame(R, index=original_matrix.index, columns=original_matrix.columns)
    user_row = original_matrix.loc[u_id]
    user_not_watched = user_row[user_row == 0].index.tolist()
    top_x = R_df.loc[u_id, user_not_watched].sort_values(ascending=False).head(x)
    anime_ids = top_x.index
    anime_names = df[df['a_id'].isin(anime_ids)]['title'].unique()
    return pd.DataFrame({'title': anime_names, 'predicted_rating': top_x.values})


def compute_rmse(R, df):
    matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    delta = df[['u_id', 'a_id', 'score']]
    user_id_to_index = {u_id: idx for idx, u_id in enumerate(matrix.index)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(matrix.columns)}
    errors = [(R[user_id_to_index[row['u_id']], anime_id_to_index[row['a_id']]] - row['score']) ** 2 for _, row in delta.iterrows()]
    return np.sqrt(np.mean(errors))
