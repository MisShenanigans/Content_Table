import pandas as pd
import numpy as np
import cvxpy as cp

def load_data(path="data/100x100.csv"):
    df = pd.read_csv(path)
    matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    return df, matrix

def als(df_or_matrix, rank, iterations, regularization=0.1):
    if isinstance(df_or_matrix, pd.DataFrame):
        if 'u_id' in df_or_matrix.columns and 'a_id' in df_or_matrix.columns and 'score' in df_or_matrix.columns:
            matrix = df_or_matrix.pivot(index='u_id', columns='a_id', values='score').fillna(0).values
        else:
            matrix = df_or_matrix.values
    else:
        matrix = df_or_matrix

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

def recommend_anime(R, u_id, original_df, x=5):
    pivot_matrix = original_df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    user_idx = pivot_matrix.index.get_loc(u_id)
    user_ratings = pivot_matrix.loc[u_id]

    user_pred_row = pd.Series(R[user_idx], index=pivot_matrix.columns)
    unwatched = user_ratings[user_ratings == 0].index
    top_x_pred = user_pred_row.loc[unwatched].sort_values(ascending=False).head(x)

    title_lookup = original_df.drop_duplicates(subset=["a_id"])[["a_id", "title"]].set_index("a_id")
    recommendations = pd.DataFrame({
        "a_id": top_x_pred.index,
        "predicted_rating": top_x_pred.values,
        "title": title_lookup.loc[top_x_pred.index]["title"].values
    })

    return recommendations.reset_index(drop=True)

def baseline_model(df, regularization=1.0):
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    delta = df[['u_id', 'a_id', 'score']]
    avg_rating = df['score'].mean()
    num_users, num_anime = data_matrix.shape

    b_u = cp.Variable(num_users)
    b_i = cp.Variable(num_anime)

    user_id_to_index = {u_id: idx for idx, u_id in enumerate(data_matrix.index)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(data_matrix.columns)}

    training_error = []
    for idx, row in delta.iterrows():
        u = user_id_to_index[row['u_id']]
        i = anime_id_to_index[row['a_id']]
        pred = avg_rating + b_u[u] + b_i[i]
        training_error.append((pred - row['score']) ** 2)

    loss = cp.sum(training_error)
    reg = cp.sum_squares(b_u) + cp.sum_squares(b_i)
    prob = cp.Problem(cp.Minimize(loss + regularization * reg))
    prob.solve()

    if prob.status != cp.OPTIMAL:
        raise ValueError("Baseline model optimization failed.")

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

    for _, row in delta.iterrows():
        u = user_id_to_index[row['u_id']]
        i = anime_id_to_index[row['a_id']]
        constraints.append(R[u, i] == row['score'])

    prob = cp.Problem(cp.Minimize(cp.normNuc(R)), constraints)
    prob.solve(solver=cp.SCS)

    if prob.status == cp.OPTIMAL:
        print("Nuclear norm model succeeded.")
        return R.value
    else:
        print(f"Nuclear norm model failed: {prob.status}")
        return None

def nuclear_norm_model_matrix(data_matrix):
    num_users, num_anime = data_matrix.shape
    R = cp.Variable((num_users, num_anime))
    constraints = [R[i, j] == data_matrix[i, j]
                   for i in range(num_users)
                   for j in range(num_anime) if data_matrix[i, j] != 0]
    prob = cp.Problem(cp.Minimize(cp.normNuc(R)), constraints)
    prob.solve(solver=cp.SCS)

    if prob.status == cp.OPTIMAL:
        print("Nuclear norm model (matrix input) succeeded.")
        return R.value
    else:
        print(f"Nuclear norm model failed: {prob.status}")
        return None

def spectral_regularization_model(_lambda, df):
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    delta = df[['u_id', 'a_id', 'score']]
    num_users, num_anime = data_matrix.shape

    R = cp.Variable((num_users, num_anime))
    user_id_to_index = {u_id: idx for idx, u_id in enumerate(data_matrix.index)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(data_matrix.columns)}

    training_error = []
    for _, row in delta.iterrows():
        u = user_id_to_index[row['u_id']]
        i = anime_id_to_index[row['a_id']]
        training_error.append((R[u, i] - row['score']) ** 2)

    loss = 0.5 * cp.sum(training_error)
    reg = _lambda * cp.norm(R, "nuc")
    prob = cp.Problem(cp.Minimize(loss + reg))
    prob.solve()

    if prob.status == cp.OPTIMAL:
        print("Spectral regularization model succeeded.")
        return R.value
    else:
        print(f"Spectral regularization model failed: {prob.status}")
        return None

def RMSE(R, df):
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
    delta = df[['u_id', 'a_id', 'score']]
    N = len(delta)

    user_id_to_index = {u_id: idx for idx, u_id in enumerate(data_matrix.index)}
    anime_id_to_index = {a_id: idx for idx, a_id in enumerate(data_matrix.columns)}

    squared_errors = []
    for _, row in delta.iterrows():
        u = user_id_to_index[row['u_id']]
        i = anime_id_to_index[row['a_id']]
        squared_errors.append((R[u, i] - row['score']) ** 2)

    return np.sqrt(np.sum(squared_errors) / N)

def top_k_largest(matrix, k=5):
    flat_indices = np.argpartition(matrix.ravel(), -k)[-k:]
    sorted_indices = flat_indices[np.argsort(matrix.ravel()[flat_indices])[::-1]]
    return [(matrix.flat[idx], np.unravel_index(idx, matrix.shape)) for idx in sorted_indices]
