#  Content_Table

we chose animes with at least 10 reviews and users with at least 3 reviews for accuracy 

since the matrix is too large, I cannot push it to git. instead, use the code below to create the matrix (shouldn't take too long, at least I hope so)
```python
    df = pd.read_csv("data/final_data.csv")
    data_matrix = df.pivot(index='u_id', columns='a_id', values='score').fillna(0)
```
find user id of the row i using 
```python
user_id = data_matrix.index[i]
```
find anime id of the row j using 
```python
anime_id = data_matrix.columns[i]
```

find the column of anime with a_id x using
```python 
column = data_matrix.columns.get_loc(x)
```

find the row of user with u_id x using
```python 
row = data_matrix.index.get_loc(x)
```
