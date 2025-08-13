'''import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Dictionary to store unique values of each column
unique_values = {}

# Iterate over each column and get unique values
for column in df.columns:
    unique_values[column] = df[column].unique().tolist()

# Print the unique values for each column
for column, values in unique_values.items():
    print(f"Unique values in '{column}':")
    print(values)
    print()
'''

'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(subset=['TotalCharges'], inplace=True)

features = data.drop(columns=['customerID', 'Churn', 'TotalCharges'])
target = data['TotalCharges']

categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = model.score(X_test, y_test)
print(f'R^2 score: {score}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted TotalCharges')
plt.show()
'''

import math
from scipy.spatial import distance

def read_input():
    # Using hardcoded input for easier debugging
    input = """3 2 2
0 0
1 0
2 0
3 0"""
    
    input = input.strip().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    B = int(input[idx])
    idx += 1
    R = int(input[idx])
    idx += 1

    blueberry_plants = []
    for _ in range(B):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        blueberry_plants.append((x, y))

    redberry_plants = []
    for _ in range(R):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        redberry_plants.append((x, y))

    return N, B, R, blueberry_plants, redberry_plants

def can_match(N, B, R, blueberry_plants, redberry_plants, dist_threshold):
    from collections import deque
    
    graph = [[] for _ in range(B)]
    for i in range(B):
        for j in range(R):
            if distance.euclidean(blueberry_plants[i], redberry_plants[j]) >= dist_threshold:
                graph[i].append(j)

    match_r = [-1] * R

    def bpm(u, seen):
        for v in graph[u]:
            if not seen[v]:
                seen[v] = True
                if match_r[v] == -1 or bpm(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    result = 0
    for i in range(B):
        seen = [False] * R
        if bpm(i, seen):
            result += 1
        if result >= N:
            return True

    return False

def maximize_minimal_distance(N, B, R, blueberry_plants, redberry_plants):
    low = 0
    high = max(distance.euclidean(blueberry_plants[i], redberry_plants[j]) for i in range(B) for j in range(R))
    result = 0

    while high - low > 1e-7:
        mid = (low + high) / 2
        if can_match(N, B, R, blueberry_plants, redberry_plants, mid):
            result = mid
            low = mid
        else:
            high = mid

    return result

def main():
    N, B, R, blueberry_plants, redberry_plants = read_input()
    result = maximize_minimal_distance(N, B, R, blueberry_plants, redberry_plants)
    print(f"{result:.6f}")

if __name__ == "__main__":
    main()
