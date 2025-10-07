import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# Load main dataset
df = pd.read_csv("cleaned_movies_metadata.csv")

# Load and merge Rotten Tomatoes data
rt = pd.read_csv("rotten_tomatoes_movies.csv")
df = df.merge(rt[['title', 'rating']], how='left', on='title')
df['rating'] = df['rating'].astype('category').cat.codes

# Load and merge credits data
credits = pd.read_csv("credits.csv")
df = df.merge(credits[['id', 'cast', 'crew']], how='left', on='id')

# Parse cast and crew features
def parse_cast(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        return len([c for c in cast_list if c.get('order', 999) < 5])
    except:
        return 0

def parse_crew(crew_str, department):
    try:
        crew_list = ast.literal_eval(crew_str)
        return sum(1 for c in crew_list if c.get('department') == department)
    except:
        return 0

df['num_top_cast'] = df['cast'].apply(parse_cast)
df['num_directors'] = df['crew'].apply(lambda x: parse_crew(x, 'Directing'))
df['num_writers'] = df['crew'].apply(lambda x: parse_crew(x, 'Writing'))
df['num_producers'] = df['crew'].apply(lambda x: parse_crew(x, 'Production'))

# Visualization 1: Revenue Distribution
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df['revenue'].dropna()), bins='auto', kde=True)
plt.title("Distribution of Log-Transformed Revenue (0-15 Range)")
plt.xlabel("Log(1 + Revenue)")
plt.ylabel("Frequency")
plt.yscale('log')
plt.xlim(0, 15)
plt.tight_layout()
plt.savefig("figures/revenue_distribution.png")
plt.close()

# Visualization 2: Budget vs Revenue
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='budget', y='revenue', alpha=0.5)
plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("figures/budget_vs_revenue.png")
plt.close()

# Visualization 3: Average Revenue by Genre
if 'main_genre' in df.columns:
    genre_revenue = df.groupby('main_genre')['revenue'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_revenue.values, y=genre_revenue.index)
    plt.title("Average Revenue by Genre")
    plt.xlabel("Average Revenue")
    plt.ylabel("Main Genre")
    plt.tight_layout()
    plt.savefig("figures/genre_revenue.png")
    plt.close()

# Feature selection
features = ['budget', 'popularity', 'runtime', 'rating', 
            'num_top_cast', 'num_directors', 'num_writers', 'num_producers']
df['main_genre_encoded'] = df['main_genre'].astype('category').cat.codes
features.append('main_genre_encoded')
target = 'log_revenue'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize augmented features
X_train_aug = X_train.copy()
X_test_aug = X_test.copy()
y_resid = y_train.copy()

# AugBoost iterations
N_ITER = 3
for iteration in range(N_ITER):
    print(f"\n=== AugBoost Iteration {iteration + 1} ===")

    lgb_train = lgb.Dataset(X_train_aug, label=y_resid)
    lgb_model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}, 
                          lgb_train, num_boost_round=100)

    pred_train = lgb_model.predict(X_train_aug)
    y_resid = y_train - pred_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_aug)

    ann_model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    ann_model.fit(X_train_scaled, y_resid, 
                  validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    ann_train_features = ann_model.predict(X_train_scaled)
    ann_test_features = ann_model.predict(X_test_scaled)

    X_train_aug[f'ann_out_{iteration}'] = ann_train_features.flatten()
    X_test_aug[f'ann_out_{iteration}'] = ann_test_features.flatten()

# Final LGBM training
final_lgb_train = lgb.Dataset(X_train_aug, label=y_train)
final_model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1},
                        final_lgb_train, num_boost_round=100)

# Final predictions
final_preds = final_model.predict(X_test_aug)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
print(f"\n✅ Final AugBoost RMSE: {rmse:.4f}")

# Save final performance comparison plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, final_preds, alpha=0.5)
plt.xlabel("Actual Log Revenue")
plt.ylabel("Predicted Log Revenue")
plt.title("Final Model: Actual vs Predicted Log Revenue")
plt.tight_layout()
plt.savefig("figures/final_model_performance.png")
plt.close()






