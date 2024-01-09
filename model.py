import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Reading data from csv file
df = pd.read_csv('carData.csv')

# Considering the final dataframe by including every feature except the first one
final_df = df.iloc[:, 1:]

# Calculating years from the purchase of car and assigning it to Age feature after creation
final_df['Age'] = list(map(lambda x: 2020 - x, final_df['Year']))

# Drop the Year column
final_df.drop('Year', axis=1, inplace=True)

# One hot encoding the categorical columns using get_dummies function
final_df = pd.get_dummies(final_df, drop_first=True)

# inputs and targets df
x = final_df.iloc[:, 1:]
y = final_df.iloc[:, 0]

# Model building
#Train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

## Hyperparameters
# RandomizesdsearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create a random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_model = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=random_grid, scoring="neg_mean_squared_error",
                               n_iter=10, cv=5, random_state=42, n_jobs=1)

rf_random.fit(x_train, y_train)

#Predictions
predictions = rf_random.predict(x_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)

# Calculate R-squared score
r_squared = r2_score(y_test, predictions)

# Output the metrics
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")