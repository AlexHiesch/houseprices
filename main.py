"""
Projekt zum Vorhersagen von Bostoner Hauspreise
"""

# Importieren unsere Bibliotheken
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Wir laden die Boston Hauspreise in ein Objekt
boston_data = load_boston()


df = pd.DataFrame(boston_data["data"], columns=boston_data["feature_names"])

target = boston_data["target"]
target = pd.Series(target)

# print(type(target))

# Datensplit in Train und Test zu 25:75%


X_train, X_test, y_train, y_test = train_test_split(df, target)

"""
print(f"Shape Training \n Features: {X_train.shape} \t Target: {y_train.shape}" )
print(f"Shape Test \n Features: {X_test.shape} \t Target: {y_test.shape}" )
"""

# Training


model = LinearRegression()
model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)
# print(y_pred)


# Evaluation


rmse = mean_squared_error(y_test, y_pred, squared=True)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")


# Model persistieren


dump(model, "ml/model")

# Daten persisiteren
df.to_csv("data/row.csv")
