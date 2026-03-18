import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# load iris dataset, 150 samples, 4 features, 3 classes (labels)
data = load_iris(as_frame=True)

# feature and label extraction
X = data.data  # type: ignore
y = data.target  # type: ignore

# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# machine learning model training
model = RandomForestClassifier()

# train model
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# save model
# joblib.dump(model, "models/iris_rf_model.pkl")
print(X.describe())
