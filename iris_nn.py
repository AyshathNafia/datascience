# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 🔹 Step 2: Load Dataset
# Make sure iris.csv is in the same folder
data = pd.read_csv("iris.csv")

# 🔹 Step 3: Explore Dataset
print("First 10 rows:\n", data.head(10))
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:\n", data.describe())

# 🔹 Step 4: Data Cleaning
print("\nMissing values:\n", data.isnull().sum())

# If any missing values exist, remove them
data = data.dropna()

# 🔹 Step 5: Encode Categorical Data (species → numeric)
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

# 🔹 Step 6: Split Features and Target
X = data.drop('species', axis=1)
y = data['species']

# 🔹 Step 7: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 8: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Step 9: Build Neural Network
model = Sequential()

# Input + Hidden Layers
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))

# Output Layer (3 classes → softmax)
model.add(Dense(3, activation='softmax'))

# 🔹 Step 10: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🔹 Step 11: Train Model
print("\nTraining Model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=5)

# 🔹 Step 12: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# 🔹 Step 13: Predictions
predictions = model.predict(X_test)

# Convert probabilities → class labels
predicted_classes = np.argmax(predictions, axis=1)

print("\nPredicted Classes:", predicted_classes[:10])
print("Actual Classes   :", y_test.values[:10])
