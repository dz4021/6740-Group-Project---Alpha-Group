import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Constants
LOOK_BACK = 5
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1  # 10% of TRAIN_SIZE
TEST_SIZE = 0.1   # 10% of TRAIN_SIZE

# Decrease verbosity of TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
data = pd.read_csv(r'model_input.csv')

# Function to create dataset for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Selecting features and target
features = ['Sentiment', 'Daily Average Sentiment', 'SPY']
target_col_1 = 'Next Day Ticker Price Change Direction'
target_col_5 = '5 Day Ticker Price Change Direction'

# Standardize the data
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare input data for the model
X, y_next_day = create_dataset(data[features], data[target_col_1], LOOK_BACK)
_, y_5_day = create_dataset(data[features], data[target_col_5], LOOK_BACK)

# Splitting the dataset into train, validation, and test sets
X_train_full, X_test, y_train_full_next_day, y_test_next_day = train_test_split(X, y_next_day, test_size=TEST_SIZE, random_state=0)
X_train_full, X_test, y_train_full_5_day, y_test_5_day = train_test_split(X, y_5_day, test_size=TEST_SIZE, random_state=0)

X_train, X_val, y_train_next_day, y_val_next_day = train_test_split(X_train_full, y_train_full_next_day, test_size=VALID_SIZE, random_state=0)
X_train, X_val, y_train_5_day, y_val_5_day = train_test_split(X_train_full, y_train_full_5_day, test_size=VALID_SIZE, random_state=0)

# Model Building
def build_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create models for both next day and 5 day predictions
model_next_day = build_model((LOOK_BACK, len(features)))
model_5_day = build_model((LOOK_BACK, len(features)))

# Training the models
history_next_day = model_next_day.fit(X_train, y_train_next_day, epochs=20, batch_size=32, validation_data=(X_val, y_val_next_day), verbose=1)
history_5_day = model_5_day.fit(X_train, y_train_5_day, epochs=20, batch_size=32, validation_data=(X_val, y_val_5_day), verbose=1)

# Optionally, evaluate the models
print(model_next_day.evaluate(X_test, y_test_next_day))
print(model_5_day.evaluate(X_test, y_test_5_day))