import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
X = np.load("X_data.npy")
y = np.load("y_data.npy")

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Convert labels to categorical
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save model
model.save("gesture_model.h5")

print("Model saved as gesture_model.h5")