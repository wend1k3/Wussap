import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd

# Assuming your dataset is loaded into X, y
# Placeholder for dataset loading
# X, y = load_your_data()
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

x_train = train[["AI_Interaction_Level", "Satisfaction_with_AI_Services"]]
x_test = test[["AI_Interaction_Level", "Satisfaction_with_AI_Services"]]
y_train =train["Customer_Churn"]
y_test = test["Customer_Churn"]

# Split the dataset (this part is dependent on how you load/preprocess your data)


# Data Preprocessing
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Define the model building function
def build_model(learning_rate=0.001, units=160, dropout_rate=0.4):
    model = Sequential([
        Dense(units, activation='relu', input_shape=(x_train_scaled.shape[1],)),
        Dropout(dropout_rate),
        Dense(units // 2, activation='relu'),
        Dropout(dropout_rate / 2),
        Dense(2, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the model
model = build_model()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train_scaled, y_train_cat, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[reduce_lr, early_stopping],
    verbose=2
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2%}")




