import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    labels = data['label'].values 
    images = data.drop('label', axis=1).values  
    
    # Reshape images to 28x28
    images = images.reshape(-1, 28, 28, 1)  
    images = images / 255.0  
    

    labels = to_categorical(labels, num_classes=25)  
    
    return images, labels


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(25, activation='softmax')  # 25 output classes (A-Y, excluding J)
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

X_train, y_train = load_data('data/sign_mnist_train.csv')
X_test, y_test = load_data('data/sign_mnist_test.csv')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = create_model()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

model.save('models/sign_language_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
