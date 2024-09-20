import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sign_language_model.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28)) / 255.0
    reshaped = np.reshape(resized, (1, 28, 28, 1))

    # Predict gesture
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)

    letter = chr(predicted_class + 65 if predicted_class < 9 else predicted_class + 66)

    cv2.putText(frame, f'Predicted: {letter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
