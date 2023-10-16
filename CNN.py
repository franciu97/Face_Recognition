import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

#===Initialize webcam and variables===
webcam = cv2.VideoCapture(0)
face_data = []
labels = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#===Ciclo principale per la cattura dei volti===
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_data.append(face_resized)
        labels.append(1)  # Label 1 per user
    cv2.imshow('Face Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()

#===Caricamento dei volti sconosciuti===
unknown_faces_path = 'insert/your/path/to/unknown/folder'
for filename in os.listdir(unknown_faces_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(unknown_faces_path, filename)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (64, 64))
        face_data.append(img_resized)
        labels.append(0)  # Label 0 per sconosciuti

#===Preparazione dei dati===
X = np.array(face_data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#===Inizializzazione e compilazione del modello CNN===
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Dropout layer

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Dropout layer

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Dropout layer

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

#===Data Augmentation===
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

#===Addestramento del modello===
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=10,
                    validation_data=(X_test, y_test))

#===Display metrics===
print('Final training loss:', history.history['loss'][-1])
print('Final training accuracy:', history.history['accuracy'][-1])
print('Final training precision:', history.history['precision'][-1])
print('Final training recall:', history.history['recall'][-1])
print('Final validation loss:', history.history['val_loss'][-1])
print('Final validation accuracy:', history.history['val_accuracy'][-1])
print('Final validation precision:', history.history['val_precision'][-1])
print('Final validation recall:', history.history['val_recall'][-1])

#===Salvataggio modello===
model.save('mio_modello_regularization.h5')