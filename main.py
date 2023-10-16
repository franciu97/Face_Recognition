import cv2
import face_recognition
import csv
from tensorflow.keras.models import load_model
import numpy as np

#===Carica CNN===
model = load_model('mio_modello_regularization.h5')

#===Carica l'immagine di riferimento e calcola encoding===
reference_image_fra = face_recognition.load_image_file\
    ('insert/your/path/to/subject/folder')
reference_encoding_fra = face_recognition.face_encodings(reference_image_fra)[0]

#===Webcam===
cap = cv2.VideoCapture(0)

#===Creazione lista per encoding===
def predict_face(frame, face_location):
    top, right, bottom, left = face_location
    face = frame[top:bottom, left:right]
    face = cv2.resize(face, (64, 64))
    face = np.expand_dims(face, axis=0)
    prediction = model.predict(face)
    return prediction

face_encodings_list = []

#===Ciclo cattura volti e identificazione===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_encodings_list.append(list(face_encoding))

        #===Metodo di face encoding===
        matches_fra = face_recognition.compare_faces([reference_encoding_fra], face_encoding, tolerance=0.5)

        #===CNN===
        prediction = predict_face(frame, (top, right, bottom, left))

        name = 'Unknown'
        color = (0, 0, 255)

        #===Combinazione dei metodi===
        if True in matches_fra and prediction >= 0.5:
            name = 'Subject'
            color = (0, 255, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#===Salvataggio encoding in CSV===
with open('face_encodings.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Encoding'])
    for encoding in face_encodings_list:
        writer.writerow(encoding)