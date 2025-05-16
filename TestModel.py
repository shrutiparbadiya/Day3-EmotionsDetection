import cv2
import pickle
from Utils import get_face_landmarks
import numpy as np

emotions = ['HAPPY', 'SAD', 'SURPRISED']

with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, static_image_mode=True)
    # face_landmarks = np.array(face_landmarks)
    output = model.predict([face_landmarks])

    cv2.putText(frame, emotions[int(output[0])],
                (10, frame.shape[0] - 1),
                cv2.FONT_HERSHEY_PLAIN,
                3,(0,255,0),5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# img = cv2.imread(r"C:\Users\dell\PycharmProjects\Day3-EmotionDetection\Day3-Resources\emotion\test\surprised\im16.png")
# face_landmarks = get_face_landmarks(img, static_image_mode=False)
# face_landmarks = np.array(face_landmarks)
# output = model.predict(face_landmarks.reshape(1, -1))
# print(output)