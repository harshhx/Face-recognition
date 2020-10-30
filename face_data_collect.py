import cv2
import numpy
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
dataset_path = "./data/"
face_section =[]
name = input("Enter the name of the person: ")
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            # store the 10th Face
            face_data.append(face_section)
            print(len(face_data))
            pass
    if not ret:
        continue
    cv2.imshow("Video Frame", frame)
    # cv2.imshow("face section", face_section)

    # Wait for user input - q, then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = numpy.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

numpy.save(dataset_path+name+".npy",face_data)
print("data succesfully saved at: "+dataset_path+name+".npy")

cap.release()
cv2.destroyAllWindows()
