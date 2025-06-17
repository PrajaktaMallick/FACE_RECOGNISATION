import os
import cv2
import numpy as np
from deepface import DeepFace

dir="Dataset"
os.makedirs(dir, exist_ok=True)

def create_dataset(name):
    person=os.path.join(dir,name)
    os.makedirs(person, exist_ok=True)
    cap=cv2.VideoCapture(0)
    c=0
    while c<30:
        ret,frame=cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for(x,y,w,h) in faces:
            c+=1
            face_img=frame[y:y+h, x:x+w]
            face_path=os.path.join(person,f"{name}_{c}.jpg")
            cv2.imwrite(face_path, face_img)
            cv2.rectangle(frame,(x, y),(x + w, y + h),(255,0,0),2)
            cv2.imshow("Capturing Faces",frame)
            if cv2.waitKey(1) & 0xFF == ord('q') and c>=30:
                break
    cap.release()
    cv2.destroyAllWindows()

def train_dataset():
    embedding={}
    s=False
    for i in os.listdir(dir):
        person_path=os.path.join(dir,i)

        if os.path.isdir(person_path):
            embedding[i] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    emd=DeepFace.represent(img_path, model_name='Facenet512', enforce_detection=False)[0]['embedding']
                    embedding[i].append(emd)
                    s=True
                except Exception as e:
                    print("fail to train data")
    print(s)
    return embedding


def recognize_face(embedding):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        try:
            analyzed_face = DeepFace.analyze(face_img, actions=['age','emotion','gender','race'],detector_backend="retinaface", enforce_detection=False)
            if isinstance(analyzed_face, list):
                analyzed_face = analyzed_face[0]

                age = analyzed_face['age']
                gender= analyzed_face['dominant_gender']
                emotion= analyzed_face['dominant_emotion']
                race=analyzed_face['dominant_race']

                face_embedding = DeepFace.represent(face_img, model_name='Facenet512',detector_backend="retinaface", enforce_detection=False)[0]['embedding']

                match=None
                max_similarity = -1
                for person, person_embeddings in embedding.items():
                    for emb in person_embeddings:
                        similarity = np.dot(face_embedding, emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(emb))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person
                if max_similarity>0.7:
                    label=f"{match} ({max_similarity:.2f})"
                else:
                    label="Unknown person"
                text=f"{label}| Age:{int(age)}| Gender:{gender}| Emotion:{emotion}| Race:{race}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print("Error in analyzing face:", e)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("1. Create Dataset")
        print("2. Train Dataset")
        print("3. Recognize Face")
        print("4. Exit")
        choice = input("Enter your choice: ")
        match choice:
            case "1":
                name = input("Enter the name of the person: ")
                create_dataset(name)
            case "2":
                embedding = train_dataset()
                np.save("embedding.npy", embedding)
                print("Dataset trained successfully.")
            case "3":
                if os.path.exists("embedding.npy"):
                    embedding = np.load("embedding.npy", allow_pickle=True).item()
                    recognize_face(embedding)
                else:
                    print("Please train the dataset first.")
            case "4":
                print("Exiting...")
                break
            case _:
                print("Invalid choice. Please try again.")