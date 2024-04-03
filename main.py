import pandas as pd
import numpy as np
import cv2
import pickle
import mediapipe as mp
from tensorflow.keras.models import load_model
from datetime import datetime
import easygui
import face_recognition
import os
import sys
import shutil
import matplotlib.pyplot as plt



# Dictionary mapping emotion labels to human-readable strings
emotion_dict = {0:'bored', 1:'confused', 2: 'drowsy',3:'frustrated', 4:'interested',5: 'Looking Away'}



#loading sequential model
model1 = load_model('C:\\Users\\nanda\\Downloads\\model1.h5')
#loading random forest model
with open('student_body_language.pkl', 'rb') as f:
    model2 = pickle.load(f)



# Dictionary to store known person's face encodings
known_persons = {}
# Dictionary to store each person's emotions
person_emotions = {}
# Output directory for storing face images
output_directory = 'output_heads'
os.makedirs(output_directory, exist_ok=True)



# Function to predict expressions using model1
def prediction1(image):
    image_arr = []
    pic = cv2.resize(image, (48, 48))
    image_arr.append(pic)
    image_arr = np.array(image_arr)
    image_arr = image_arr.astype('float32')
    image_arr /= 255
    prediction = model1.predict(image_arr)
    print(prediction[0])
    return prediction[0]



mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
# Function to predict expression using model2
def prediction2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            results = holistic.process(image)
            pose = results.pose_landmarks.landmark
            pose_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row
            x = pd.DataFrame([row])
            prediction_probability=model2.predict_proba(x)[0]
            print(prediction_probability)
            return prediction_probability
        except:
            print("[0 0 0 0 0 0]")
            return np.array([0,0,0,0,0,0])



# Drawing identified face and pose lines on image
def drawings(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            results = holistic.process(image)
            #1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image



#To display visualized output
def create_output_tab(person_emotions):

    for person,emotions in person_emotions.items():

        person_folder = os.path.join(output_directory, f'person_{person}')
        folder_files = os.listdir(person_folder)
        face_image_path = os.path.join(person_folder, folder_files[0])
        img = cv2.imread(face_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img.shape[:2]
        left = 0
        top = 0
        bottom = height - 1
        right = width - 1
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        # Generate visualizations for each detected face
        plt.figure(figsize=(10, 5))

        # Load the detected face image
        face_image = img[top:bottom, left:right]

        # Plot the face image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Person {person} - Face')

        #plot the analysed image of person
        plt.subplot(1, 3, 2)
        analysed_image=drawings(img)
        plt.imshow(cv2.cvtColor(analysed_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Analysis of Person {person} - Face')

        # Pie Chart
        plt.subplot(1, 3, 3)
        labels, counts = zip(*[(emotion, person_emotions[person].count(emotion)) for emotion in set(person_emotions[person])])
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f'Person {person} - Dominant Emotion Pie Chart')

        plt.tight_layout()
        plt.show()



# Parse user choice from command line arguments
choice = int(sys.argv[1])
if choice == 1:
    # Camera feed
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("--------------------------------------------------------------------------------")
    print("The camera feed analysis is chosen :")
    print("The system camera is being opened")
    print("--------------------------------------------------------------------------------")

elif choice == 2:
    # Video file
    path = easygui.fileopenbox(default='*')
    cap = cv2.VideoCapture(path)
    print("--------------------------------------------------------------------------------")
    print("The video at the path ", path, " is chosen :")
    print("--------------------------------------------------------------------------------")

elif choice == 3:
    # Image file
    image_path = easygui.fileopenbox(default='*')
    img = cv2.imread(image_path)
    time_rec = datetime.now()
    print("--------------------------------------------------------------------------------")
    print("The image at the path ", image_path, " is chosen :")
    print("--------------------------------------------------------------------------------")


    face_locations = face_recognition.face_locations(img)

    if(len(face_locations)<=1):

        emotions1 = np.array([0, 0, 0, 0, 0, 0])
        emotions2 = np.array([0, 0, 0, 0, 0, 0])

        print("------------new person is detected----------")

        emotions1 = prediction1(img)
        emotions2 = prediction2(img)

        total_emotions = emotions1 + emotions2
        emotion = emotion_dict[np.argmax(total_emotions)]

        person=1
        person_emotions[person] = person_emotions.get(person, []) + [emotion]
        person_folder = os.path.join(output_directory, f'person_{person}')
        os.makedirs(person_folder, exist_ok=True)
        cv2.imwrite(os.path.join(person_folder, f'frame_head_{person}_{len(person_emotions[person])}.jpg'), img)

    else:

        for face_location in face_locations:

            top, right, bottom, left = face_location
            roi = img[top:bottom, left:right]

            face_encoding = face_recognition.face_encodings(img, [face_location])[0]
            found_match = False

            for person, encodings in known_persons.items():
                if any(face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)):
                    found_match = True
                    break

            if not found_match:
                print("------------new person is detected----------")
                person = len(known_persons) + 1
                known_persons[person] = [face_encoding]

            emotions1=np.array([0,0,0,0,0,0])
            emotions2=np.array([0,0,0,0,0,0])

            emotions1 = prediction1(roi)
            emotions2 = prediction2(roi)

            total_emotions = emotions1 + emotions2
            emotion = emotion_dict[np.argmax(total_emotions)]

            person_emotions[person] = person_emotions.get(person, []) + [emotion]
            person_folder = os.path.join(output_directory, f'person_{person}')
            os.makedirs(person_folder, exist_ok=True)
            cv2.imwrite(os.path.join(person_folder, f'frame_head_{person}_{len(person_emotions[person])}.jpg'), roi)

    create_output_tab(person_emotions)



# Continue for camera feed or video file
if choice <= 2:
    frame_counter = 0
    while cap.isOpened():
        time_rec = datetime.now()
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            FPS = int(cap.get(cv2.CAP_PROP_FPS))
            img = cv2.flip(frame, 1)

            if choice == 1 or (choice == 2 and frame_counter % FPS == 0):
                face_locations = face_recognition.face_locations(img)

                for face_location in face_locations:

                    top, right, bottom, left = face_location
                    roi = img[top:bottom, left:right]

                    face_encoding = face_recognition.face_encodings(img, [face_location])[0]
                    found_match = False

                    for person, encodings in known_persons.items():
                        if any(face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)):
                            found_match = True
                            break

                    if not found_match:
                        print("----------new person is detected----------")
                        person = len(known_persons) + 1
                        known_persons[person] = [face_encoding]

                    emotions1 = np.array([0, 0, 0, 0, 0, 0])
                    emotions2 = np.array([0, 0, 0, 0, 0, 0])

                    emotions1 = prediction1(roi)
                    emotions2 = prediction2(roi)

                    total_emotions = emotions1 + emotions2
                    emotion = emotion_dict[np.argmax(total_emotions)]


                    person_emotions[person] = person_emotions.get(person, []) + [emotion]
                    person_folder = os.path.join(output_directory, f'person_{person}')
                    os.makedirs(person_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(person_folder, f'frame_head_{person}_{len(person_emotions[person])}.jpg'), roi)


            cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
            if(choice==1):
                analysed_image=drawings(img)
                cv2.imshow('Video', analysed_image)
            else:
                cv2.imshow('Video', img)
            cv2.resizeWindow('Video', 1000, 600)

            if cv2.waitKey(1) == 27:  # press ESC to break
                cap.release()
                cv2.destroyAllWindows()
                break

        else:
            break
    create_output_tab(person_emotions)



#Remove the output directory and its contents
shutil.rmtree(output_directory)
