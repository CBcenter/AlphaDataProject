import cv2
import sqlite3
from deepface import DeepFace
import hashlib
from datetime import datetime
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchsummary import summary
from torchvision.models import resnet18
from facenet_pytorch import MTCNN
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleModel, self).__init__()
        resnet = resnet18(pretrained=False)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

db_path = '/home/AlphadataUbuntuprod/ADProjectprod/face_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create face_data table if not exists
cursor.execute('DROP TABLE IF EXISTS face_data')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_data (
        id INTEGER PRIMARY KEY,
        customer_id TEXT,
        age TEXT,
        gender TEXT,
        emotion TEXT,
        ethnicity TEXT,
        landmarks TEXT,
        timestamp TEXT,
        time_detected INTEGER  -- Added time_detected column
    )
''')
conn.commit()

# Create Likes table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Likes (
        Customer_id TEXT PRIMARY KEY,
        age_avg TEXT,
        gender_avg TEXT,
        emotion_avg TEXT,
        ethnicity_avg TEXT,
        time_spent_ttl TEXT,
        total_time_person_detected INTEGER  -- New column for total time person detected
    )
''')
conn.commit()

loaded_model = SimpleModel()

checkpoint = torch.load('/home/AlphadataUbuntuprod/ADProjectprod/EthnicityRecognition-UTKFaces/lightning_logs/version_2/checkpoints/epoch=29-step=8670.ckpt', map_location=torch.device('cpu'))

model_dict = loaded_model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
loaded_model.load_state_dict(pretrained_dict)

loaded_model = torch.quantization.quantize_dynamic(
    loaded_model, {torch.nn.Linear}, dtype=torch.qint8
)

print("Keys in the loaded checkpoint:")
for key in loaded_model.state_dict():
    print(key)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

mtcnn = MTCNN(keep_all=True)

video_path = '/home/AlphadataUbuntuprod/ADProjectprod/TestVid_Folder/project_video_tempv4.mp4'
#ideo_path = 'rtsp://92.168.3.17/live3'
#video_path = 'https://youtu.be/fup-jRZKd1M?si=zWEskfpBZFQuV9NS'
#bbb
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()
    
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)

customer_data = {}  # Dictionary to store data for each customer ID
customer_embeddings = {}  # Dictionary to store face embeddings with customer ID
time_delay = 5

start_time = time.time()

frame_count = 0
frame_interval = 5  

def generate_customer_id():
    unique_string = f"{time.time()}"
    customer_id = hashlib.md5(unique_string.encode()).hexdigest()
    return customer_id

def collect_and_insert_data():
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read frame.")
        return None

    resized_frame = cv2.resize(frame, (800, 600))

    faces, _ = mtcnn.detect(resized_frame)

    face_info_list = []
    if faces is not None:
        for face in faces:
            x, y, w, h = face

            face_roi = frame[int(y):int(y+h), int(x):int(x+w)]

            try:
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

                frame_roi_resized = cv2.resize(face_roi, (800, 600))

                result = {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'frame_roi_resized': frame_roi_resized,
                }

                face_info_list.append(result)

            except (ValueError, TypeError, KeyError) as e:
                print(f"Error processing frame: {e}")

    print(f"{len(face_info_list)} faces detected in this frame.")
    return frame, face_info_list

def face_embedding(face_roi_resized):
    # Preprocess the face image and get the embeddings
    pil_image = Image.fromarray(face_roi_resized)
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        embeddings = loaded_model(input_batch)

    return embeddings[0].numpy()

try:
    while True:
        frame_count += 1
        if frame_count >= 10:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

            print(f"FPS: {fps}")

        ret, frame = video_capture.read()
        if frame_count % frame_interval != 0:
            continue

        frame, face_info_list = collect_and_insert_data()

        if face_info_list:
            for face_info in face_info_list:
                x = face_info['x']
                y = face_info['y']
                w = face_info['w']
                h = face_info['h']
                frame_roi_resized = face_info['frame_roi_resized']

                if frame_roi_resized is not None:
                    try:
                        result = DeepFace.analyze(frame_roi_resized, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

                        if isinstance(result, list) and len(result) > 0:
                            result = result[0]
                            age = result['age']
                            gender = max(result['gender'], key=result['gender'].get)
                            emotion = result['dominant_emotion']
                            ethnicity = result['dominant_race']  # Extract ethnicity information
                            landmarks = result.get('landmarks', [])

                            # Get the face embedding
                            embedding = face_embedding(frame_roi_resized)

                            # Check if there is a customer with a similar embedding
                            customer_id = None
                            for cid, stored_embedding in customer_embeddings.items():
                                similarity = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
                                if similarity > 0.8:  # You may adjust this threshold
                                    customer_id = cid
                                    break

                            # If no similar customer found, create a new customer
                            if customer_id is None:
                                customer_id = generate_customer_id()
                                customer_embeddings[customer_id] = embedding

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Update customer data
                            if customer_id not in customer_data:
                                customer_data[customer_id] = {
                                    'ages': [],
                                    'genders': [],
                                    'emotions': [],
                                    'ethnicities': [],
                                    'timestamps': [],
                                    'total_time_detected': 0  # New variable to store total time person detected
                                }

                            customer_data[customer_id]['ages'].append(float(age))
                            customer_data[customer_id]['genders'].append(gender)
                            customer_data[customer_id]['emotions'].append(emotion)
                            customer_data[customer_id]['ethnicities'].append(ethnicity)
                            customer_data[customer_id]['timestamps'].append(timestamp)

                            # Calculate time spent
                            time_spent = time.time() - start_time

                            # Accumulate total time person detected
                            customer_data[customer_id]['total_time_detected'] += time_spent

                            # Display information
                            text_to_display = [
                                f"ID: {customer_id}",
                                f"Age: {age}",
                                f"Gender: {gender}",
                                f"Emotion: {emotion}",
                                f"Ethnicity: {ethnicity}",
                                f"Time Spent: {time_spent:.2f} seconds"  # Display time spent
                            ]

                            for i, text in enumerate(text_to_display):
                                cv2.putText(frame, text, (int(x) + int(w) + 10, int(y) + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                            # Insert data into face_data table
                            cursor.execute('''
                                INSERT INTO face_data (customer_id, age, gender, emotion, ethnicity, landmarks, timestamp, time_detected)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (customer_id, age, gender, emotion, ethnicity, str(landmarks), timestamp, int(time_spent)))
                            conn.commit()

                            # Update Likes table in real-time
                            avg_age = np.mean(customer_data[customer_id]['ages'])
                            avg_gender = max(set(customer_data[customer_id]['genders']), key=customer_data[customer_id]['genders'].count)
                            avg_emotion = max(set(customer_data[customer_id]['emotions']), key=customer_data[customer_id]['emotions'].count)
                            avg_ethnicity = max(set(customer_data[customer_id]['ethnicities']), key=customer_data[customer_id]['ethnicities'].count)
                            time_spent_ttl = customer_data[customer_id]['total_time_detected']  # Update with total time person detected

                            # Check if the customer exists in Likes table
                            cursor.execute('SELECT * FROM Likes WHERE Customer_id=?', (customer_id,))
                            existing_customer = cursor.fetchone()

                            if existing_customer:
                                # If the customer exists, update the existing row
                                cursor.execute('''
                                    UPDATE Likes
                                    SET age_avg=?, gender_avg=?, emotion_avg=?, ethnicity_avg=?, time_spent_ttl=?, total_time_person_detected=?
                                    WHERE Customer_id=?
                                ''', (str(avg_age), avg_gender, avg_emotion, avg_ethnicity, str(time_spent_ttl), time_spent_ttl, customer_id))
                            else:
                                # If the customer does not exist, insert a new row
                                cursor.execute('''
                                    INSERT INTO Likes (Customer_id, age_avg, gender_avg, emotion_avg, ethnicity_avg, time_spent_ttl, total_time_person_detected)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (customer_id, str(avg_age), avg_gender, avg_emotion, avg_ethnicity, str(time_spent_ttl), time_spent_ttl, customer_id))

                            conn.commit()

                    except Exception as e:
                        print(f"Error analyzing face: {e}")

            cv2.imshow('Video', frame)

        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    conn.close()

# Fetch data from Likes table and print
cursor.execute('SELECT * FROM Likes')
likes_data = cursor.fetchall()

print("Likes Data:")
for row in likes_data:
    customer_id = row[0]
    age_avg = row[1]
    gender_avg = row[2]
    emotion_avg = row[3]
    ethnicity_avg = row[4]
    time_spent_ttl = row[5]
    total_time_person_detected = row[6]

    print(f"Customer ID: {customer_id}")
    print(f"Age Avg: {age_avg}")
    print(f"Gender Avg: {gender_avg}")
    print(f"Emotion Avg: {emotion_avg}")
    print(f"Ethnicity Avg: {ethnicity_avg}")
    print(f"Time Spent Total: {time_spent_ttl} seconds")
    print(f"Total Time Person Detected: {total_time_person_detected} seconds")
    print("--------------")

summary(loaded_model, (3, 224, 224))
