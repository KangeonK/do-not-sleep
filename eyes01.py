import json
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# JSON 파일에서 눈의 위치 정보를 추출하는 함수
def extract_eye_region(image_path, json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Reye와 Leye의 위치 정보 추출
    leye_pos = json_data['ObjectInfo']['BoundingBox']['Leye']['Position']
    reye_pos = json_data['ObjectInfo']['BoundingBox']['Reye']['Position']

    # 눈 영역 잘라내기
    leye_region = image[leye_pos[1]:leye_pos[3], leye_pos[0]:leye_pos[2]]
    reye_region = image[reye_pos[1]:reye_pos[3], reye_pos[0]:reye_pos[2]]
    
    return leye_region, reye_region

# JSON 파일에서 눈의 상태 (열림/닫힘)를 추출하는 함수
def get_eye_state(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    
    # 눈 상태 (True: 열림, False: 닫힘)
    leye_opened = json_data['ObjectInfo']['BoundingBox']['Leye']['Opened']
    reye_opened = json_data['ObjectInfo']['BoundingBox']['Reye']['Opened']

    # 두 눈이 모두 열려있으면 'fine', 하나라도 닫혀있으면 'drowsy'
    if leye_opened and reye_opened:
        return 'fine'
    else:
        return 'drowsy'

# 이미지와 JSON 파일 경로
image_dir = '/Users/kangeonkim/Downloads/do-not-sleep/졸음운전자/Training/[원천]bbox(통제환경)/001_G1'
json_dir = '/Users/kangeonkim/Downloads/do-not-sleep/졸음운전자/Training/[라벨]bbox(통제환경)/001_G1'

X = []
y = []

# 데이터 준비
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(json_dir, filename)
        image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))
        
        leye_img, reye_img = extract_eye_region(image_path, json_path)

        if leye_img is not None and reye_img is not None and leye_img.size != 0 and reye_img.size != 0:
            leye_img = cv2.resize(leye_img, (24, 24))
            reye_img = cv2.resize(reye_img, (24, 24))

            # 라벨 추출
            eye_state = get_eye_state(json_path)
            label = 1 if eye_state == 'fine' else 0

            X.append(leye_img)
            X.append(reye_img)
            y.append(label)
            y.append(label)
        else:
            print(f"Error in image: {filename}")

# 리스트를 NumPy 배열로 변환
X = np.array(X).reshape(-1, 24, 24, 1)
y = np.array(y)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CNN 모델 설계 및 컴파일
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# 모델 저장
model.save('drowsy_detection_model.h5')
