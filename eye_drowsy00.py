import json
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def extract_eye_region(image_path, json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
        
    # GRAY 이미지 변환
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Reye와 Leye의 위치
    leye_pos = json_data['ObjectInfo']['BoundingBox']['Leye']['Position']
    reye_pos = json_data['ObjectInfo']['BoundingBox']['Reye']['Position']

    # 눈 영역
    leye_region = image[leye_pos[1]:leye_pos[3], leye_pos[0]:leye_pos[2]]
    reye_region = image[reye_pos[1]:reye_pos[3], reye_pos[0]:reye_pos[2]]
    
    return leye_region, reye_region, leye_pos, reye_pos


def get_eye_state(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    
    # 눈 상태 (True: 열림, False: 닫힘)
    leye_opened = json_data['ObjectInfo']['BoundingBox']['Leye']['Opened']
    reye_opened = json_data['ObjectInfo']['BoundingBox']['Reye']['Opened']

    # 두 눈 중 하나가 열려있으면 'fine', 하나라도 닫혀있으면 'drowsy'
    if leye_opened == 'true' or reye_opened == 'true' :
        return 'fine'
    else:
        return 'drowsy'

# 폴더경로: [라벨]bbox(통제환경), [원천]bbox(통제환경)
root_json_dir = 'folder path'
root_image_dir = 'folder path'

# 폴더 이름 목록 얻기
folder_names = [name for name in os.listdir(root_json_dir) if os.path.isdir(os.path.join(root_json_dir, name))]

# 각 폴더 속 데이터셋 불러오기
for folder_name in folder_names:
    image_dir = os.path.join(root_image_dir, folder_name)
    json_dir = os.path.join(root_json_dir, folder_name)

    X = []
    y = []

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))
            
            leye_img, reye_img, leye_pos, reye_pos = extract_eye_region(image_path, json_path)
            eye_state = get_eye_state(json_path)
            label = 1 if eye_state == 'fine' else 0

            if leye_img is not None and reye_img is not None:
                if 0 not in leye_pos and leye_img.size > 0:
                    leye_img = cv2.resize(leye_img, (24, 24))
                    X.append(leye_img)
                    y.append(label)
                if 0 not in reye_pos and reye_img.size > 0:
                    reye_img = cv2.resize(reye_img, (24, 24))
                    X.append(reye_img)
                    y.append(label)


    # 데이터셋 준비
    X = np.array(X).reshape(-1, 24, 24, 1)
    y = np.array(y)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # CNN
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

    #컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping과 ModelCheckpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # 모델 학습
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
    
    #모델 저장
    model.save('final_drowsy_model.h5')
