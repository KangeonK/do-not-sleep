import os
import json
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 눈 감음 = 1, 눈뜸 = 0 으로 바꾸기
def label_from_json(json_data):
    leye_opened = json_data['ObjectInfo']['BoundingBox']['Leye']['Opened']
    reye_opened = json_data['ObjectInfo']['BoundingBox']['Reye']['Opened']
    leye_visible = json_data['ObjectInfo']['BoundingBox']['Leye']['isVisible']
    reye_visible = json_data['ObjectInfo']['BoundingBox']['Reye']['isVisible']
    
    if leye_visible == 'false' or reye_visible == 'false':
        return None
    else:
        return 0 if (leye_opened == 'true' and reye_opened == 'true') else 1

def resize_images(root_image_dir, root_json_dir, folder_names, new_size=(153, 96)):
    X = []  # jpg 운전자 이미지 데이터
    y = []  # json 라벨링 데이터
    total_images = 0  # 처리될 총 이미지 수

    for folder_name in folder_names:
        json_dir = os.path.join(root_json_dir, folder_name)
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                total_images += 1

    processed_images = 0  # 현재까지 처리된 이미지 수

    for folder_name in folder_names:
        image_dir = os.path.join(root_image_dir, folder_name)
        json_dir = os.path.join(root_json_dir, folder_name)

        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(json_dir, filename)
                image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))

                with open(json_path, 'r') as file:
                    json_data = json.load(file)

                label = label_from_json(json_data)
                if label is not None:
                    y.append(label)
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            img = img.resize(new_size)
                            img_array = np.array(img) / 255.0
                            X.append(img_array.astype('float32'))

                processed_images += 1
                print(f"진행 상황: {processed_images}/{total_images} 이미지 처리됨")

    return np.array(X), np.array(y)
        
# 폴더경로: [라벨]bbox(통제환경), [원천]bbox(통제환경)
root_json_dir = '/Users/kangeonkim/Downloads/do-not-sleep/졸음운전자/Training/[라벨]bbox(통제환경)'
root_image_dir = '/Users/kangeonkim/Downloads/do-not-sleep/졸음운전자/Training/[원천]bbox(통제환경)'
folder_names = [name for name in os.listdir(root_json_dir) if os.path.isdir(os.path.join(root_json_dir, name))]

X, y = resize_images(root_image_dir, root_json_dir, folder_names)
print("X 데이터 형태:", X.shape)
print("y 데이터 형태:", y.shape)
#데이터셋 분할 : 학습/테스트
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델설계 : CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 153, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()

#컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
model_checkpoint = ModelCheckpoint('best_drowsy_model.hdf5', monitor='val_loss', save_best_only=True, verbose=1)

# 모델 학습
history = model.fit(
    X_train, y_train,batch_size=16,
    epochs=10,validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint])

# 모델 저장
model.save('last_drowsy_model.hdf5')

# 테스트셋 정확도
score = model.evaluate(X_test, y_test)
print("Test accuracy : ", score[1])