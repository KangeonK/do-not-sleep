import os
import json
import cv2
import numpy as np
from scipy.spatial import distance

def get_keypoint(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    return json_data['ObjectInfo']['KeyPoints']['Points']



def calculate_EAR(eye): # 눈 거리 계산
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)

    return ear_aspect_ratio


def calculate_MAR(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[8])
    mouth_aspect_ratio = (A + B)/(2.0 * C)
    return mouth_aspect_ratio


def img_read_resize(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img


def get_eye_state(keypoint):
    
    # 눈 특징점
    leye_pos = keypoint[35:41]
    reye_pos = keypoint[41:47]
    
    #리스트에 있는 문자열들을 실수로 변환
   
    leye_EAR = calculate_EAR(leye_pos)
    reye_EAR = calculate_EAR(reye_pos)

    # 열려있으면 1, 닫혀있으면 0
    EAR = (leye_EAR+reye_EAR)/2
    EAR = round(EAR,2)
 
    # EAR 값이 작을 눈을 적게 뜬 것임
    if EAR<0.19:
        return 0
    else:
        return 1


def get_mouth_state(keypoint):

    mouth_pos = keypoint[47:59]

    MAR = calculate_MAR(mouth_pos)
        # 입이 열려있으면 1, 닫혀있으면 0
    if MAR > 0.79:
        return 1
    else:
        return 0

def current_status(eye_state, mouth_state):
    if eye_state == 1 and mouth_state == 1:
        return 'yawning'
    elif eye_state == 1 and mouth_state != 1:
        return 'drowsy'
    else:
        return 'fine'

root_json_dir = '/content/drive/MyDrive/testData'
root_image_dir = '/content/drive/MyDrive/testData'

folder_names = [name for name in os.listdir(root_json_dir) if os.path.isdir(os.path.join(root_json_dir, name))]

img = []
status = []

for folder_name in folder_names:
    image_dir = os.path.join(root_image_dir, folder_name)
    json_dir = os.path.join(root_json_dir, folder_name)

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))

            image = img_read_resize(image_path)

            #keypoint
            keypoint = get_keypoint(json_path)
            keypoint = list(map(float, keypoint))
            keypoint = [[x, y] for x, y in zip(keypoint[0 :: 2], keypoint[1 :: 2])]
            
            eye_state = get_eye_state(keypoint)
            mouth_state = get_mouth_state(keypoint)

            current_status(eye_state, mouth_state)

            img.append(image)
            status.append(current_status(eye_state, mouth_state))



print(len(img))
print(status)
