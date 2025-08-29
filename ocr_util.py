# Copyright 2025 OPPO

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import math
from PIL import Image
import os
import json
def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points

def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst


def ocr(image_path, ocr_detection, ocr_recognition):
    text_data = []
    coordinate = []
    
    image_full = cv2.imread(image_path)
    #print(image_full)
    det_result = ocr_detection(image_full)
    #print(det_result)
    det_result = det_result['polygons'] 
    #print(det_result)
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i])
        image_crop = crop_image(image_full, pts)
        
        try:
            result = ocr_recognition(image_crop)['text'][0]
        except:
            continue

        box = [int(e) for e in list(pts.reshape(-1))]
        box = [box[0], box[1], box[4], box[5]]
        
        text_data.append(result)
        coordinate.append(box)
        
    else:
        return text_data, coordinate

def get_ocr(image_path, ocr_detection, ocr_recognition):

    img = Image.open(image_path)
    width, height = img.size
    
    div_width = width / 1000
    div_height = height / 1000
    
    text_data, coordinate = ocr(image_path, ocr_detection, ocr_recognition)
    combined_data = []
    
    for i in range(len(text_data)):
        text = text_data[i]
        coords = coordinate[i]
    
        avg1 = round((coords[0] + coords[2]) / 2 / div_width, 1)
        avg2 = round((coords[1] + coords[3]) / 2 / div_height, 1)
    
        combined_data.append([text, avg1, avg2])
    
    return combined_data


# image_path = "/data1/home/wuzheng/OS-Kairos/dataset/Amap/images/1736272901.0854428_1.png"
# ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
# combined_data = get_ocr(image_path, ocr_detection, ocr_recognition)
# obs = {}
# obs['image_path'] = image_path
# obs['ocr'] = combined_data
# print(obs)

# output_dir = "/data1/home/wuzheng/OCR_test"
# os.makedirs(output_dir, exist_ok=True)
# output_filename = os.path.basename(image_path).replace('.png', '.json')
# output_path = os.path.join(output_dir, output_filename)

# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(obs, f, ensure_ascii=False, indent=4)
