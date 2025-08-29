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

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json
from ocr_util import get_ocr
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

for user_num in range(5, 10):  # This will iterate from 1 to 9
    user = f'user{user_num}'
    input_path = f'/data2/home/wuzheng/IFRAgent/Trajectories/{user}/test_dataset/data.json'
    
    with open(input_path, 'r') as f:
        data = json.load(f)

    for obs in data:
        obs['ocr'] = get_ocr(obs['image_path'], ocr_detection, ocr_recognition)

    output_path = f'/data2/home/wuzheng/IFRAgent/Trajectories/{user}/test_dataset/data_ocr.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f'Processed {user} successfully') 
