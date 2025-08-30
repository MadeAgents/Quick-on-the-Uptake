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

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from rewriting import rewriting_bygpt, rewriting_byqwen
from RAG.RAGToolbox import Jinaembedding, Vectordatabase
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from zhipuai import ZhipuAI
import base64

model = "glm" #gpt or glm or qwen

def run_all_tests(test_func, base_data_path, log_dir, **kwargs):
    user_groups = {
        "user1-5": ["user1", "user2", "user3", "user4", "user5"],
        "user6-9": ["user6", "user7", "user8", "user9"]
    }
    
    group_results = {}

    for group_name, users in user_groups.items():
        group_metrics = {
            'sr_total': 0, 'type_total': 0, 'iar_total': 0, 'count': 0,
            'user_details': []
        }

        for user in users:
            data_path = f"{base_data_path}/{user}/test_dataset/data_ocr_rewritten.json"
            print(f"Testing {test_func.__name__} - user {user}...")
            
            result = test_func(
                data_path=data_path,
                log_dir=log_dir,
                user_name=user,
                **kwargs
            )
            
            group_metrics['user_details'].append({
                'user': user,
                'sr_ratio': result['sr_ratio'],
                'type_ratio': result['type_ratio'],
                'iar_ratio': result['iar_ratio']
            })
            
            group_metrics['sr_total'] += result['sr_ratio']
            group_metrics['type_total'] += result['type_ratio']
            group_metrics['iar_total'] += result['iar_ratio']
            group_metrics['count'] += 1

        if group_metrics['count'] > 0:
            group_results[group_name] = {
                'sr_avg': group_metrics['sr_total'] / group_metrics['count'],
                'type_avg': group_metrics['type_total'] / group_metrics['count'],
                'iar_avg': group_metrics['iar_total'] / group_metrics['count'],
                'details': group_metrics['user_details']
            }

    summary_path = os.path.join(log_dir, f"{test_func.__name__}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(group_results, f, indent=2, ensure_ascii=False)
    
    print(f"{test_func.__name__} is finished！Saving to {summary_path}")
    return group_results

def test_loop_IFRAgent(data_path, log_dir, user_name=""):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"IFRAgent_test_log_{user_name}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    sr_total = 0
    type_total = 0
    iar_total = 0
    total_obs = 0
    log_content = []
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        total_obs = len(data)
        log_content.append(f"Loaded {total_obs} observations from {data_path}\n")
        for idx, obs in enumerate(data, 1):
            try:
                action = get_action_IFRAgent(obs)
                sr, type_, iar = eval_action(action, obs)
                print(sr)
                sr_total += sr
                type_total += type_
                iar_total += iar  
                log_entry = (
                    f"Observation {idx}/{total_obs}:\n"
                    f"Action: {action}\n"
                    f"Results - SR: {sr}, Type: {type_}, IAR: {iar}\n"
                    "----------------------------------------\n"
                )
                print(log_entry)
                log_content.append(log_entry)
                
            except Exception as e:
                error_msg = f"Error processing observation {idx}: {str(e)}\n"
                log_content.append(error_msg)
                continue
        
        sr_ratio = sr_total / total_obs if total_obs > 0 else 0
        type_ratio = type_total / total_obs if total_obs > 0 else 0
        iar_ratio = iar_total / total_obs if total_obs > 0 else 0
        
        final_result = (
            "\n============== FINAL RESULTS ==============\n"
            f"Total observations: {total_obs}\n"
            f"Success Rate (SR): {sr_ratio:.4f} ({sr_total}/{total_obs})\n"
            f"Type Accuracy: {type_ratio:.4f} ({type_total}/{total_obs})\n"
            f"IAR Accuracy: {iar_ratio:.4f} ({iar_total}/{total_obs})\n"
            "==========================================\n"
        )
        log_content.append(final_result)
        
        with open(log_path, 'w') as log_file:
            log_file.writelines(log_content)
        
        print(f"Test completed. Log saved to {log_path}")
        
        return {
            'sr_ratio': sr_ratio,
            'type_ratio': type_ratio,
            'iar_ratio': iar_ratio,
            'log_path': log_path
        }
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}\n"
        log_content.append(error_msg)
        with open(log_path, 'w') as log_file:
            log_file.writelines(log_content)
        print(f"Error occurred. Partial log saved to {log_path}")
        raise

def test_loop_API(data_path, log_dir="logs", user_name=""):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"IFRAgent_test_log_{user_name}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    sr_total = 0
    type_total = 0
    iar_total = 0
    total_obs = 0
    log_content = []
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        total_obs = len(data)
        log_content.append(f"Loaded {total_obs} observations from {data_path}\n")
        for idx, obs in enumerate(data, 1):
            try:
                action = get_action_API(obs)
                sr, type_, iar = eval_action(action, obs)
                sr_total += sr
                type_total += type_
                iar_total += iar  
                log_entry = (
                    f"Observation {idx}/{total_obs}:\n"
                    f"Action: {action}\n"
                    f"Results - SR: {sr}, Type: {type_}, IAR: {iar}\n"
                    "----------------------------------------\n"
                )
                print(log_entry)
                log_content.append(log_entry)
                
            except Exception as e:
                error_msg = f"Error processing observation {idx}: {str(e)}\n"
                log_content.append(error_msg)
                continue
        
        sr_ratio = sr_total / total_obs if total_obs > 0 else 0
        type_ratio = type_total / total_obs if total_obs > 0 else 0
        iar_ratio = iar_total / total_obs if total_obs > 0 else 0
        
        final_result = (
            "\n============== FINAL RESULTS ==============\n"
            f"Total observations: {total_obs}\n"
            f"Success Rate (SR): {sr_ratio:.4f} ({sr_total}/{total_obs})\n"
            f"Type Accuracy: {type_ratio:.4f} ({type_total}/{total_obs})\n"
            f"IAR Accuracy: {iar_ratio:.4f} ({iar_total}/{total_obs})\n"
            "==========================================\n"
        )
        log_content.append(final_result)
        
        with open(log_path, 'w') as log_file:
            log_file.writelines(log_content)
        
        print(f"Test completed. Log saved to {log_path}")
        
        return {
            'sr_ratio': sr_ratio,
            'type_ratio': type_ratio,
            'iar_ratio': iar_ratio,
            'log_path': log_path
        }
    
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}\n"
        log_content.append(error_msg)
        with open(log_path, 'w') as log_file:
            log_file.writelines(log_content)
        print(f"Error occurred. Partial log saved to {log_path}")
        raise



def eval_action(action, obs):
    _, iar = eval_action_single(action, obs['action'])
    type_ = 0
    sr = 0 
    for act in obs['action_list']:
        single_type, single_sr = eval_action_single(action, act)
        if single_type == 1:
            type_ = 1  
        if single_sr == 1:
            sr = 1   
    return sr, type_, iar


def eval_action_single(action, label):
    if (label == "PRESS_BACK") or (label == "PRESS_HOME") or (label == "COMPLETE") or (label == "WAIT"):
        if label == action:
            return 1,1
    elif label.startswith("SCROLL"):
        if not action.startswith("SCROLL"):
            return 0,0
        elif action!=label:
            return 1,0
        else: return 1, 1
    elif label.startswith("CLICK"):
        if not action.startswith("CLICK"):
            return 0, 0
        else:
            label_pattern = r'CLICK <point>\[\[(\d+),\s*(\d+)\]\]</point>'
            action_pattern = r'CLICK <point>\[\[(\d+),\s*(\d+)\]\]</point>'           
            label_match = re.match(label_pattern, label)
            action_match = re.match(action_pattern, action)
            if not label_match or not action_match:
                return 1, 0
            x1, y1 = map(int, label_match.groups())
            x2, y2 = map(int, action_match.groups())
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            relative_distance = distance / 1000
            if relative_distance < 0.14:
                return 1, 1
            else:
                return 1, 0            
    elif label.startswith("LONG_PRESS"):
        if not action.startswith("LONG_PRESS"):
            return 0, 0
        else:
            label_pattern = r'LONG_PRESS <point>\[\[(\d+),\s*(\d+)\]\]</point>'
            action_pattern = r'LONG_PRESS <point>\[\[(\d+),\s*(\d+)\]\]</point>'
            label_match = re.match(label_pattern, label)
            action_match = re.match(action_pattern, action)
            if not label_match or not action_match:
                return 1, 0
            x1, y1 = map(int, label_match.groups())
            x2, y2 = map(int, action_match.groups())
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            relative_distance = distance / 1000
            if relative_distance < 0.14:
                return 1, 1
            else:
                return 1, 0
    elif label.startswith("TYPE"):
        if not action.startswith("TYPE"):
            return 0,0
        elif text_match(action, label) == 1:
            return 1,1
        else: return 1,0
    return 0,0

def text_match(action, label):
    similarity = SequenceMatcher(None, action, label).ratio()
    return 1 if similarity >= 0.8 else 0

def get_action_API(obs):
    prompt = (
    "### Background ###\n"
    "You are an expert in completing tasks based on screenshots and instructions. "
    "I will provide you with a mobile screenshot and a final goal. "
    "Based on the mobile screenshot and the final goal. I need you to determine the action to take. "
    f"Final Goal: {obs['task']}\n"
    "### Screenshot information ###\n"
    "To help you understand the information in the screenshot, I first performed OCR. Here are the names and coordinates of the icons obtained through OCR:"
    f"Coordinates of the icons: {obs['ocr']}"
    "### Response requirements ###\n"
    "Your skill set includes both basic and custom actions:\n"
    "1. Basic Actions\n"
    "Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability.\n"
    "Basic Action 1: CLICK\n"
    "    - Purpose: Click at the specified position.\n"
    "    - Format: CLICK <point>[[x-axis,y-axis]]</point>\n"
    "    - Example Usage: CLICK <point>[[101,872]]</point>\n"
    "    - Tips:The x-coordinate represents the thousandth part of the screen's width, counted from left to right.The y-coordinate represents the thousandth part of the screen's height, counted from top to bottom.Obviously, the range of both x and y is [0, 1000].\n\n"
    "Basic Action 2: TYPE\n"
    "    - Purpose: Enter specified text at the designated location.\n"
    "    - Format: TYPE [input text]\n"
    "    - Example Usage: TYPE [Shanghai shopping mall]\n\n"
    "Basic Action 3: SCROLL\n"
    "    - Purpose: SCROLL in the specified direction.\n"
    "    - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n"
    "    - Example Usage: SCROLL [UP]\n\n"
    "2. Custom Actions\n"
    "Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n"
    "Custom Action 1: LONG_PRESS\n"
    "    - Purpose: Long press at the specified position.\n"
    "    - Format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n"
    "    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\n"
    "Custom Action 2: PRESS_BACK\n"
    "    - Purpose: Press a back button to navigate to the previous screen.\n"
    "    - Format: PRESS_BACK\n"
    "    - Example Usage: PRESS_BACK\n\n"
    "Custom Action 3: PRESS_HOME\n"
    "    - Purpose: Press a home button to navigate to the home page.\n"
    "    - Format: PRESS_HOME\n"
    "    - Example Usage: PRESS_HOME\n\n"
    "Custom Action 4: WAIT\n"
    "    - purpose: Wait for the screen to load.\n"
    "    - format: WAIT\n"
    "    - example usage: WAIT\n\n"
    "Custom Action 5: COMPLETE\n"
    "    - Purpose: Indicate the task is finished.\n"
    "    - Format: COMPLETE\n"
    "    - Example Usage: COMPLETE\n\n"
    "### Output format ###\n"
    "Your response must exactly follow the template:\n"
    "{action: ACTION_NAME}\n"
    "Replace `ACTION_NAME` with one of:\n"
    "- CLICK <point>[[x,y]]</point>\n"
    "- TYPE [input text]\n"
    "- SCROLL [UP/DOWN/LEFT/RIGHT]\n"
    "- LONG_PRESS <point>[[x,y]]</point>\n"
    "- PRESS_BACK\n"
    "- PRESS_HOME\n"
    "- WIAT\n"
    "- COMPLETE"
    )
    if model == "gpt":
        client = OpenAI(
            base_url="",
            api_key= ""
        )
        with open(obs['image_path'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(       
            model="gpt-4o",
            messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content

    elif model == "glm":
        client = ZhipuAI(
            api_key= ""
        )
        with open(obs['image_path'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":base64_image
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
        

    elif model == "qwen":
        client = OpenAI(
            base_url="",
            api_key= ""
        )
        with open(obs['image_path'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(       
            model="qwen-vl-max",
            messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
    
    print(answer)
    action_pattern = r"{action: (.+?)}"
    match = re.search(action_pattern, answer)
    if match:
        action = match.group(1)
        print(action)
    return action


def get_action_IFRAgent(obs):
    prompt = (
    "### Background ###\n"
    "You are an expert in completing tasks based on screenshots and instructions. "
    "I will provide you with a mobile screenshot, a final goal and the step list. "
    "Based on the mobile screenshot, the final goal and the step list. I need you to determine the action to take. "
    f"Final Goal: {obs['query_rewritten']}\n"
    f"step list: {obs['step_list_rewritten']}\n"
    "### Screenshot information ###\n"
    "To help you understand the information in the screenshot, I first performed OCR. Here are the names and coordinates of the icons obtained through OCR:"
    f"Coordinates of the icons: {obs['ocr']}"
    "### Response requirements ###\n"
    "Your skill set includes both basic and custom actions:\n"
    "1. Basic Actions\n"
    "Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability.\n"
    "Basic Action 1: CLICK\n"
    "    - Purpose: Click at the specified position.\n"
    "    - Format: CLICK <point>[[x-axis,y-axis]]</point>\n"
    "    - Example Usage: CLICK <point>[[101,872]]</point>\n"
    "    - Tips:The x-coordinate represents the thousandth part of the screen's width, counted from left to right.The y-coordinate represents the thousandth part of the screen's height, counted from top to bottom.Obviously, the range of both x and y is [0, 1000].\n\n"
    "Basic Action 2: TYPE\n"
    "    - Purpose: Enter specified text at the designated location.\n"
    "    - Format: TYPE [input text]\n"
    "    - Example Usage: TYPE [Shanghai shopping mall]\n\n"
    "Basic Action 3: SCROLL\n"
    "    - Purpose: SCROLL in the specified direction.\n"
    "    - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n"
    "    - Example Usage: SCROLL [UP]\n\n"
    "2. Custom Actions\n"
    "Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n"
    "Custom Action 1: LONG_PRESS\n"
    "    - Purpose: Long press at the specified position.\n"
    "    - Format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n"
    "    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\n"
    "Custom Action 2: PRESS_BACK\n"
    "    - Purpose: Press a back button to navigate to the previous screen.\n"
    "    - Format: PRESS_BACK\n"
    "    - Example Usage: PRESS_BACK\n\n"
    "Custom Action 3: PRESS_HOME\n"
    "    - Purpose: Press a home button to navigate to the home page.\n"
    "    - Format: PRESS_HOME\n"
    "    - Example Usage: PRESS_HOME\n\n"
    "Custom Action 4: WAIT\n"
    "    - purpose: Wait for the screen to load.\n"
    "    - format: WAIT\n"
    "    - example usage: WAIT\n\n"
    "Custom Action 5: COMPLETE\n"
    "    - Purpose: Indicate the task is finished.\n"
    "    - Format: COMPLETE\n"
    "    - Example Usage: COMPLETE\n\n"


    "### Output format ###\n"
    "Your response must exactly follow the template:\n"
    "{action: ACTION_NAME}\n"
    "Replace `ACTION_NAME` with one of:\n"
    "- CLICK <point>[[x,y]]</point>\n"
    "- TYPE [input text]\n"
    "- SCROLL [UP/DOWN/LEFT/RIGHT]\n"
    "- LONG_PRESS <point>[[x,y]]</point>\n"
    "- PRESS_BACK\n"
    "- PRESS_HOME\n"
    "- WIAT\n"
    "- COMPLETE"
    )
    if model == "gpt":
        client = OpenAI(
            base_url="",
            api_key= ""
        )
        with open(obs['image_path'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(       
            model="gpt-4o",
            messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content

    elif model == "glm":
        client = ZhipuAI(
            api_key= ""
        )
        with open(obs['image_path'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":base64_image
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
        

    elif model == "qwen":
        client = OpenAI(
            base_url="",
            api_key= ""
        )
        with open(obs['image_path'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(       
            model="qwen-vl-max",
            messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
    
    print(answer)
    action_pattern = r"{action: (.+?)}"
    match = re.search(action_pattern, answer)
    if match:
        action = match.group(1)
        print(action)
    return action


base_data_path = "/data1/home/wuzheng/IFRAgent/Trajectories"
log_dir = "/data1/home/wuzheng/IFRAgent/Logs/main/GPT-4o_IFRAgent"

run_all_tests(
    test_func=test_loop_IFRAgent,
    base_data_path=base_data_path,
    log_dir=log_dir
)

