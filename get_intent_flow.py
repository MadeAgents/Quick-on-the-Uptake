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

from openai import OpenAI
import json
import os
import base64
from typing import List, Dict, Any

def get_intent_flow_loop():
    for i in range(7, 8): 
        get_intent_flow(i)
    return 0

user_number = 1
def get_intent_flow(user_number):
    feature_path = f"./Trajectories/user{user_number}/intent_flow_implicit.json"
    images_path = f"./Trajectories/user{user_number}/support_dataset"  
    with open(feature_path, 'r', encoding='utf-8') as f:
        intent_flow_implicit = json.load(f)
    intent_flow_explicit = []

    data_json_path = os.path.join(images_path, 'data.json')
    if not os.path.exists(data_json_path):
        print(f"Data file not found at {data_json_path}")
    try:
        with open(data_json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        items = []
        prev_task = None
        for item in data_list:
            current_task = item.get('task')
            if prev_task is None or current_task == prev_task:
                items.append(item)
            else:
                if items:
                    intent_flow_implicit = process_single_trajectory_implicit(items, intent_flow_implicit)
                    intent_flow_explicit_single = process_single_trajectory_explicit(items)
                    intent_flow_explicit.append(intent_flow_explicit_single)
                    items = [item]                        
            prev_task = current_task
        if items:
            intent_flow_implicit = process_single_trajectory_implicit(items, intent_flow_implicit)
            intent_flow_explicit_single = process_single_trajectory_explicit(items)
            intent_flow_explicit.append(intent_flow_explicit_single)     
    except json.JSONDecodeError:
        print(f"Invalid JSON in data file at {data_json_path}")
    except UnicodeDecodeError:
        print(f"Encoding error in data file at {data_json_path}")

    
    output_path = f"./Trajectories/user{user_number}/intent_flow_explicit.json"
    output_path2 = f"./Trajectories/user{user_number}/intent_flow_implicit.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(intent_flow_explicit, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved intent_flow_explicit to {output_path}")
    except Exception as e:
        print(f"Failed to save intent_flow_explicit to {output_path}: {str(e)}")
    try:
        with open(output_path2, 'w', encoding='utf-8') as f:
            json.dump(intent_flow_implicit, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved intent_flow_implicit to {output_path2}")
    except Exception as e:
        print(f"Failed to save intent_flow_implicit to {output_path2}: {str(e)}")

def call_openai_api(prompt: str, image_paths: List[str]) -> str:
    client = OpenAI(
        base_url="https://api.gpts.vin/v1",
        api_key="sk-nO4o2zE3reOUJRVq4BS4xPW0CqoniTIwl7Cys11i2ce1zzsr"
    )
    
    content = [{"type": "text", "text": prompt}]
    
    for path in image_paths:
        try:
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
        except Exception as e:
            print(f"⚠️ Image: {path}, error: {e}")
            continue
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": "API failed"})

def process_single_trajectory_explicit(items: List[Dict[str, Any]]) -> str:
    prompt = (
        "You are an expert in identifying user operation workflows based on user action trajectory images."
        f"The current instruction the user is executing is {items[0]['task']}."
        "You need to complete the list corresponding to the step_list key in the following JSON format."
        "Please pay special attention to areas that may reflect user habits, such as different sorting methods for search results, preferred flavors when users order food, and the tone of voice users use when chatting with different people."
        "If the task is in Chinese, then your step_list should also be in Chinese. If the task is in English, then your step_list should also be in English."
        "Please only output your response in the following JSON format, without any additional output."
        '{"task": "' + items[0]['task'] + '", '
        '"step_list": [""]}'
    )
    image_paths = [item['image_path'] for item in items]
    answer = call_openai_api(prompt, image_paths)
    return answer

def process_single_trajectory_implicit(items: List[Dict[str, Any]], feature_data: Dict[str, Any]) -> str:    
    prompt = (
        "You are an expert in extracting user profiles based on user action instructions and behavior trajectories.\n"
        "The current user profile is as follows:\n"
        f"{json.dumps(feature_data, indent=2, ensure_ascii=False)}\n"
        f"The current instruction the user is executing is {items[0]['task']}\n"
        "If the instruction is in Chinese, your filled content must also be in Chinese. If the instruction is in English, your filled content must also be in English.\n"
        "You must strictly adhere to the existing JSON format of the user profile, and can only fill in content within the original JSON structure.\n"
         "Please pay special attention to areas that may reflect user habits, such as different sorting methods for search results, preferred flavors when users order food, and the tone of voice users use when chatting with different people."
        "You need to strictly determine which domain in 'domain behavior preferences' and which software in 'software behavior preferences' the current action belongs to,\n"
        "and avoid modifying unrelated domain behavior preferences or software behavior preferences.\n"
        "If you identify obvious errors in the existing user profile, you should confidently correct them.\n"
        "You should extract the user's behavioral preferences and habits, including but not limited to:\n"
        "- Food preferences when ordering meals\n"
        "- Language style in conversations\n"
        "- Whether they particularly focus on or like videos when watching them\n"
        "- Specific sorting habits when shopping or browsing search results\n"
        "Please only output the modified user profile in JSON format, without any additional output."
    )
    image_paths = [item['image_path'] for item in items]
    answer = call_openai_api(prompt, image_paths)
    return answer
    
get_intent_flow_loop()
#get_intent_flow_loop()
