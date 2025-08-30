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

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import re
import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from rewriting import rewriting_bygpt, rewriting_byqwen
from RAG.RAGToolbox import Jinaembedding, Vectordatabase
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image 


def run_all_tests(test_func, model, processor, base_data_path, log_dir, **kwargs):
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
            data_path = f"{base_data_path}/{user}/test_dataset/data_rewritten.json"
            print(f"Testing {test_func.__name__} - user {user}...")
            
            result = test_func(
                model=model,
                processor=processor,
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

def test_loop_IFRAgent(model, processor, data_path, log_dir, user_name=""):
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
                action = get_action_IFRAgent(model, processor, obs)
                action = transfer_tars2atlas(action, obs)
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

def test_loop_tars(model, processor, data_path, log_dir="logs", user_name=""):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_log_{user_name}_{timestamp}.txt"
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
                action = get_action_tars(model, processor, obs)
                action = transfer_tars2atlas(action, obs)
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

def transfer_tars2atlas(action, obs):
    if action == "press_home()":
        return "PRESS_HOME"
    elif action == "press_back()":
        return "PRESS_BACK"
    elif action.startswith("finished"):
        return "COMPLETE"
    elif action.startswith("wait"):
        return "WAIT"
        
    def extract_coordinates(action_str):
        # Get image dimensions
        try:
            with Image.open(obs['image_path']) as img:
                width, height = img.size
        except:
            width, height = 1000, 1000  # Default values if image can't be read
        
        if "point=" in action_str:
            match = re.search(r'\((\d+),(\d+)\)', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)
        elif "start_box=" in action_str:
            match = re.search(r'\((\d+),(\d+)\)', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)        
        elif "point=" in action_str:
            match = re.search(r'point=[\'"]<point>(\d+)\s+(\d+)</point>[\'"]', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)
            match = re.search(r'point=[\'"](\d+)\s+(\d+)[\'"]', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)
        return None, None
        
    if action.startswith("long_press"):
        x, y = extract_coordinates(action)
        if x and y:
            return f"LONG_PRESS <point>[[{x},{y}]]</point>"
    elif action.startswith("type"):
        content_match = re.search(r"content='([^']*)'", action)
        content = content_match.group(1) if content_match else ""
        return f"TYPE [{content}]"
    elif action.startswith("click"):
        x, y = extract_coordinates(action)
        if x and y:
            return f"CLICK <point>[[{x},{y}]]</point>"
    elif action.startswith("scroll"):
        x, y = extract_coordinates(action)
        dir_match = re.search(r"direction='(up|down|left|right)'", action, re.IGNORECASE)
        direction = dir_match.group(1).upper() if dir_match else "UP"
        if x and y:
            return f"SCROLL [{direction}]"
    return action


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

def get_action_tars(model, processor, obs):
    prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
            ## Output Format
            ```
            Thought: ...
            Action: ...
            ```
            ## Action Space

            click(point='<point>x1 y1</point>')
            long_press(point='<point>x1 y1</point>')
            type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
            scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
            press_home()
            press_back()
            wait() #Sleep for 5s and take a screenshot to check for any changes.
            finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


            ## Note
            - Use Chinese in `Thought` part.
            - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

            ## User Instruction
            {obs['task']}
            """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                }
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    #print(output_text[0])
    pattern = r"Action:\s*(.*?)<\|im_end\|>"
    match = re.search(pattern, output_text[0])
    if match:
        action = match.group(1)
        print(action)
    else:
        print("No matching content found.")
    return action

def get_action_IFRAgent(model, processor, obs):
    prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
            ## Output Format
            ```
            Thought: ...
            Action: ...
            ```
            ## Action Space

            click(point='<point>x1 y1</point>')
            long_press(point='<point>x1 y1</point>')
            type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
            scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
            press_home()
            press_back()
            wait() #Sleep for 5s and take a screenshot to check for any changes.
            finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


            ## Note
            - Use Chinese in `Thought` part.
            - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

            ## User Instruction
            {obs['query_rewritten']}

            ## Standard Operating Procedure
            {obs['step_list_rewritten']}
            """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                }
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    #print(output_text[0])
    pattern = r"Action:\s*(.*?)<\|im_end\|>"
    match = re.search(pattern, output_text[0])
    if match:
        action = match.group(1)
        print(action)
    else:
        print("No matching content found.")
    return action

# model_path = "/data1/home/models/Qwen/Qwen3-4B"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# help_model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="bfloat16",
#     device_map="auto",
#     attn_implementation="flash_attention_2"
# )

# embedding_model = Jinaembedding(r"/data1/home/wuzheng/IFRAgent/Code/RAG/jina-embeddings-v2-base-zh") 
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data1/home/models/UI-TARS-1.5-7B", torch_dtype="bfloat16", device_map="auto",attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(
    "/data1/home/models/UI-TARS-1.5-7B", use_fast=True
)
base_data_path = "/data1/home/wuzheng/IFRAgent/Trajectories"
log_dir = "/data1/home/wuzheng/IFRAgent/Logs/main/TARS1.5_IFRAgent"

run_all_tests(
    test_func=test_loop_IFRAgent,
    model=model,
    processor=processor,
    base_data_path=base_data_path,
    log_dir=log_dir
)
