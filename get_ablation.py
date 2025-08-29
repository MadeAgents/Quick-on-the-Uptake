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

import os
import json
from tqdm import tqdm  
from openai import OpenAI
from RAG.RAGToolbox import Jinaembedding, Vectordatabase
from transformers import AutoModelForCausalLM, AutoTokenizer


base_dir = "/data1/home/wuzheng/IFRAgent/Trajectories"

def rewriting_byqwen(query, user_number, embedding_model, SOP_model, tokenizer):
    feature_path = f"../Trajectories/user{user_number}/intent_flow_explicit.json"
    rag_path = f"../Trajectories/user{user_number}/rag_database"
    db=Vectordatabase()
    db.load_vector(rag_path)
    explicit = db.query_score(query, embedding_model, 1)
    similarity, key, value = explicit[0]
    #print(value)
    with open(feature_path, 'r', encoding='utf-8') as f:
        implicit = json.load(f)
    prompt = (
    "You are now a mobile phone operation expert. I need you to help me break down a mobile operation instruction into multi-step instructions. Please strictly follow my example format for the output. I will provide you with a relevant example of instruction decomposition."
    "If the instruction I give you is in Chinese, you should output in Chinese; if the instruction I give you is in English, you should output in English."
    "For example:\n"
    f"Original instruction: {key}\n"
    f"Decomposed instructions: {value}\n"
    f"Original instruction: {query}\n"
    "Decomposed instructions:\n"
    "Please directly output the decomposed instructions in list form, without any additional text:"
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    #print(prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(SOP_model.device)
    generated_ids = SOP_model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    step_list = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return step_list

def process_user_folder(user_folder):
    user_number = int(user_folder.replace("user", ""))
    
    data_path = os.path.join(base_dir, user_folder, "test_dataset", "data.json")
    
    if not os.path.exists(data_path):
        print(f"Data file not found for {user_folder}, skipping...")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    unique_tasks = {}
    for item in data:
        if 'task' in item:
            task = item['task']
            if task not in unique_tasks:
                unique_tasks[task] = {
                    'step_list': None
                }
    for task in tqdm(unique_tasks, desc=f"Processing tasks for {user_folder}"):
        try:
            step_list = rewriting_byqwen(
                task,
                user_number, 
                embedding_model,
                SOP_model,
                tokenizer
            ) 
            unique_tasks[task]['step_list'] = step_list
        except Exception as e:
            print(f"Error processing task '{task}' in {user_folder}: {str(e)}")
            unique_tasks[task]['step_list'] = ""
    
    for item in data:
        if 'task' in item:
            task = item['task']
            item['step_list'] = unique_tasks[task]['step_list']

    
    output_path = os.path.join(base_dir, user_folder, "test_dataset", "data_ablation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(data)} items for {user_folder}, saved to {output_path}")


def main():
    user_folders = ['user1']
    for user_folder in user_folders:
        process_user_folder(user_folder)

if __name__ == "__main__":

    model_path = "/data1/home/models/Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    SOP_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    embedding_model = Jinaembedding(r"/data1/home/wuzheng/IFRAgent/Code/RAG/jina-embeddings-v2-base-zh") 
    
    main()
