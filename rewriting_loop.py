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
from rewriting import rewriting_bygpt, rewriting_byqwen

base_dir = "/data1/home/wuzheng/IFRAgent/Trajectories"

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
                    'query_rewritten': None,
                    'step_list_rewritten': None
                }
    for task in tqdm(unique_tasks, desc=f"Processing tasks for {user_folder}"):
        try:
            query_rewritten, step_list_rewritten = rewriting_byqwen(
                task,
                user_number, 
                embedding_model, 
                SOP_model, 
                rewriting_model, 
                tokenizer
            )
            
            unique_tasks[task]['query_rewritten'] = query_rewritten
            unique_tasks[task]['step_list_rewritten'] = step_list_rewritten
        except Exception as e:
            print(f"Error processing task '{task}' in {user_folder}: {str(e)}")
            unique_tasks[task]['query_rewritten'] = ""
            unique_tasks[task]['step_list_rewritten'] = []

    for item in data:
        if 'task' in item:
            task = item['task']
            item['query_rewritten'] = unique_tasks[task]['query_rewritten']
            item['step_list_rewritten'] = unique_tasks[task]['step_list_rewritten']
    
    output_path = os.path.join(base_dir, user_folder, "test_dataset", "data_rewritten_warmtest.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(data)} items for {user_folder}, saved to {output_path}")


def main():
    user_folders = [f for f in os.listdir(base_dir) if f.startswith("user") and os.path.isdir(os.path.join(base_dir, f))]
    print(user_folders)


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
    rewriting_model_path = "/data1/home/wuzheng/LLaMA-Factory-main/saves/IFRAGENT"
    rewriting_model = AutoModelForCausalLM.from_pretrained(
        rewriting_model_path,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    embedding_model = Jinaembedding(r"/data1/home/wuzheng/IFRAgent/Code/RAG/jina-embeddings-v2-base-zh") 
    
    main()
