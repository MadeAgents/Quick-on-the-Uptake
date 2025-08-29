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
from RAG.RAGToolbox import Jinaembedding, Vectordatabase
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

user_number = 1
def rewriting_bygpt(query, user_number, embedding_model):
    feature_path = f"../Trajectories/user{user_number}/intent_flow_implicit.json"
    rag_path = f"../Trajectories/user{user_number}/rag_database"
    db=Vectordatabase()
    db.load_vector(rag_path)
    explicit = db.query_score(query, embedding_model, 1)
    similarity, key, value = explicit[0]
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
    client = OpenAI(
        base_url="",
        api_key= ""
    )
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                    ],
                }
            ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    chat_response = completion
    step_list = chat_response.choices[0].message.content
    prompt = (
        'You are now a personalized instruction rewriting expert. I will provide you with a user instruction, a set of sub-steps decomposed from the instruction, and a user profile. Your goal is to rewrite the instruction and its sub-steps based on the user profile to better meet the user’s personalized needs.'
        f'User instruction: {query}'
        f'Instruction sub-steps: {step_list}'
        f'User profile: {implicit}'
        'First, you must determine which domain scenario this instruction belongs to: diet, journey, chat, video, shop, search, or music.'
        'Then, identify the user’s software usage preferences in this domain scenario.'
        'Next, rewrite the user instruction and sub-steps based on the relevant domain behavior preferences and software behavior preferences.'
        'You should pay attention to common user preferences in the domain behavior and software behavior, such as taste preferences in food ordering, tone in chatting, specific sorting habits after searching, habits of liking videos or following creators, etc.'
        'Incorporate these observed common preferences into the rewritten instruction and sub-steps.'
        'Your output must be in strict JSON format, containing the following fields:'
        '{'
        '  "domain_scenario": "Identified domain scenario string",'
        '  "software_preference": "User’s software preference in this domain (string)",'
        '  "rewritten_instruction": "Rewritten user instruction (string)",'
        '  "rewritten_substeps": ["Rewritten", "sub-steps", "string list"]'
        '}'
        'Note: You must only return this JSON object, without any additional explanations or comments.'
        'If the instruction I give you is in Chinese, the JSON values must also be in Chinese; if the instruction is in English, the JSON values must be in English. However, the JSON keys must remain unchanged.'
    )
    #print(prompt)
    client = OpenAI(
        base_url="",
        api_key=""
    )
    messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                        ],
                    }
                ]
    completion = client.chat.completions.create(
          model="gpt-4o",
          messages=messages
        )
    chat_response = completion
    result = chat_response.choices[0].message.content
    cleaned_result = re.sub(r'^```json|```$', '', result, flags=re.MULTILINE).strip()
    data = json.loads(cleaned_result)
    query_rewritten = data["rewritten_instruction"]
    step_list_rewritten = data["rewritten_substeps"]
    #print(query)
    #print(query_rewritten)
    #print(step_list_rewritten)
    return query_rewritten, step_list_rewritten

def rewriting_byqwen(query, user_number, embedding_model, SOP_model, rewriting_model, tokenizer):
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
    #print(step_list)
    prompt = (
        'You are now a personalized instruction rewriting expert. I will provide you with a user instruction, a set of sub-steps decomposed from the instruction, and a user profile. Your goal is to rewrite the instruction and its sub-steps based on the user profile to better meet the user’s personalized needs.'
        f'User instruction: {query}'
        f'Instruction sub-steps: {step_list}'
        f'User profile: {implicit}'
        'First, you must determine which domain scenario this instruction belongs to: diet, journey, chat, video, shop, search, or music.'
        'Then, identify the user’s software usage preferences in this domain scenario.'
        'Next, rewrite the user instruction and sub-steps based on the relevant domain behavior preferences and software behavior preferences.'
        'You should pay attention to common user preferences in the domain behavior and software behavior, such as taste preferences in food ordering, tone in chatting, specific sorting habits after searching, habits of liking videos or following creators, etc.'
        'Incorporate these observed common preferences into the rewritten instruction and sub-steps.'
        'Your output must be in strict JSON format, containing the following fields:'
        '{'
        '  "domain_scenario": "Identified domain scenario string",'
        '  "software_preference": "User’s software preference in this domain (string)",'
        '  "rewritten_instruction": "Rewritten user instruction (string)",'
        '  "rewritten_substeps": ["Rewritten", "sub-steps", "string list"]'
        '}'
        'Note: You must only return this JSON object, without any additional explanations or comments.'
        'If the instruction I give you is in Chinese, the JSON values must also be in Chinese; if the instruction is in English, the JSON values must be in English. However, the JSON keys must remain unchanged.'
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
    model_inputs = tokenizer([text], return_tensors="pt").to(rewriting_model.device)
    generated_ids = rewriting_model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    cleaned_result = re.sub(r'^```json|```$', '', result, flags=re.MULTILINE).strip()
    data = json.loads(cleaned_result)
    query_rewritten = data["rewritten_instruction"]
    step_list_rewritten = data["rewritten_substeps"]
    #print(query)
    print(query_rewritten)
    print(step_list_rewritten)
    return query_rewritten, step_list_rewritten


# embedding_model=Jinaembedding(r"/data1/home/wuzheng/IFRAgent/Code/RAG/jina-embeddings-v2-base-zh") 
# query = "点一份麻辣烫"
# query_rewritten, step_list_rewritten = rewriting_bygpt(query, user_number, embedding_model)

# model_path = "/data1/home/models/Qwen/Qwen3-0.6B"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto"
# )
# query_rewritten, step_list_rewritten = rewriting_byqwen(query, user_number, embedding_model, model, tokenizer)
# print(query_rewritten)
# print(step_list_rewritten)
