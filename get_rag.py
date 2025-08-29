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

import json
from RAG.RAGToolbox import Jinaembedding, Vectordatabase

def get_rag_loop(embedding_model):
    for i in range(1,10): 
        get_rag(i,embedding_model)
    return 0


def get_rag(user_number,embedding_model):
    explicit_path = f"../Trajectories/user{user_number}/intent_flow_explicit.json"
    docs = json.load(open(explicit_path, encoding='utf-8'))
    transformed_docs = {item["task"]: item["step_list"] for item in docs}
    database=Vectordatabase(transformed_docs)
    Vectors=database.get_vector(embedding_model)
    database.persist(path=f"../Trajectories/user{user_number}/rag_database")

embedding_model=Jinaembedding(r"/data1/home/wuzheng/IFRAgent/Code/RAG/jina-embeddings-v2-base-zh") 
get_rag_loop(embedding_model)
