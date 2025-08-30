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
from RAGToolbox import Jinaembedding, Vectordatabase

import json
docs = json.load(open(r"D:\Users\Desktop\data_framework_v2\Trajectories\user1\intent_flow_explicit.json", encoding='utf-8'))
embedding_model=Jinaembedding(r"D:\Users\Desktop\data_framework_v2\RAG\jina-embeddings-v2-base-zh") 
database=Vectordatabase(docs)
Vectors=database.get_vector(embedding_model)
database.persist(path='rag_database')
