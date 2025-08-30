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
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(script_dir, 'rag_database')
embedding_model=Jinaembedding(r"D:\Users\Desktop\data_framework_v2\RAG\jina-embeddings-v2-base-zh") 
db=Vectordatabase()
db.load_vector(database_path)
#SOP = db.query("点一份煎饼果子",embedding_model)
SOP = db.query_score("我需要一把刀",embedding_model,1)
print(SOP)
similarity, key, value = SOP[0]
print(similarity)
print(key)
print(value)
