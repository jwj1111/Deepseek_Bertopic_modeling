from sentence_transformers import SentenceTransformer
import numpy as np

#载入文本
docs=[]
with open(r"数据/原始语料/text.txt",'r',encoding='utf-8') as infile:
    for line in infile:
        docs.append(line.strip())

#词嵌入（使用sentence-transformer：paraphrase-multilingual-MiniLM-L12-v2）
embedding_model=SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings=embedding_model.encode(docs,show_progress_bar=True)
print(type(embeddings),embeddings.shape)

#保存文本向量
np.save(r'数据/embedding_cache/emb_sbert.npy',embeddings)
