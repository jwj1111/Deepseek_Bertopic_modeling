import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertModel
from tqdm import tqdm
import numpy as np

#载入文本
docs=[]
with open(r"数据/原始语料/text.txt",'r',encoding='utf-8') as infile:
    for line in infile:
        docs.append(line.strip())

# 使用Bert:hfl/chinese-bert-wwm哈工大模型
model_name="hfl/chinese-bert-wwm"
model=BertModel.from_pretrained(model_name)
tokenizer=BertTokenizer.from_pretrained(model_name)

#在gpu运行
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() #评估模式

#切分batch
batch_size=32
data_loader=DataLoader(docs,batch_size=batch_size)

#词嵌入
cls_embeddings=[]
for batch_txts in tqdm(data_loader):
    inputs=tokenizer(batch_txts,padding=True,truncation=True,return_tensors='pt',max_length=512) #编码数据，长度512，截长补短
    inputs.to(device) #将数据存于gpu
    with torch.no_grad():
        outputs=model(**inputs)
    cls_embeddings.append(outputs.last_hidden_state[:,0].cpu().numpy()) #取出每句cls的语义向量
cls_embeddings_np=np.vstack(cls_embeddings)

#保存
np.save(r"数据/embedding_cache/emb_hfl.npy",cls_embeddings_np)
