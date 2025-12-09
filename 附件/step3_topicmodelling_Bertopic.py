import numpy as np
from bertopic import BERTopic
from transformers.pipelines import pipeline
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

#1.载入文本
text_cutted='text_cutted_bert.txt'
docs=[]
with open(f"数据/tokenizing_cache/{text_cutted}",'r',encoding='utf-8') as infile:
    for line in infile:
        docs.append(line.strip())

#2.构建bertopic
#创建词向量模型，读取本次缓存向量
embedding_model=pipeline('feature-extraction',model='hfl/chinese-bert-wwm')
embeddings=np.load(r"数据/embedding_cache/emb_hfl.npy")
#创建降维模型
umap_model=UMAP(
    n_neighbors=15, #关注向量附近15个（与bertopic中umap默认参数一致）
    n_components=5, #向量降为5维（与bertopic中umap默认参数一致）
    min_dist=0.0, #最小距离（与bertopic中umap默认参数一致）（可控制离群值）
    metric='cosine', #余弦距离（与bertopic中umap默认参数一致）
    random_state=42 #随机种子（与bertopic中umap默认参数一致）
)
#创建聚类模型
hdbscan_model=HDBSCAN(
    min_cluster_size=50, #聚类大小（可控制离群值）
    min_samples=20, #最少参考样本（可控制离群值）
    metric='euclidean' #欧几里得距离(与bertopic中hdbscan默认参数一致)
)
#创建统计模型
vectorizer_model=CountVectorizer(stop_words=['deepseek','deep','seek','DeepSeek','Deep','Seek','深度求索'])
#搭建Bertopic
topic_model=BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model
)

#3.训练模型
topics,probs=topic_model.fit_transform(docs,embeddings=embeddings)#传入训练好的词向量
#查看主题
topic_model.get_topic_info()
#设置标签名
topic_model.set_topic_labels({
    0:"主题一",
    1:"主题二",
    2:"主题三",
    3:"主题四",
    4:"主题五",
    5:"主题六",
    6:"主题七",
    7:"主题八"
})
#查看文档分布（UMAP散点图）
reduced_embeddings=UMAP(n_neighbors=10,n_components=2,min_dist=0.0,metric='cosine').fit_transform(embeddings)
clustering_result=topic_model.visualize_documents(docs,reduced_embeddings=reduced_embeddings,hide_document_hover=True,custom_labels=True)
clustering_result.write_html(r"数据/clustering_result.html")
#主题关键词可视化
keywords_html=topic_model.visualize_barchart(n_words=10,width=500,height=500,custom_labels=True)
keywords_html.write_html(r"数据/关键词.html")
#保存聚类信息（用于解读主题）
topic_docs=topic_model.get_document_info(docs)
original_texts=[]
with open(r"数据/原始语料/text.txt",'r',encoding='utf-8') as infile:
    for line in infile:
        original_texts.append(line.strip())
topic_docs.insert(1,'og_texts',original_texts)
topic_docs.to_excel(r"数据/聚类结果.xlsx",sheet_name='documents',index=False,header=True)
