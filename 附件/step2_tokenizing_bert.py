import jieba
import string

#载入文本
docs=[]
with open(r"数据/原始语料/text.txt",'r',encoding='utf-8') as infile:
    for line in infile:
        docs.append(line.strip())

#设置标点、停用词
puncts=list('…。！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～'+string.punctuation)
stopwords=[]
with open(r"数据/others/hit_stopwords.txt",'r',encoding='utf-8') as infile:
    for line in infile:
        stopwords.append(line.strip())
#设置用户词典
jieba.load_userdict(r"数据/others/userdict.txt")

#分词
cutted_txts=[]
for doc in docs:
    if len(doc)>512:
        doc=doc[:512]
    txt_list=jieba.lcut(doc)
    txt_list_filtered=[token for token in txt_list if token not in stopwords and token not in puncts and token is not ' ']
    cutted_txt=' '.join(txt_list_filtered)
    cutted_txts.append(cutted_txt)

#保存
with open(r'数据/tokenizing_cache/text_cutted_bert.txt','w',encoding='utf-8') as outfile:
    for cutted_txt in cutted_txts:
        outfile.write(cutted_txt+'\n')