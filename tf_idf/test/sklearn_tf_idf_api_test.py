from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

"""
scikit-learn包进行TF-IDF分词权重计算主要用到了两个类：CountVectorizer和TfidfTransformer。其中
CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在第i
个文本下的词频。即各个词语出现的次数，通过get_feature_names()可看到所有文本的关键字，通过toarray()
可看到词频矩阵的结果。
"""
# 实例化CountVectorizer
countVectorizer = CountVectorizer()

# 语料库
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    ]

term_matrix = countVectorizer.fit_transform(corpus)

# 打印词频矩阵
print(term_matrix.toarray())

# 打印文本关键字

print(countVectorizer.get_feature_names())

# TfidfTransformer是统计vectorizer中每个词语的tf-idf值
tfidfTransformer = TfidfTransformer()

tfidf_matrix = tfidfTransformer.fit_transform(term_matrix)

print(tfidf_matrix.toarray())