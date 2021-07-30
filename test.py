# from krwordrank.sentence import summarize_with_sentences
from krwordrank.word import KRWordRank

f = open("./data/fullscript.txt", 'r', encoding="UTF8")
line = f.readline()
line=line.split(".")
texts=[]

for i in line:
    texts.append(i)
f.close()


wordrank_extractor = KRWordRank(
    min_count = 2, # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 20, # 단어의 최대 길이
    verbose = True
    )

beta = 0.85    # PageRank의 decaying factor beta
max_iter = 10

keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))
