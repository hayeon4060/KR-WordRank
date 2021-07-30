from krwordrank.sentence import keysentence
from krwordrank.sentence import make_vocab_score
from krwordrank.sentence import MaxScoreTokenizer
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
    max_length = 10, # 단어의 최대 길이
    verbose = True
    )

beta = 0.85    # PageRank의 decaying factor beta
max_iter = 10

keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter, num_keywords=100)

stopwords = {'영화', '관람객', '너무', '정말', '진짜'}
vocab_score = make_vocab_score(keywords, stopwords, scaling=lambda x:1)
tokenizer = MaxScoreTokenizer(vocab_score)

penalty = lambda x: 0 if 25 <= len(x) <= 80 else 1

sents = keysentence(
    vocab_score, texts, tokenizer.tokenize,
    penalty=penalty,
    diversity=0.3,
    topk=10
)
for sent in sents:
    print(sent)