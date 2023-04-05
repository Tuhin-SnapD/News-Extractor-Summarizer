from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def preprocess(sentence):
    stop_words = stopwords.words('english')
    words = word_tokenize(sentence)
    tokens = [word.lower() for word in words]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    sent = ' '.join([word for word in stripped if word not in stop_words])
    return sent


def tfIdf(corpus):
    corpus = sent_tokenize(corpus)
    documents = [preprocess(sent) for sent in corpus]
    tfidf = TfidfVectorizer()
    tfIdf_mat = tfidf.fit_transform(documents)
    df = pd.DataFrame(tfIdf_mat.todense(),
                      columns=tfidf.get_feature_names_out())
    df['p_sentence'] = [word_tokenize(sent) for sent in documents]
    df['sentences'] = corpus
    return df


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]


def lcs_etoile(s1, s2):
    if len(s1) == 0:
        return 0
    else:
        l2 = s2.copy()
        s = 0
        for w1 in s1:
            if len(l2) > 0:
                l = [lcs(w1, w2) for w2 in l2]
                if max(l) > 0:
                    del l2[l.index(max(l))]
                    if max(l) / len(w1) >= 0.6:
                        s += max(l) / len(w1)
        return s / float(len(s1))


def cosine(p1, p2):
    dot_product = sum(p*q for p, q in zip(p1, p2))
    magnitude = sqrt(sum([val**2 for val in p1])) * \
        sqrt(sum([val**2 for val in p2]))
    if not magnitude:
        return 0.0
    return dot_product/magnitude


def similarite(frame, alpha=0.9, similarity_threshold=0.07):
    similarities = np.zeros(shape=(frame.values.shape[0],
                                   frame.values.shape[0]))
    degrees = np.zeros(shape=(frame.values.shape[0]))
    for ind1, s1 in frame.iterrows():
        for ind2, s2 in frame.iterrows():
            sim = alpha*cosine(s1.values[:len(frame.columns)-2], s2.values[
                :len(frame.columns)-2]) + (1-alpha)*lcs_etoile(s1.values[
                    -2], s2.values[-2])
            if sim > similarity_threshold:
                similarities[ind1][ind2] = 1
                degrees[ind2] += 1
            else:
                similarities[ind1][ind2] = 0.0
    return similarities / degrees.reshape(frame.values.shape[0], 1), degrees


def powerMethod(similarity_matrix,
                degrees,
                stopping_criterion=0.00005,
                max_loops=3000):

    p_initial = np.ones(shape=len(degrees))/len(degrees)
    i = 0
    # loop until no change between successive matrix iterations
    while True:
        i += 1
        p_update = np.matmul(similarity_matrix.T, p_initial)
        delta = np.linalg.norm(p_update - p_initial)
        if delta < stopping_criterion or i >= max_loops:
            break
        else:
            p_initial = p_update
    p_update = p_update/np.max(p_update)
    return p_update


class Summarization:

    def __init__(self, corpus, sum_size=5):
        self.corpus = corpus
        self.sum_size = sum_size
        self.scores = ''
        self.similarite = ''

    def summarize(self):
        tf_idf = tfIdf(self.corpus)
        self.similarite, degrees = similarite(tf_idf)
        self.scores = powerMethod(self.similarite, degrees)
        sent_index = np.argsort(self.scores)[::-1][:self.sum_size]
        sent_index.sort()
        sent_list = list(tf_idf.loc[sent_index]['sentences'].values)
        summu = ' '.join(sent_list)
        return summu

    def graph(self):
        edges = []
        for ids, v in enumerate(self.similarite):
            for i, s in enumerate(v):
                if i != ids and s > 0.0:
                    edges.append(("s"+str(ids), "s"+str(i)))
        G = nx.Graph()
        G.add_edges_from(edges)
        options = {'node_size': 800, 'width': 1, 'arrowstyle': '-'}
        pos = nx.fruchterman_reingold_layout(G)
        plt.figure(figsize=(12, 12))
        nx.draw_networkx(G, pos, arrows=True, with_labels=True, 
                        node_color='#c7e9b4', edge_color='#a6a6a6', **options)
        # plt.axis('off')
        plt.show()


corpus = """Acer has launched its new Aspire 3 15 and Aspire 3 14 laptops with what it claims are India's first laptop with an Intel N-series CPU. The Acer Aspire 3 is powered by the newly launched Intel Core i3-N305 CPU and is designed for casual and everyday use. The specifications are quite decent as the Aspire 3 features 8GB of RAM, SSD storage, a full-HD display, Windows 11, and a starting weight of around 1.5kg.

The laptops feature Acer's PurifiedVoice and AI Noise Reduction software enhancements, which are said to effectively analyse environmental ambient sound components and automatically select the most effective noise cancelling mode.

Acer's Aspire 3 also comes with BlueLightShield to help lower the harmful light exposure to the users. It comes with LPDDR5 RAM and is said to feature an enhanced thermal system for more effective heat dissipation.

Acer Nitro 5 With AMD Ryzen 7000 Series Launched in India At This Price
Acer Aspire 3 15, Aspire 3 14 price in India and availability
The Acer Aspire 3 15 officially starts at Rs. 39,999 in India and is available through Acer's offline and online stores, and other popular stores such as Vijay Sales and Amazon. Looking at Acer's online store, the 15-inch model is currently priced at Rs. 33,990 and Rs. 37,990 for the 256GB and 512GB SSD variants, respectively. Meanwhile, there's only a single variant of the 14-inch model listed for Rs. 37,990 that comes with 512GB of SSD storage.

Acer Aspire 3 15, Aspire 3 14 specifications
Acer Aspire 3 15 is only available in Pure Silver in India, and measures 18.9mm in thickness with a weight of about 1.7kg. The Aspire 3 14 on the other hand has the same thickness as the 15-inch model, but weighs 1.5kg. Both sizes offer full-HD resolution (1920x1080) TFT LCD displays with Acer's BlueLightShield to cut out blue light.

Wireless connectivity includes Wi-Fi 6 and Bluetooth 5.1. All models come with 8GB of LPDDR5 RAM with either 256GB or 512GB of PCIe Gen3 (NVMe) SSD storage."""

doc = Summarization(corpus, 1)
my_sum = doc.summarize()

doc.graph()
print(my_sum)
