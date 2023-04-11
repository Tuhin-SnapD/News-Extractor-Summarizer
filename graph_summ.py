""" 
The code is a Python implementation of a summarization algorithm that extracts the most relevant sentences from a given text corpus. It uses 
natural language processing (NLP) libraries such as NLTK and scikit-learn to preprocess and vectorize the text corpus, and then computes the 
similarity between sentences based on the cosine similarity and longest common subsequence (LCS) algorithms. Finally, it uses the power method to 
identify the most important sentences and returns a summary of the text.

The code consists of the following functions:

preprocess(sentence) - preprocesses a given sentence by removing stop words, punctuation, and converting all words to lowercase.

tfIdf(corpus) - tokenizes the given corpus into sentences, preprocesses each sentence using preprocess(sentence), and vectorizes the sentences 
using the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm.

lcs(X, Y) - computes the length of the longest common subsequence between two strings X and Y.

lcs_etoile(s1, s2) - computes the average LCS ratio between a list of words s1 and a list of words s2.

cosine(p1, p2) - computes the cosine similarity between two vectors p1 and p2.

similarite(frame, alpha=0.9, similarity_threshold=0.07) - computes the similarity matrix between all pairs of sentences in a given dataframe 
frame using the cosine similarity and LCS algorithms. The parameter alpha controls the weight of the cosine similarity, and similarity_threshold 
is the threshold for two sentences to be considered similar.

powerMethod(similarity_matrix, degrees, stopping_criterion=0.00005, max_loops=3000) - applies the power method to a given similarity matrix to 
identify the most important sentences based on their degrees.

Summarization(corpus, sum_size=5) - a class that performs summarization on a given corpus of text. The parameter corpus is the input text, and 
sum_size is the number of sentences in the summary.

summarize() - a method of the Summarization class that performs summarization using the methods described above and returns a summary of the text.

graph() - a method of the Summarization class that generates a graph of the sentences in the text and their similarity scores. It uses the NetworkX and Matplotlib libraries to create the graph. """

from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import plotly.graph_objs as go
import os
import plotly.io as pio


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
counter = 0
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
    
    counter = 0

    def graph(self):
        edges = []
        for ids, v in enumerate(self.similarite):
            for i, s in enumerate(v):
                if i != ids and s > 0.0:
                    edges.append(("s"+str(ids), "s"+str(i)))
        G = nx.Graph()
        G.add_edges_from(edges)
        options = {'node_size': 800, 'width': 1, 'arrowstyle': '-'}
        pos = nx.kamada_kawai_layout(G)         
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=[],
                size=10,
                line=dict(width=2)))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['color'] += tuple(['#c7e9b4'])
            node_trace['text'] += tuple([node])

        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#a6a6a6'),
            mode='lines')
            
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Graph of Similarity',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        fig.update_traces(marker=dict(symbol='circle'))
        fig.show()
        # Save the graph as an image file in the specified output directory
        output_directory = "dataset/graphs"  # Specify the desired output directory
        file_name = "graph_similarity.png"  # Specify the desired file name and format
        # Create the output directory if it does not exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Save the graph as an image file with a predefined file name based on the counter
        global counter  # Declare the counter as global again
        counter += 1  # Increment the counter

        if counter == 1:
            file_name = "business.png"
        elif counter == 2:
            file_name = "entertainment.png"
        elif counter == 3:
            file_name = "health.png"
        elif counter == 4:
            file_name = "science.png"
        else:
            file_name = "graph_similarity_{}.png".format(counter)

        file_path = os.path.join(output_directory, file_name)
        pio.write_image(fig, file_path)

# # Open the file in read mode
# with open('summaries/business_one_line_summary.txt', 'r') as file:
#     # Read the contents of the file
#     corpus = file.read()

# doc = Summarization(corpus, 2)
# my_sum = doc.summarize()

# doc.graph()
# print(my_sum)

input_dir = r'dataset/multi-summaries'
output_dir = r'dataset/final'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# Iterate through all txt files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # Construct the output filenames
        output_filename = os.path.splitext(filename)[0]
        output_text_file = os.path.join(output_dir, output_filename + '.txt')
        output_image_file = os.path.join(output_dir, output_filename + '.png')

        # Open the input text file
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
            text = file.read()

        # Generate the summary and graph
        doc = Summarization(text, 1)
        my_sum = doc.summarize()
        doc.graph()

        # Save the summary to a text file
        with open(output_text_file, 'w', encoding='utf-8') as file:
            file.write(my_sum)

        # Save the graph as a PNG image
        #fig.write_image(output_image_file)

# # Iterate through all files in the input directory
# for filename in os.listdir(input_dir):
#     if filename.endswith('.txt'):
#         # Construct the output filename
#         output_filename = os.path.splitext(filename)[0] + '.txt'
#         # Open the input CSV file
#         with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
#             # create a reader 
#             text = file.read()

#     doc = Summarization(text, 1)
#     my_sum = doc.summarize()
#     doc.graph() #Generates image using plotly.graph_objs

