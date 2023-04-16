"""
The code is a Python implementation of a text summarization algorithm. It uses natural language processing (NLP) libraries such as NLTK and
scikit-learn to preprocess and vectorize a given text corpus. Then, it computes the similarity between sentences based on cosine similarity
and longest common subsequence (LCS) algorithms. Finally, it uses the power method to identify the most important sentences and returns a
summary of the text.

The code consists of the following functions:

    preprocess(sentence): This function preprocesses a given sentence by removing stop words, punctuation, and converting all words to
    lowercase.

    tfIdf(corpus): This function tokenizes the given corpus into sentences, preprocesses each sentence using preprocess(sentence), and
    vectorizes the sentences using the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm.

    lcs(X, Y): This function computes the length of the longest common subsequence between two strings X and Y.

    lcs_etoile(s1, s2): This function computes the average LCS ratio between a list of words s1 and a list of words s2.

    cosine(p1, p2): This function computes the cosine similarity between two vectors p1 and p2.

    similarite(frame, alpha=0.9, similarity_threshold=0.07): This function computes the similarity matrix between all pairs of sentences in
    a given dataframe frame using the cosine similarity and LCS algorithms. The parameter alpha controls the weight of the cosine
    similarity, and similarity_threshold is the threshold for two sentences to be considered similar.

    powerMethod(similarity_matrix, degrees, stopping_criterion=0.00005, max_loops=3000): This function applies the power method to a given
    similarity matrix to identify the most important sentences based on their degrees.

    Summarization(corpus, sum_size=5): This class performs summarization on a given corpus of text. The parameter corpus is the input text,
    and sum_size is the number of sentences in the summary.

    summarize(): This method of the Summarization class performs summarization using the methods described above and returns a summary of
    the text.

    graph(): This method of the Summarization class generates a graph of the sentences in the text and their similarity scores. It uses the
    NetworkX and Matplotlib libraries to create the graph.
"""

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
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running graph_summ.py")

def preprocess(sentence):
    # Use set for faster membership check
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    tokens = word_tokenize(sentence)
    # Combine lowercasing and punctuation removal
    stripped = [word.lower().translate(table) for word in tokens]
    # Use a generator expression instead of list comprehension for memory efficiency
    sent = ' '.join(word for word in stripped if word not in stop_words)
    return sent


def tfIdf(corpus):
    corpus = sent_tokenize(corpus)
    documents = [preprocess(sent) for sent in corpus]
    tfidf = TfidfVectorizer()
    tfIdf_mat = tfidf.fit_transform(documents)
    feature_names = tfidf.get_feature_names_out()
    # Convert to dense array directly
    df = pd.DataFrame(tfIdf_mat.toarray(), columns=feature_names)
    df['p_sentence'] = [word_tokenize(sent) for sent in documents]
    df['sentences'] = corpus
    return df


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    # Initialize with zeros instead of None
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):  # Start loop from 1 instead of 0
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
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
                max_l = 0  # Store max value outside loop to avoid redundant calculations
                max_idx = -1  # Store index of max value outside loop to avoid redundant calculations
                for idx, w2 in enumerate(l2):
                    l = lcs(w1, w2)
                    if l > max_l:
                        max_l = l
                        max_idx = idx
                if max_l > 0:
                    del l2[max_idx]
                    if max_l / len(w1) >= 0.6:
                        s += max_l / len(w1)
        return s / float(len(s1))


def cosine(p1, p2):
    dot_product = np.dot(p1, p2)
    magnitude = np.sqrt(np.sum(p1 ** 2)) * np.sqrt(np.sum(p2 ** 2))
    if magnitude == 0.0:
        return 0.0
    return dot_product / magnitude


def similarite(frame, alpha=0.9, similarity_threshold=0.07):
    num_rows, num_cols = frame.values.shape
    similarities = alpha * np.dot(frame.values[:, :num_cols-2], frame.values[:, :num_cols-2].T) + \
        (1 - alpha) * np.array([lcs_etoile(s1[-2], s2[-2])
                                for s1, s2 in zip(frame.values, frame.values)])
    degrees = np.zeros(shape=(num_rows))
    for ind1 in range(num_rows):
        s1 = frame.iloc[ind1].values
        for ind2 in range(num_rows):
            s2 = frame.iloc[ind2].values
            sim = alpha * cosine(s1[:num_cols-2], s2[:num_cols-2]) + \
                (1-alpha) * lcs_etoile(s1[-2], s2[-2])
            if sim > similarity_threshold:
                similarities[ind1][ind2] = 1
                degrees[ind2] += 1
            else:
                similarities[ind1][ind2] = 0.0
    return similarities / degrees.reshape(num_rows, 1), degrees


def powerMethod(similarity_matrix,
                degrees,
                stopping_criterion=0.00005,
                max_loops=3000):

    p_initial = np.ones(shape=len(degrees)) / len(degrees)
    i = 0
    while True:
        i += 1
        p_update = np.dot(similarity_matrix.T, p_initial)
        delta = np.linalg.norm(p_update - p_initial)
        if delta < stopping_criterion or i >= max_loops:
            break
        else:
            p_initial = p_update
    p_update /= np.max(p_update)
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
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        fig.update_traces(marker=dict(symbol='circle'))
        fig.show()
        return fig


# Output as txt file
input_dir = r'dataset/multi-summaries'
output_dir = r'dataset'

# Create the 'final' and 'graphs' directories if they don't exist
final_dir = os.path.join(output_dir, 'final')
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

graphs_dir = os.path.join(output_dir, 'graphs')
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # Construct the output filenames for TXT files
        output_filename = os.path.splitext(filename)[0]
        output_text_file = os.path.join(final_dir, output_filename + '.txt')

        # Read the input text file
        with open(os.path.join(input_dir, filename), 'r') as file:
            text = file.read()

        # Generate the summary and graph for TXT files
        doc = Summarization(text, 1)
        my_sum = doc.summarize()

        # Save the summary to a text file
        with open(output_text_file, 'w', encoding='utf-8') as file:
            file.write(my_sum)

        # Construct the output filename for PNG files using the same output_filename as the text file
        graph_filename = output_filename + '.png'
        file_path = os.path.join(graphs_dir, graph_filename)

        # Save the graph as an image file
        pio.write_image(doc.graph(), file_path)

        print(Fore.GREEN + f"Graph and Summary generation for {filename} completed")

print(Fore.GREEN + f"\nGraph and Summary generation completed, check dataset/multi-summaries")