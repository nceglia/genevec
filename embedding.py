from singlecellexperiment import SingleCellExperiment

import torch
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
import seaborn as sns
import pandas
import matplotlib.cm as cm
import umap
import tqdm

import numpy
import operator
import random
import pickle
import collections
import sys
import os

import genevec

class GeneEmbedding(object):

    def __init__(self, vector, context):
        self.vector = vector
        self.context = context
        self.embeddings = dict()
        for gene in tqdm.tqdm(self.context.expressed_genes):
            embedding = self.vector[self.context.gene_index[gene]]
            self.embeddings[gene] = embedding

    def compute_similarities(self):
        similarities = collections.defaultdict(dict)
        for gene in tqdm.tqdm(self.context.expressed_genes):
            embedding = self.embeddings[gene]
            distances = dict()
            for target in self.context.expressed_genes:
                if gene in similarities[target]:
                    distances[target] = similarities[target][gene]
                v = self.embeddings[target]
                distance = float(cosine_similarity(numpy.array(embedding).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
                distances[target] = distance
            similarities[gene] = distances
        return similarities

    def kmeans(self, k=10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.vector)
        clusters = zip(self.context.expressed_genes, kmeans.labels_)
        _clusters = []
        for gene, cluster in clusters:
            _clusters.append("C"+str(cluster))
        return _clusters

    def plot_umap(self, png, genes, clusters):
        trans = umap.UMAP(n_neighbors=5, random_state=42).fit(self.vector)
        x = trans.embedding_[:, 0]
        y = trans.embedding_[:, 1]
        plt.figure(figsize = (8,8))
        data = {"x":x,"y":y, "Cluster":clusters}
        df = pandas.DataFrame.from_dict(data)
        ax = plt.gca()
        for x, y, gene in zip(x, y, self.context.expressed_genes):
            if gene in genes:
                ax.text(x+.02, y, str(gene), fontsize=8)
        sns.scatterplot(data=df,x="x", y="y",hue="Cluster")
        plt.savefig(png)

    def plot_tsne(self, png, genes, clusters):
        pca = TSNE(n_components=2)
        pcs = pca.fit_transform(self.vector)
        pcs = numpy.transpose(pcs)
        plt.figure(figsize = (8,8))
        data = {"x":pcs[0],"y":pcs[1], "Cluster":clusters}
        df = pandas.DataFrame.from_dict(data)
        ax = plt.gca()
        for x, y, gene in zip(pcs[0], pcs[1], self.context.expressed_genes):
            if gene in genes:
                ax.text(x+.02, y, str(gene), fontsize=8)
        sns.scatterplot(data=df,x="x", y="y",hue="Cluster")
        plt.savefig(png)

    def plot_distance_vs_expression(self, png):
        similarities = self.compute_similarities()
        exprs = []
        dist = []
        genes = self.context.expressed_genes
        for genea in genes:
            for geneb in genes:
                cells = list(self.context.data[gene].keys())
                expression = []
                for cell in set(cells).intersection(set(list(context.data[genea].keys()))):
                    exp = context.data[genea][cell] - context.data[gene][cell]
                    expression.append(abs(exp))
                dist.append(similarities[genea][genen])
                exprs.append(numpy.mean(expression))
        print(max(exprs), min(exprs))
        plt.figure()
        plt.scatter(dist,exprs)
        plt.savefig(png)

class CellEmbedding(object):

    def __init__(self, context, embed):
        self.context = context
        self.embed = embed
        self.celltypes = dict(zip(context.cells, context.metadata["cell_type"]))
        self.samples = dict(zip(context.cells, context.metadata["patient_id"]))
        self.data = collections.defaultdict(list)
        for cell, genes in self.context.cell_to_gene.items():
            for gene in genes:
                self.data[cell].append(embed.embeddings[gene])
        self.matrix = []
        self.celltype = []
        self.sample = []
        for cell, vectors in self.data.items():
            xvec = list(numpy.average(vectors, axis=0))
            self.matrix.append(xvec)
            self.celltype.append(self.celltypes[cell])
            self.sample.append(self.samples[cell])

    def kmeans(self, k=10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.matrix)
        clusters = kmeans.labels_
        _clusters = []
        for cluster in clusters:
            _clusters.append("C"+str(cluster))
        return _clusters

    def compute_similarities(self):
        labels = self.kmeans()
        cluster_genes = dict()
        vectors = collections.defaultdict(list)
        for vec, label in zip(self.matrix, labels):
            vectors[label].append(vec)
        for label, vecs in vectors.items():
            distances = dict()
            vec = list(numpy.average(vecs, axis=0))
            for target in self.context.expressed_genes:
                v = self.embed.embeddings[target]
                distance = float(cosine_similarity(numpy.array(vec).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
                distances[target] = distance
            sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
            cluster_genes[label] = [x[0] for x in sorted_distances[:20]]
        return cluster_genes

    def plot_tsne(self, png):
        clusters = self.kmeans()
        pca = TSNE(n_components=2)
        pcs = pca.fit_transform(self.matrix)
        pcs = numpy.transpose(pcs)
        plt.figure(figsize = (8,8))
        data = {"x":pcs[0],"y":pcs[1], "Sample": self.sample, "Cluster": clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Sample')
        plt.savefig(png)

    def plot_umap(self, png):
        clusters = self.kmeans()
        pca = umap.UMAP(n_components=2)
        pcs = pca.fit_transform(self.matrix)
        pcs = numpy.transpose(pcs)
        plt.figure(figsize = (8,8))
        data = {"x":pcs[0],"y":pcs[1], "Sample": self.sample, "Cluster": clusters}
        df = pandas.DataFrame.from_dict(data)
        sns.scatterplot(data=df,x="x", y="y", hue='Sample')
        plt.savefig(png)

def main():
    vector = pickle.load(open("run6/W2.p","rb")).tolist()
    context = genevec.Context.load("cancer.p")
    embed = GeneEmbedding(vector, context)
    clusters = embed.kmeans()
    cembed = CellEmbedding(context, embed)
    cembed.plot_tsne("celltype.png")
    cluster_genes = cembed.compute_similarities()
    all_genes = set()
    sets = []
    for cluster, genes in cluster_genes.items():
        for gene in genes:
            all_genes.add(gene)
        sets.append(set(genes))
    common_genes = set.intersection(*sets)
    for cluster, genes in cluster_genes.items():
        print(cluster, set(genes).difference(common_genes))
    embed.plot_tsne("tsne.png", all_genes, clusters)
    embed.plot_umap("umap.png", all_genes, clusters)
    # similarities = embed.compute_similarities()
    # for gene, distances in similarities.items():
    #     sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
    #     print(gene, sorted_distances[:5])

if __name__ == '__main__':
    main()
