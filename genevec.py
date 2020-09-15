from singlecellexperiment import SingleCellExperiment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

import operator
from itertools import permutations
import argparse
import tqdm
import random
import pickle
import collections
import sys
import os

import embedding


class Context(object):

    def __init__(self):
        pass

    @classmethod
    def build(context_class, rdata, lower_bound=5.0):
        context = context_class()
        context.rdata = rdata
        context.sce = SingleCellExperiment.fromRData(context.rdata)
        context.genes = [x.upper() for x in context.sce.rownames]
        context.matrix = context.sce.assays["counts"]
        context.cells = context.sce.colnames
        context.cell_index, context.index_cell = Context.index_cells(context.sce.colnames)
        context.data, context.cell_to_gene = context.expression(context.matrix, \
                                                                  context.genes, \
                                                                  context.index_cell, \
                                                                  lower_bound=lower_bound)
        context.expressed_genes = context.get_expressed_genes(context.data)
        context.gene_index, context.index_gene = Context.index_geneset(context.expressed_genes)
        context.coexpressed = context.coexpression(context.data, \
                                                   context.gene_index)
        context.lower_bound = lower_bound
        context.metadata = context.sce.colData
        return context

    @classmethod
    def load(context_class, path):
        context = context_class()
        serialized = pickle.load(open(path, "rb"))
        context.unserialize(serialized)
        return context

    @staticmethod
    def index_geneset(genes):
        gene_index = {w: idx for (idx, w) in enumerate(genes)}
        index_gene = {idx: w for (idx, w) in enumerate(genes)}
        return gene_index, index_gene

    @staticmethod
    def index_cells(cells):
        cell_index = {w: idx for (idx, w) in enumerate(cells)}
        index_cell = {idx: w for (idx, w) in enumerate(cells)}
        return cell_index, index_cell

    def get_expressed_genes(self, data):
        genes = list()
        remove = collections.defaultdict(list)
        for cellid, cell in data.items():
            for expr, expr_genes in cell.items():
                if len(expr_genes) > 1:
                    genes += expr_genes
                else:
                    remove[cellid].append(expr)
        for cellid, expr in remove.items():
            for val in expr:
                data[cellid].pop(val)
        return list(set(genes))

    def inverse_filter(self, data):
        cell_to_gene = collections.defaultdict(list)
        inv_data = collections.defaultdict(lambda : collections.defaultdict(list))
        for gene, cells in data.items():
            num_cells = len(list(cells.keys()))
            if num_cells >= 50 and num_cells < 1000:
                for cell, val in cells.items():
                    inv_data[cell][str(val)].append(gene)
                    cell_to_gene[cell].append(gene)
        return inv_data, cell_to_gene

    @staticmethod
    def filter_gene(symbol):
        if symbol.startswith("RP") \
           or symbol.startswith("MT-") \
           or "." in symbol \
           or "rik" in symbol.lower() \
           or "linc" in symbol.lower() \
           or "orf" in symbol.lower():
            return True
        return False

    def expression(self, matrix, genes, cells, lower_bound=3):
        gene_index, index_gene = Context.index_geneset(genes)
        nonzero = matrix.nonzero()
        keep = numpy.where(matrix.data > lower_bound)[0]
        n_keep = len(keep)
        nonzero_mask = csr_matrix((numpy.ones(n_keep), (nonzero[0][keep], nonzero[1][keep])), shape=matrix.shape)
        data = collections.defaultdict(dict)
        nonzero = nonzero_mask.nonzero()
        entries = list(zip(nonzero[0],nonzero[1]))
        for gene, cell in tqdm.tqdm(entries):
            symbol = index_gene[gene]
            if not Context.filter_gene(symbol):
                barcode = cells[cell]
                data[symbol][barcode] = int(matrix[gene,cell])#round(matrix[gene,cell],3)
        inv_data = self.inverse_filter(data)
        return inv_data

    def coexpression(self, inv_data, gene_index):
        idx_pairs = list()
        counts = collections.defaultdict(int)
        for cell, genes in tqdm.tqdm(inv_data.items()):
            for expression, genes in genes.items():
                for pair in permutations(genes,2):
                    if counts[pair] > 5 and counts[pair] < 50:
                        try:
                            idx_gene = gene_index[pair[0]]
                            idx_target = gene_index[pair[1]]
                        except KeyError as e:
                            continue
                        idx_pairs.append((idx_gene, idx_target))
                    counts[pair] += 1
        print(len(idx_pairs))
        return numpy.array(idx_pairs)

    def serialize(self):
        serialized = dict()
        for attr, value in self.__dict__.items():
            if attr != "sce" and attr != "inv_data" and attr != "data":
                serialized[attr] = value
        return serialized

    def unserialize(self, serialized):
        for attribute, value in serialized.items():
            setattr(self, attribute, value)

    def save(self, filename):
        serialized = self.serialize()
        pickle.dump(serialized, open(filename,"wb"))

    def __eq__(self, other):
        if set(self.expressed_genes) == set(other.expressed_genes) \
           and (self.coexpressed == other.coexpressed).all() \
           and set(self.cells) == set(other.cells):
           return True
        return False

class Network(object):

    def __init__(self, context, output, load=False, embedding_dims=30):
        self.embedding_dims = embedding_dims
        self.context = context
        self.input_dimension = len(self.context.expressed_genes)
        self.output = output
        if not os.path.exists(output):
            os.makedirs(output)
        self.W1_path = os.path.join(output, "W1.p")
        self.W2_path = os.path.join(output, "W2.p")
        self.context_path = os.path.join(output, "context.p")
        if os.path.exists(self.W1_path) and os.path.exists(self.W2_path) and load:
            self.load_weights()
        self.losses = []

    def get_one_hot(self, word_idx):
        x = torch.zeros(self.input_dimension).float()
        x[word_idx] = 1.0
        return x

    def train_network(self, learning_rate=0.1, epochs=50, decay=0.01, tol= 0.0001, reload=False):
        if reload:
            W1, W2 = self.load_weights()
        else:
            W1 = Variable(torch.randn(self.embedding_dims, len(self.context.expressed_genes)).float(), requires_grad=True)
            W2 = Variable(torch.randn(len(self.context.expressed_genes), self.embedding_dims).float(), requires_grad=True)
        print("Starting Training...")
        for epo in range(epochs):
            loss_val = 0
            for signal, target in tqdm.tqdm(self.context.coexpressed):
                x = Variable(self.get_one_hot(signal))
                y_true = Variable(torch.from_numpy(numpy.array([target])).long())
                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)
                log_softmax = F.log_softmax(z2, dim=0)
                loss = F.nll_loss(log_softmax.view(1,-1), y_true)
                loss_val += loss.data.item()
                loss.backward()
                W1.data -= learning_rate * W1.grad.data
                W2.data -= learning_rate * W2.grad.data
                W1.grad.data.zero_()
                W2.grad.data.zero_()
            learning_rate = learning_rate * (1.0-decay)
            print(f'Loss at epo {epo}: {loss_val/len(self.context.coexpressed)}')
            self.losses.append(loss_val/len(self.context.coexpressed))
            self.save_weights(W1, W2)
            if len(self.losses) > 3 and abs(self.losses[-2] - self.losses[-1]) < tol:
                break
        return W1, W2

    def load_weights(self):
        W1 = pickle.load( open( self.W1_path, "rb" ) )
        W2 = pickle.load( open( self.W2_path, "rb" ) )
        return W1, W2

    def plot_loss(self, filename='loss.png'):
        plt.figure()
        plt.plot(range(len(self.losses)), self.losses, label="Negative Log likelihood Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output,filename))

    def save_weights(self, W1, W2):
        pickle.dump(W1, open(self.W1_path, "wb"))
        pickle.dump(W2, open(self.W2_path, "wb"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdata", type=str, help='SingleCellExperiment rds file.')
    parser.add_argument("--context", help='Context filename.')
    parser.add_argument("--output", help="Output folder")

    args = parser.parse_args()
    context = Context.build(args.rdata)
    context.save(args.context)
    model = Network(context, args.output)
    _, vector = model.train_network(reload=False)
    vector = vector.tolist()
    embed = embedding.GeneEmbedding(vector, context)
    clusters = embed.kmeans()



    embed = GeneEmbedding(vector, context)
    clusters = embed.kmeans()
    cembed = CellEmbedding(context, embed)
    cembed.plot_tsne("celltype.png")
    cluster_genes = cembed.compute_similarities()
    all_genes = list()
    sets = []
    for cluster, genes in cluster_genes.items():
        for gene in genes:
            all_genes.add(gene)
        sets.append(set(genes))

    common_genes = set.intersection(*sets)
    for cluster, genes in cluster_genes.items():
        print(cluster, set(genes).difference(common_genes))
    embed.plot_tsne(os.path.join(args.output, "tsne.png"), all_genes, clusters)
    embed.plot_umap(os.path.join(args.output, "umap.png"), all_genes, clusters)

if __name__ == '__main__':
    main()
