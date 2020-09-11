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
import tqdm
import random
import pickle
import collections
import sys
import os

output = sys.argv[1]
try:
    os.makedirs(output)
except Exception as e:
    pass
sce = sys.argv[2]


class Context(object):

    def __init__(self, sce, lower_bound=5, max_cells_with_expression=500):
        self.sce = SingleCellExperiment.fromRData(rdata)
        self.genes = [x.upper() for x in sce.rownames]
        self.lower_bound = lower_bound
        cells = sce.assays["logcounts"]
        nonzero = cells.nonzero()
        keep = numpy.where(cells.data > lower_bound)[0]
        n_keep = len(keep)
        self.nonzero_mask = csr_matrix((numpy.ones(n_keep), (nonzero[0][keep], nonzero[1][keep])), shape=cells.shape)
        self.gene_index, self.index_gene = Context.index_geneset(self.genes)
        self.data = self.expression()
        self.data = self.filter_ubiqitous(self.data, max_cells_with_expression)
        self.expressed_genes = list(data.keys())
        self.gene_reindex, self.reindex_gene = self.index_geneset(expressed_genes)
        self.idx_pairs = compute_idx_pairs(self.data, self.expressed_genes, self.gene_reindex, self.reindex_gene)

    def index_geneset(genes):
        gene_index = {w: idx for (idx, w) in enumerate(genes)}
        index_gene = {idx: w for (idx, w) in enumerate(genes)}
        return gene_index, index_gene

    def filter_ubiqitous(self, data, max_cells_with_expression):
        remove = []
        for gene, cell in data.items():
            if len(list(cell.keys())) > max_cells_with_expression:
                remove.append(gene)
        for gene in remove:
            data.pop(gene)
        return data

    @staticmethod
    def filter_gene(gene):
        if self.index_gene[gene].startswith("RP") \\
           or self.index_gene[gene].startswith("MT-") \\
           or "." in self.index_gene[gene] \\
           or "rik" in self.index_gene[gene].lower() \\
           or "linc" in self.index_gene[gene].lower() \\
           or "orf" in self.index_gene[gene].lower():
            return True
        return False

    def expression(self, index_gene):
        data = collections.defaultdict(dict)
        nonzero = self.nonzero_mask.nonzero()
        for gene, cell in tqdm.tqdm(list(zip(nonzero[0],nonzero[1]))):
            if Context.filter_gene(gene): continue
            data[index_gene[gene]][cell] = cells[gene,cell]
        return data

    def compute_idx_pairs(self, data, expressed_genes, gene_reindex, reindex_gene):
        idx_pairs = list()
        for gene_target in tqdm.tqdm(expressed_genes):
            if gene_target not in expressed_genes: continue
            for gene_context in expressed_genes:
                if gene_target == gene_context: continue
                for cell, count in data[gene_target].items():
                    if cell in data[gene_context] and data[gene_context][cell] == data[gene_target][cell]:
                        idx_target = gene_reindex[gene_target]
                        idx_context = gene_reindex[gene_context]
                        idx_pairs.append((idx_target, idx_context))
        return numpy.array(idx_pairs)

    def save(self):
        pass

    @classmethod
    def load(context_class, path):
        self.idx_pairs = pickle.load(open(pairs_path, "rb"))
        self.index_gene = pickle.load(open(index_gene_path, "rb"))
        self.gene_index = pickle.load(open(gene_index_path, "rb"))
        self.genes = pickle.load(open(genes_path, "rb"))


class Network(object):

    def __init__(self, context, output, embedding_dims=30):
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.context = context
        self.output = output
        self.W1_path = os.path.join(output, "W1.p")
        self.W2_path = os.path.join(output, "W2.p")
        self.losses_path = os.path.join(output,"losses.p")
        self.losses = []

    @staticmethod
    def get_input_layer(word_idx):
        x = torch.zeros(len(genes)).float()
        x[word_idx] = 1.0
        return x

    @staticmethod
    def get_output_layer(word_idx):
        x = torch.zeros(len(genes)).long()
        x[word_idx] = 1.0
        return x

    def train_network(self, learning_rate=0.1, epochs=20, reload=False, save=True):
        if reload:
            W1 = pickle.load(open(self.W1_path, "rb"))
            W2 = pickle.load(open(self.W2_path, "rb"))
        else:
            W1 = Variable(torch.randn(self.embedding_dims, len(self.context.expressed_genes)).float(), requires_grad=True)
            W2 = Variable(torch.randn(len(self.context.expressed_genes), self.embedding_dims).float(), requires_grad=True)
        print("Starting Training...")
        for epo in range(epochs):
            loss_val = 0
            for signal, target in tqdm.tqdm(self.context.idx_pairs):
                x = Variable(Network.get_input_layer(signal))
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
            print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
            self.losses.append(loss_val/len(idx_pairs))
            if save:
                pickle.dump( W1, open( self.W1_path, "wb" ) )
                pickle.dump( W2, open( self.W2_path, "wb" ) )
            if len(self.losses) > 3 and abs(self.losses[-2] - self.losses[-1]) < 0.0001:
                break

    def load_weights(self):
        W1 = pickle.load( open( self.W1_path, "rb" ) )
        W2 = pickle.load( open( self.W2_path, "rb" ) )
        return W1, W2

    def plot_loss(self, filename = 'loss.png'):
        plt.figure()
        plt.plot(range(len(self.losses)),self.losses, label="Negative Log likelihood Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output,filename))

    def save(self):
        pickle.dump( losses, open(losses_path, "wb" ) )
        pickle.dump( W1, open( self.W1_path, "wb" ) )
        pickle.dump( W2, open( self.W2_path, "wb" ) )
        self.context.save(self.output)

    def load(self):
        self.context = Context.load(self.output)


class GeneEmbedding(object):
    def __init__(self):
        pass

    def compute_similarities(genes, vector, gene_index):
        similarities = dict()
        goi = [x.split(",")[0] for x in open("genes.txt","r").read().splitlines()]
        downstream = []
        for gene in tqdm.tqdm(goi):
            if gene not in genes: continue
            embedding = vector[gene_index[gene]]
            distances = dict()
            for target in genes:
                if target not in gene_index: continue
                v = vector[gene_index[target]]
                distance = float(cosine_similarity(numpy.array(embedding).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
                distances[target] = distance
            sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
            similarities[gene] = sorted_distances
        return similarities

    def plot_pca(vector, genes):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(vector)
        clusters = zip(genes,kmeans.labels_)
        for gene, cluster in clusters:
            if gene == "CD8A":
                print(cluster)
            if gene == "TYROBP":
                print(cluster)
        pca = TSNE(n_components=2)
        pcs = pca.fit_transform(vector)
        pcs = numpy.transpose(pcs)
        plt.figure(figsize = (8,8))
        plt.scatter(pcs[0],pcs[1])
        plt.savefig("tsne2.png")



    def load_context():
        # losses_path = os.path.join(output,"losses.p")
        # similarities_path = os.path.join(output,"similarities.p")
        pairs_path = os.path.join(output,"pairs.p")
        genes_path = os.path.join(output, "genes.p")
        gene_index_path = os.path.join(output, "index.p")
        index_gene_path = os.path.join(output, "index_gene.p")
        # losses = pickle.load(open(losses_path, "rb"))
        # similarities = pickle.load(open(similarities_path, "rb"))
        idx_pairs = pickle.load(open(pairs_path, "rb"))
        index_gene = pickle.load(open(index_gene_path, "rb"))
        gene_index = pickle.load(open(gene_index_path, "rb"))
        genes = pickle.load(open(genes_path, "rb"))
        return genes, idx_pairs, index_gene, gene_index




do_load_context = False
if do_load_context:
    print("Loading context...")
    genes, idx_pairs, index_gene, gene_index = load_context()
else:
    print("Setting up context...")
    idx_pairs, data, width, gene_index, index_gene, genes = setup_context(sce)
    pickle.dump(idx_pairs, open(pairs_path,"wb"))
    pickle.dump(genes, open(genes_path,"wb"))
    pickle.dump(gene_index, open(gene_index_path,"wb"))
    pickle.dump(index_gene, open(index_gene_path,"wb"))

for epoch in range(3):
    if epoch == 0:
        reload = do_load_context
    else:
        reload = True
    W1, W2, local_loss = train_network(idx_pairs, epochs=5, reload=reload)
    losses += local_loss

    plot_loss(losses)
    if len(losses) > 3 and abs(losses[-2] - losses[-1]) < 0.0001:
        break

W1, W2 = load_weights()
vector = W2.tolist()
plot_pca(vector, genes)
similarities = compute_similarities(genes, vector, gene_index)
while True:
    gene = input()
    if gene in similarities:
        print(gene, similarities[gene][:10])
    else:
        print("No dice.")
