from singlecellexperiment import SingleCellExperiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy
import operator
import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import collections
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import networkx as nx
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
import sys
import os
from scipy.sparse import csr_matrix
import seaborn as sns
import pandas
import matplotlib.cm as cm


output = sys.argv[1]
try:
    os.makedirs(output)
except Exception as e:
    pass
sce = sys.argv[2]

def setup_context(rdata):
    sce = SingleCellExperiment.fromRData(rdata)
    genes = sce.rownames
    genes = [x.upper() for x in genes]
    cells = sce.assays["counts"]
    nonzero = cells.nonzero()
    keep = numpy.where(cells.data > 7)[0]
    n_keep = len(keep)
    _cells = csr_matrix((numpy.ones(n_keep), (nonzero[0][keep], nonzero[1][keep])), shape=cells.shape)

    gene_index = {w: idx for (idx, w) in enumerate(genes)}
    index_gene = {idx: w for (idx, w) in enumerate(genes)}

    data = collections.defaultdict(dict)
    nonzero = _cells.nonzero()
    for gene, cell in tqdm.tqdm(list(zip(nonzero[0],nonzero[1]))):
        if index_gene[gene].startswith("RP") or index_gene[gene].startswith("MT-") or "." in index_gene[gene] or "rik" in index_gene[gene].lower() or "linc" in index_gene[gene]: continue
        data[index_gene[gene]][cell] = cells[gene,cell]

    remove = []
    for gene, cell in data.items():
        if len(list(cell.keys())) > 300:
            remove.append(gene)

    for gene in remove:
        data.pop(gene)

    idx_pairs = set()
    width = cells.shape[1]
    expressed_genes = list(data.keys())

    gene_reindex = {w: idx for (idx, w) in enumerate(expressed_genes)}
    reindex_gene = {idx: w for (idx, w) in enumerate(expressed_genes)}

    for gene_target in tqdm.tqdm(expressed_genes):
        for gene_context in expressed_genes:
            if gene_target == gene_context: continue
            for cell, count in data[gene_target].items():
                if cell in data[gene_context] and data[gene_context][cell] == data[gene_target][cell]:
                    idx_target = gene_reindex[gene_target]
                    idx_context = gene_reindex[gene_context]
                    if (idx_context, idx_target) not in idx_pairs:
                        idx_pairs.add((idx_target, idx_context))


    idx_pairs = numpy.array(list(idx_pairs))
    print(len(idx_pairs))
    return idx_pairs, data, width, gene_reindex, reindex_gene, expressed_genes

def get_input_layer(word_idx):
    x = torch.zeros(len(genes)).float()
    x[word_idx] = 1.0
    return x

def get_output_layer(word_idx):
    x = torch.zeros(len(genes)).long()
    x[word_idx] = 1.0
    return x

def train_network(idx_pairs, epochs=20, embedding_dims=50, reload=False, save=True):
    global output
    W1_path = os.path.join(output, "W1.p")
    W2_path = os.path.join(output, "W2.p")
    if reload:
        W1 = pickle.load(open(W1_path, "rb"))
        W2 = pickle.load(open(W2_path, "rb"))
    else:
        W1 = Variable(torch.randn(embedding_dims, len(genes)).float(), requires_grad=True)
        W2 = Variable(torch.randn(len(genes), embedding_dims).float(), requires_grad=True)
    num_epochs = epochs
    learning_rate = 0.1
    losses = []
    print("Starting Training...")
    for epo in range(num_epochs):
        loss_val = 0
        for signal, target in tqdm.tqdm(idx_pairs):
            x = Variable(get_input_layer(signal))
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
        losses.append(loss_val/len(idx_pairs))
        if save:
            pickle.dump( W1, open( W1_path, "wb" ) )
            pickle.dump( W2, open( W2_path, "wb" ) )
        if len(losses) > 3 and abs(losses[-2] - losses[-1]) < 0.01:
            break
    return W1, W2, losses

def load_weights():
    global output
    W1_path = os.path.join(output, "W1.p")
    W2_path = os.path.join(output, "W2.p")
    W1 = pickle.load( open( W1_path, "rb" ) )
    W2 = pickle.load( open( W2_path, "rb" ) )
    return W1, W2

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

def build_cell_embedding(rdata, i_gene_index, i_index_gene, vector, genes):
    original_vec = vector
    rows = open("celltypes.csv","r").read().splitlines()
    celltypes = []
    barcodes = []
    for row in rows[1:]:
        row = row.replace('"',"").split(",")
        celltypes.append(row[2])
    sce = SingleCellExperiment.fromRData(rdata)
    genes = sce.rownames
    genes = [x.upper() for x in genes]
    cells = sce.assays["counts"]
    nonzero = cells.nonzero()
    keep = numpy.where(cells.data > 7)[0]
    n_keep = len(keep)
    _cells = csr_matrix((numpy.ones(n_keep), (nonzero[0][keep], nonzero[1][keep])), shape=cells.shape)

    gene_index = {w: idx for (idx, w) in enumerate(genes)}
    index_gene = {idx: w for (idx, w) in enumerate(genes)}

    data = collections.defaultdict(list)
    nonzero = _cells.nonzero()
    for gene, cell in tqdm.tqdm(list(zip(nonzero[0],nonzero[1]))):
        if index_gene[gene].startswith("RP") or index_gene[gene].startswith("MT-") or "." in index_gene[gene] or "rik" in index_gene[gene].lower() or "linc" in index_gene[gene]: continue
        gene = index_gene[gene]
        try:
            index = i_gene_index[gene]
            data[cell].append(vector[index])
        except Exception as e:
            continue
    print(len(data.keys()))
    matrix = []
    celltype = []
    for cell, vectors in data.items():
        xvec = list(numpy.average(vectors, axis=0))
        matrix.append(xvec)
        print(len(list(xvec)))
        celltype.append(celltypes[cell])
    matrix = numpy.array(matrix)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(matrix)
    clusters = ["C"+str(x) for x in kmeans.labels_]
    cluster_labels = collections.defaultdict(list)
    for clust, vec in zip(clusters, matrix):
        cluster_labels[clust].append(vec)
    for clust, vectors in cluster_labels.items():
        xvec = list(numpy.average(vectors, axis=0))
        distances = dict()
        for target in genes:
            if target not in i_gene_index: continue
            v = original_vec[i_gene_index[target]]
            distance = float(cosine_similarity(numpy.array(xvec).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
            distances[target] = distance
        sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
        print(clust, sorted_distances[:20])
    # for cluster,
    print(matrix.shape)
    pca = TSNE(n_components=2)
    pcs = pca.fit_transform(matrix)
    pcs = numpy.transpose(pcs)
    plt.figure(figsize = (8,8))
    #plt.scatter(pcs[0],pcs[1])
    data = {"x":pcs[0],"y":pcs[1], "Cell Type": clusters}
    df = pandas.DataFrame.from_dict(data)
    sns.scatterplot(data=df,x="x", y="y", hue='Cell Type')
    plt.savefig("cell_embedding.png")

    _genes = sce.rownames
    cells = sce.assays["logcounts"]
    cells = cells.todense()
    pca = PCA(n_components=50)
    pcs = pca.fit_transform(cells)
    print(pcs.shape)
    # pcs = numpy.transpose(pcs)
    tsne = TSNE(n_components=2)
    tss = tsne.fit_transform(pcs)
    tss = numpy.transpose(tss)
    print(tss.shape)
    print(len(tss[0]))
    input = open("tcell.txt","r").read().splitlines()
    tcell = []
    for x in input:
        if "-" in x:
            tcell.append(x.split("-")[1].strip())
    markers = []
    print(tcell)
    for genename in _genes:
        if genename.upper() in ["CD8B","CD8A","GNLY"]+tcell:
            print(genename)
            markers.append(genename)
        else:
            markers.append("Other")
    plt.figure()
    print(len(markers))
    print(len(tss[0]))
    data = {"x":tss[0],"y":tss[1], "Gene": markers}
    df = pandas.DataFrame.from_dict(data)
    sns.scatterplot(data=df,x="x", y="y", hue='Gene')
    plt.savefig("transposed_tsne.png")
    # remove = []
    # for gene, cell in data.items():
    #     if len(list(cell.keys())) > 300:
    #         remove.append(gene)
    #
    # for gene in remove:
    #     data.pop(gene)
    #
    # idx_pairs = set()
    # width = cells.shape[1]
    # expressed_genes = list(data.keys())
    #
    # gene_reindex = {w: idx for (idx, w) in enumerate(expressed_genes)}
    # reindex_gene = {idx: w for (idx, w) in enumerate(expressed_genes)}



def plot_distance_vs_expression(genea, rdata, gene_index, index_gene, similarities):
    goi = [x.split(",")[0] for x in open("genes.txt","r").read().splitlines()]
    sce = SingleCellExperiment.fromRData(rdata)
    genes = sce.rownames
    genes = [x.upper() for x in genes]
    cells = sce.assays["logcounts"]
    nonzero = cells.nonzero()
    # keep = numpy.where(cells.data > 7)[0]
    # n_keep = len(keep)
    # _cells = csr_matrix((numpy.ones(n_keep), (nonzero[0][keep], nonzero[1][keep])), shape=cells.shape)

    gene_index = {w: idx for (idx, w) in enumerate(genes)}
    index_gene = {idx: w for (idx, w) in enumerate(genes)}

    data = collections.defaultdict(dict)
    # nonzero = _cells.nonzero()
    for gene, cell in tqdm.tqdm(list(zip(nonzero[0],nonzero[1]))):
        if index_gene[gene].startswith("RP") or index_gene[gene].startswith("MT-") or "." in index_gene[gene] or "rik" in index_gene[gene].lower() or "linc" in index_gene[gene]: continue
        data[index_gene[gene]][cell] = cells[gene,cell]

    exprs = []
    dist = []
    genes = list(data.keys())
    for genea in goi:
        for gene in genes:
            cells = list(data[gene].keys())
            expression = []
            for cell in set(cells).intersection(set(list(data[genea].keys()))):
                try:
                    # if data[genea][cell] - data[gene][cell] > 100:
                    #     print(data[genea][cell], data[gene][cell])
                    exp = data[genea][cell] - data[gene][cell]
                    expression.append(abs(exp))
                except Exception as e:
                    continue
            try:
                dist.append(similarities[genea][gene])
                exprs.append(numpy.mean(expression))
            except Exception as e:
                continue
    print(max(exprs), min(exprs))
    plt.figure()
    plt.scatter(dist,exprs)
    plt.savefig("cd8a.png")

def plot_pca(vector, genes):
    input = open("tcell.txt","r").read().splitlines()
    tcell = []
    for x in input:
        if "-" in x:
            tcell.append(x.split("-")[1].strip())
    input = open("mac.txt","r").read().splitlines()
    macrophage = []
    for x in input:
        if "-" in x:
            macrophage.append(x.split("-")[1].strip())
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(vector)
    clusters = zip(genes,kmeans.labels_)
    _clusters = []
    marker = []
    mapping = collections.defaultdict(list)
    for gene, cluster in clusters:
        _clusters.append("C"+str(cluster))
        if (gene in tcell and gene not in macrophage) or gene == "CD8A" or gene == "CD8B":
            marker.append("T-cell")
            print("Tcell cluster", cluster, gene)
        elif gene in macrophage and gene not in tcell:
            marker.append("Macrophage")
            print("Mac cluster", cluster, gene)
        elif gene in macrophage and gene in tcell:
            marker.append("Macrophage - T-cell")
        else:
            marker.append("Other")
    pca = TSNE(n_components=2)
    pcs = pca.fit_transform(vector)
    pcs = numpy.transpose(pcs)
    plt.figure(figsize = (8,8))
    #plt.scatter(pcs[0],pcs[1])
    data = {"x":pcs[0],"y":pcs[1], "Cluster":_clusters, "Marker": marker}
    df = pandas.DataFrame.from_dict(data)
    print(set(df["Cluster"].tolist()))
    sns.scatterplot(data=df,x="x", y="y",hue="Cluster")
    plt.savefig("tsne.png")

def plot_loss(losses):
    global output
    plt.figure()
    plt.plot(range(len(losses)),losses, label="Negative Log likelihood Loss")
    plt.legend()
    plt.savefig(os.path.join(output,"loss.png"))

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

losses = []

losses_path = os.path.join(output,"losses.p")
similarities_path = os.path.join(output,"similarities.p")
pairs_path = os.path.join(output,"pairs.p")
genes_path = os.path.join(output, "genes.p")
gene_index_path = os.path.join(output, "index.p")
index_gene_path = os.path.join(output, "index_gene.p")

do_load_context = True
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
"""
for epoch in range(3):
    if epoch == 0:
        reload = do_load_context
    else:
        reload = True
    W1, W2, local_loss = train_network(idx_pairs, epochs=5, reload=reload)
    losses += local_loss
    pickle.dump( losses, open(losses_path, "wb" ) )
    plot_loss(losses)
    if len(losses) > 3 and abs(losses[-2] - losses[-1]) < 0.0001:
        break
"""
W1, W2 = load_weights()
vector = W2.tolist()
plot_pca(vector, genes)
# similarities = compute_similarities(genes, vector, gene_index)
build_cell_embedding("sce2.rdata", gene_index, index_gene, vector, genes)
# plot_distance_vs_expression("CD8A", "sce2.rdata", gene_index, index_gene, similarities)
while True:
    gene = input()
    if gene in similarities:
        print(gene, similarities[gene][:30])
        print(gene, similarities[gene][-30:])
        print(gene, [x[0] for x in similarities[gene][-30:]])
        print(gene, [x[0] for x in similarities[gene][:30]])
        print(gene, numpy.mean([x[1] for x in similarities[gene]]))
        print(vector[gene_index[gene]])
    else:
        print("No dice.")
