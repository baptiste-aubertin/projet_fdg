import numpy as np
from .mf import MF
from ..reader.trust import TrustGetter
from ..utility.matrix import SimMatrix
from ..utility.similarity import pearson_sp, cosine_sp
from node2vec import Node2Vec
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class SocialReg(MF):
    """
    Ma H, Zhou D, Liu C, et al. Recommender systems with social regularization[C]//Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011: 287-296.
    """

    def __init__(self, sim="cosine"):
        super(SocialReg, self).__init__()
        self.config.alpha = 0.1
        self.tg = TrustGetter()
        self.sim_fct = sim
        self.user_sim = SimMatrix()
        self.node_embeddings = self.get_node_embeddings()
    
    # Function to describe graph properties
    def desc_graph(self, g, name):
          print("=============DESCRIPTION GRAPHE "+name+"==========================")
          nb_node = len(g.nodes)
          nb_edges = len(g.edges)
          density = nx.density(g)
          print("NOMBRE DE NOEUDS : "+str(nb_node))
          print("NOMBRE D'ARCS : "+str(nb_edges))
          print("VALEUR DE DENSITÉ DU GRAPHE : "+str(density))
          degree_sequence = sorted((d for n, d in g.degree()), reverse=False)
          dmax = max(degree_sequence)
          print("DEGRÉE MAX DU GRAPHE : "+str(dmax))
          print("DEGRÉE MOYEN DU GRAPHE : "+str(np.sum(degree_sequence)/len(g.nodes)))
          print("PREMIER QUARTILE DES DEGRÉES: "+str(degree_sequence[int(len(degree_sequence)*1/4)]))
          print("DEGRÉE MÉDIAN DU GRAPHE: "+str(degree_sequence[int(len(degree_sequence)/2)]))
          print("TROISIÈME QUARTILE DES DEGRÉES: "+str(degree_sequence[int(len(degree_sequence)*3/4)]))
          nb_max = degree_sequence.count(max(degree_sequence, key=degree_sequence.count))+3
          ax = sns.histplot(degree_sequence, kde=True, color="blue")
          plt.ylim(0, nb_max)
          ax.lines[0].set_color('crimson')
          plt.savefig(name + '_histogram.png')  # Save histogram plot as an image
          plt.close()

          plt.boxplot(degree_sequence)
          plt.savefig(name + '_boxplot.png')  # Save boxplot as an image
          plt.close()

    # Train Node2Vec on the graph and return node embeddings.
    def get_node_embeddings(self) : 
        # access to super attributes
        trust_path = self.config.trust_path

        trust_data = pd.read_csv(trust_path, sep=' ', names=['source', 'target', 'trust'])

        ## remove third column

        trust_data = trust_data.drop(columns=['trust'])

        deli = nx.from_pandas_edgelist(trust_data, source='source', target='target')
        
        # Describe the graph
        self.desc_graph(deli, "Graph for data")

        # Train Node2Vec on the graph
        node2vec = Node2Vec(deli, dimensions=20, walk_length=20, num_walks=20, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Extract node embeddings
        node_embeddings = {str(node): model.wv[str(node)] for node in deli.nodes}
      
        return node_embeddings

    def init_model(self, k):
        super(SocialReg, self).init_model(k)
        self.construct_user_user_similarity()
    
    # construct user-user similarity matrix
    def construct_user_user_similarity(self):
        print('constructing user-user similarity matrix...')
        for u in self.rg.user:
            for f in self.tg.get_followees(u):
                if self.user_sim.contains(u, f):
                    continue
                sim = self.get_sim(u, f)
                self.user_sim.set(u, f, sim)

    def calculate_jaccard_similarity(self, u, k):
        """
        Calculate Jaccard similarity based on common neighbors.
        """
        # Get the set of neighbors for users u and k
        neighbors_u = set(self.tg.get_followees(u))
        neighbors_k = set(self.tg.get_followees(k))

        # Calculate Jaccard similarity
        if len(neighbors_u.union(neighbors_k)) == 0:
            return 0.0
        else:
            return len(neighbors_u.intersection(neighbors_k)) / len(neighbors_u.union(neighbors_k))

   # Calculate similarity based on the selected similarity function
    def get_sim(self, u, k):
        if self.sim_fct == "pearson_sp":
            sim = (pearson_sp(self.rg.get_row(u), self.rg.get_row(k)) + 1.0) / 2.0
        elif self.sim_fct == "nodetovec":
            sim = self.calculate_nodetovec_similarity(u, k)
        elif self.sim_fct == "jaccard":
            sim = self.calculate_jaccard_similarity(u, k)
        else:
            sim = (cosine_sp(self.rg.get_row(u), self.rg.get_row(k)) + 1.0) / 2.0  # fit the value into range [0.0,1.0]
        return sim

   # Calculate similarity using Node2Vec embeddings
    def calculate_nodetovec_similarity(self, u, k):
        emb_u = self.node_embeddings.get(str(u), None)
        emb_k = self.node_embeddings.get(str(k), None)

        if emb_u is not None and emb_k is not None:
          # Calculate cosine similarity between node embeddings
            sim = np.dot(emb_u, emb_k) / (np.linalg.norm(emb_u) * np.linalg.norm(emb_k))
            return sim
        else:
            return 0.0

    # Train the SocialReg model
    def train_model(self, k):
        super(SocialReg, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2
                p, q = self.P[u], self.Q[i]

                social_term_p, social_term_loss = np.zeros((self.config.factor)), 0.0
                followees = self.tg.get_followees(user)
                for followee in followees:
                    if self.rg.containsUser(followee):
                        s = self.user_sim[user][followee]
                        uf = self.P[self.rg.user[followee]]
                        social_term_p += s * (p - uf)
                        social_term_loss += s * ((p - uf).dot(p - uf))

                social_term_m = np.zeros((self.config.factor))
                followers = self.tg.get_followers(user)
                for follower in followers:
                    if self.rg.containsUser(follower):
                        s = self.user_sim[user][follower]
                        ug = self.P[self.rg.user[follower]]
                        social_term_m += s * (p - ug)

                # update latent vectors
                self.P[u] += self.config.lr * (
                        error * q - self.config.alpha * (social_term_p + social_term_m) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                    self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    # srg = SocialReg()
    # srg.train_model(0)
    # coldrmse = srg.predict_model_cold_users()
    # print('cold start user rmse is :' + str(coldrmse))
    # srg.show_rmse()

    rmses = []
    maes = []
    tcsr = SocialReg()
    # print(bmf.rg.trainSet_u[1])
    for i in range(tcsr.config.k_fold_num):
        print('the %dth cross validation training' % i)
        tcsr.train_model(i)
        rmse, mae = tcsr.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
