import numpy as np
from .mf import MF
from ..reader.trust import TrustGetter
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from node2vec import Node2Vec


class ExtendedSocialRec(MF):
    """
    Extension du modèle SocialRec pour inclure de nouvelles mesures de similarité.
    """

    def __init__(self):
        super(ExtendedSocialRec, self).__init__()
        self.config.alpha = 0.1
        self.config.lambdaZ = 0.01
        self.tg = TrustGetter()
        # Ajout pour les nouvelles similarités
        self.user_similarity_graph_based = None
        self.user_similarity_node2vec = None

    def init_model(self, k):
        super(ExtendedSocialRec, self).init_model(k)
        self.Z = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (self.config.factor ** 0.5)
        self.initialize_graph_based_similarity()
        self.initialize_node2vec_similarity()

    def initialize_graph_based_similarity(self):
        # Initialisation de la similarité basée sur le graphe
        G = nx.Graph()
        # Ajout des arêtes basées sur la confiance
        for user in self.tg.followees:
            for followee, weight in self.tg.get_followees(user).items():
                G.add_edge(user, followee, weight=weight)
        # Calcul de la similarité (exemple: Jaccard)
        self.user_similarity_graph_based = list(nx.jaccard_coefficient(G))

    def initialize_node2vec_similarity(self):
        # Initialisation de Node2Vec
        G = nx.Graph()
        for user in self.tg.followees:
            for followee, weight in self.tg.get_followees(user).items():
                G.add_edge(user, followee, weight=weight)
        node2vec = Node2Vec(G, dimensions=20, walk_length=16, num_walks=100, workers=2)
        model = node2vec.fit(window=10, min_count=1)
        # Obtention des vecteurs de chaque utilisateur
        user_vectors = {}
        for user in self.tg.followees:
            try:
                user_vectors[user] = model.wv[user]
            except KeyError:
                # Gérer l'utilisateur manquant, par exemple en utilisant un vecteur aléatoire ou un vecteur moyen
                user_vectors[user] = np.random.rand(model.vector_size)  # Exemple de vecteur aléatoire
        # Calcul de la similarité cosine entre les vecteurs
        self.user_similarity_node2vec = cosine_similarity(list(user_vectors.values()))

    def train_model(self, k):
        super(ExtendedSocialRec, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                # Mise à jour basée sur les relations sociales
                followees = self.tg.get_followees(user)
                zs = np.zeros(self.config.factor)
                for followee in followees:
                    if self.rg.containsUser(user) and self.rg.containsUser(followee):
                        vminus = len(self.tg.get_followers(followee))
                        uplus = len(self.tg.get_followees(user))
                        weight = np.sqrt(vminus / (uplus + vminus + 0.0)) if uplus + vminus > 0 else 1
                        zid = self.rg.user[followee]
                        z = self.Z[zid]
                        err = weight - z.dot(p)
                        self.loss += err ** 2
                        zs += -1.0 * err * p
                        self.Z[zid] += self.config.lr * (self.config.alpha * err * p - self.config.lambdaZ * z)

                # Intégration des mesures de similarité
                        
                sim_users_graph = self.user_similarity_graph_based[u]
                sim_users_node2vec = self.user_similarity_node2vec[u]
                print(sim_users_graph)
                for v, sim_value_graph in sim_users_graph:
                    zs += self.config.beta * sim_value_graph * (self.P[v] - p)
                for v, sim_value_node2vec in enumerate(sim_users_node2vec):
                    zs += self.config.gamma * sim_value_node2vec * (self.P[v] - p)

                # Mise à jour des facteurs latents
                self.P[u] += self.config.lr * (error * q - self.config.alpha * zs - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() + self.config.lambdaZ * (self.Z * self.Z).sum()

            iteration += 1
            if self.isConverged(iteration):
                break