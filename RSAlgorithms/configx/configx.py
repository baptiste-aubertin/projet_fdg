# encoding:utf-8
class ConfigX():
    """
    docstring for ConfigX

    configurate the global parameters and hyper parameters

    """

    def __init__(self):
        # Dataset Parameters
        self.dataset_name = "ep10"  # short name of datasets ["ft":"filmtrust","db":"douban","ca":"ciao", "ep":"epinions",","dls":"delicious"]
        self.k_fold_num = 5  # the num of cross validation
        self.rating_path = "./RSAlgorithms/data/%s_ratings.txt" % self.dataset_name  # the raw ratings data file
        self.rating_cv_path = "./RSAlgorithms/data/cv/"  # the cross validation file of ratings data
        self.trust_path = './RSAlgorithms/data/%s_trust.txt' % self.dataset_name  # the raw trust data file
        self.sep = ' '  # the separator of rating and trust data in triple tuple
        self.random_state = 0  # the seed of random number
        self.size = 0.90  # the ratio of train set
        self.min_val = 0.5  # the minimum rating value
        self.max_val = 4.0  # the maximum rating value

        # Model HyperParameter
        self.coldUserRating = 5  # the number of ratings a cold start user rated on items
        self.factor = 10  # the size of latent dimension for user and item.
        self.threshold = 1e-4  # the threshold value of model training 
        self.lr = 0.01  # the learning rate
        self.maxIter = 100  # the maximum number of iterations
        self.lambdaP = 0.001  # the parameter of user regularizer
        self.lambdaQ = 0.001  # the parameter of item regularizer
        self.beta = 0.01
        self.gamma = 0  # momentum coefficient
        self.isEarlyStopping = False  # early stopping flag

        # Output Parameters
        self.result_path = "./RSAlgorithms/results/"  # the directory of results
        self.model_path = "./RSAlgorithms/model/"  # the directory of well-trained variables
        self.result_log_path = "log/"  # the directory of logs when training models
