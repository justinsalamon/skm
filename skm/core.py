# CREATED: 6/24/14 2:09 PM by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
import sklearn
import pickle
import simplejson as json
from sklearn.decomposition import PCA


class SKM(object):
    '''
    Class that implements the spherical k-means algorithms, including PCA
    whitening, based on: Coats & Ng, "Learning Feature Representations with
    K-means", 2012.
    '''

    __ARGS__ = ['k', 'variance_explained', 'max_epochs',
                'assignment_change_eps', 'standardize', 'normalize',
                'pca_whiten', 'visualize', 'do_pca']

    __PARAMS__ = ['k', 'variance_explained', 'epoch', 'assignment_change',
                  'max_epochs', 'assignment_change_eps', 'nfeatures',
                  'nsamples', 'D', 'assignment', 'prev_assignment', 'visualize',
                  'initialized', 'standardize', 'normalize', 'mus', 'sigmas',
                  'pca_whiten', 'do_pca']

    # Based on sklearn 0.15.2
    __ARGSPCA__ = ['n_components', 'copy', 'whiten']

    __PARAMSPCA__ = ['components_', 'explained_variance_',
                     'explained_variance_ratio_', 'mean_', 'n_components_',
                     'n_samples_', 'noise_variance_']

    def __init__(self, k=500, variance_explained=0.99, max_epochs=100,
                 assignment_change_eps=0.01, standardize=False, normalize=False,
                 pca_whiten=True, visualize=False, do_pca=True):

        # Initialize parameters
        self.k = k
        self.variance_explained = variance_explained
        self.epoch = 0
        self.assignment_change = np.inf
        self.max_epochs = max_epochs
        self.assignment_change_eps = assignment_change_eps
        self.nfeatures = None
        self.nsamples = None
        self.D = None # centroid dictionary
        self.assignment = None # assignment vector
        self.prev_assignment = None # previous assignment vector
        self.visualize = visualize
        self.initialized = False
        self.standardize = standardize
        self.normalize = normalize
        self.mus = None
        self.sigmas = None
        self.pca_whiten = pca_whiten
        self.do_pca = do_pca

        # Initialize PCA
        self.pca = PCA(n_components=self.variance_explained, copy=False, whiten=self.pca_whiten)


    def _pca_fit_transform(self, X):
        '''
        PCA fit and transform the data
        '''
        data = self.pca.fit_transform(X.T) # transpose for PCA
        return data.T # transpose back


    def _pca_fit(self, X):
        '''
        PCA fit only (don't transform the data)
        '''
        self.pca.fit(X.T)


    def _pca_transform(self, X):
        '''
        PCA transform only (must call fit or fit_transform first)
        '''
        data = self.pca.transform(X.T)
        return data.T


    def _normalize_samples(self, X):
        '''
        Normalize the features of each sample so that their values sum to one
        (might make sense for some audio data)
        '''
        data = sklearn.preprocessing.normalize(X, axis=0, norm='l1')
        return data


    def _standardize_fit(self, X):
        '''
        Compute mean and variance (of each feature) for standardization
        '''
        self.mus = np.mean(X, 1)
        self.sigmas = np.std(X, 1)


    def _standardize_transform(self, X):
        '''
        Standardize input data (assumes standardize_fit already called)
        '''
        data = X.T - self.mus
        data /= self.sigmas
        return data.T


    def _standardize_fit_transform(self, X):
        '''
        Compute means and variances (of each feature) for standardization and
        standardize
        '''
        self._standardize_fit(X)
        data = self._standardize_transform(X)
        return data


    def _init_centroids(self):
        '''
        Initialize centroids randomly from a normal distribution and normalize
        (must call _set_dimensions first)
        '''
        # Sample randomly from normal distribution
        self.D = np.random.normal(size=[self.nfeatures, self.k])
        self._normalize_centroids()
        self.initialized = True


    def _normalize_centroids(self):
        '''
        Normalize centroids to unit length (using l2 norm)
        '''
        self.D = sklearn.preprocessing.normalize(self.D, axis=0, norm='l2')

    # @profile
    def _update_centroids(self, X):
        '''
        Update centroids based on provided sample data X
        '''
        S = np.dot(self.D.T, X)
        # centroid_index = np.argmax(S, 0)
        centroid_index = S.argmax(0) # slightly faster
        s_ij = S[centroid_index, np.arange(self.nsamples)]
        S = np.zeros([self.k, self.nsamples])
        S[centroid_index, np.arange(self.nsamples)] = s_ij
        self.D += np.dot(X, S.T)
        self.prev_assignment = self.assignment
        self.assignment = centroid_index


    def _update_centroids_memsafe(self, X):
        '''
        Update centroids based on provided sample data X.
        Try to minimize memory usage.
        '''
        Dt = self.D.T
        centroid_index = np.zeros(X.shape[1], dtype='int')
        s_ij = np.zeros(X.shape[1])
        for n,x in enumerate(X.T):
            dotprod = np.dot(Dt, x)
            centroid_index[n] = np.argmax(dotprod)
            s_ij[n] = dotprod[centroid_index[n]]

        # S = np.zeros([self.k, self.nsamples])
        # S[centroid_index, np.arange(self.nsamples)] = s_ij
        # self.D += np.dot(X, S.T)
        for n in np.arange(self.k):
            s = np.zeros(X.shape[1])
            s[centroid_index==n] = s_ij[centroid_index==n]
            self.D[:,n] += np.dot(X, s)

        self.prev_assignment = self.assignment
        self.assignment = centroid_index

    # @profile
    def _update_centroids_memsafe_fast(self, X):
        '''
        Update centroids based on provided sample data X.
        Try to minimize memory usage. Use weave for efficiency.
        '''
        Dt = self.D.T
        centroid_index = np.zeros(X.shape[1], dtype='int')
        s_ij = np.zeros(X.shape[1])
        for n,x in enumerate(X.T):
            dotprod = np.dot(Dt, x)
            centroid_index[n] = np.argmax(dotprod)
            s_ij[n] = dotprod[centroid_index[n]]

        # S = np.zeros([self.k, self.nsamples])
        # S[centroid_index, np.arange(self.nsamples)] = s_ij
        # self.D += np.dot(X, S.T)
        S = np.zeros([self.nsamples, self.k])
        S[np.arange(self.nsamples), centroid_index] = s_ij
        self.D += np.dot(X, S)

        # for n in np.arange(self.k):
        #     s = np.zeros(X.shape[1])
        #     s[centroid_index==n] = s_ij[centroid_index==n]
        #     self.D[:,n] += np.dot(X, s)

        # nfeatures = X.shape[0]
        # nsamples = X.shape[1]
        # s = np.zeros(nsamples)
        # k = self.k
        # D = self.D
        # dotproduct_command = r"""
        # for (int n=0; n<k; n++)
        # {
        #     for (int m=0; m<nsamples; m++)
        #     {
        #        if (centroid_index[m]==n)
        #             s[m] = s_ij[m];
        #         else
        #             s[m] = 0;
        #     }
        #
        #     for (int f=0; f<nfeatures; f++)
        #     {
        #         float sum = 0;
        #         for (int i=0; i<nsamples; i++)
        #         {
        #             sum += X[f*nsamples + i] * s[i];
        #         }
        #         D[f*k + n] += sum;
        #     }
        # }
        # """
        # scipy.weave.inline(dotproduct_command, ['k','nsamples','centroid_index','s','s_ij','nfeatures','X','D'])

        self.prev_assignment = self.assignment
        self.assignment = centroid_index


    # def _update_centroids_cuda(self, X):
    #     '''
    #     Update centroids based on provided sample data X using GPU via cuda
    #     '''
    #     # S = np.dot(self.D.T, X)
    #     Xcuda = cm.CUDAMatrix(X)
    #     S = cm.dot(cm.CUDAMatrix(self.D).T, Xcuda)
    #     # centroid_index = S.argmax(0) # slightly faster
    #     centroid_index = S.asarray().argmax(axis=0)
    #     # s_ij = S[centroid_index, np.arange(self.nsamples)]
    #     s_ij = S.asarray()[centroid_index, np.arange(self.nsamples)]
    #     S = np.zeros([self.nsamples, self.k])
    #     # S[centroid_index, np.arange(self.nsamples)] = s_ij
    #     S[np.arange(self.nsamples), centroid_index] = s_ij
    #     self.D += cm.dot(Xcuda, cm.CUDAMatrix(S)).asarray()
    #     self.prev_assignment = self.assignment
    #     self.assignment = centroid_index


    def _init_assignment(self):
        '''
        Initialize assignment of samples to centroids (must call _set_dimensions
        first)
        '''
        self.prev_assignment = np.zeros(self.nsamples) - 1
        self.assignment = None


    def _set_dimensions(self, X):
        '''
        Set dimensions (number of features, number of samples) based on
        dimensions of input data X
        '''
        self.nfeatures, self.nsamples = X.shape


    def _compute_assignment_change(self):
        '''
        Compute the fraction of assignments changed by the latest centroid
        update (value between 0 to 1)
        '''
        self.assignment_change = np.mean(self.assignment != self.prev_assignment)


    def _report_status(self):
        '''
        Print current epoch and assignment change fraction
        '''
        print("EPOCH: {:d} CHANGE: {:.4f}".format(self.epoch, self.assignment_change))


    def fit(self, X, memsafe=False, cuda=False):
        '''
        Fit k centroids to input data X until convergence or max number of
        epochs reached.
        '''

        # Normalize data (per sample)
        if self.normalize:
            X = self._normalize_samples(X)

        # Standardize data (across samples)
        if self.standardize:
            X = self._standardize_fit_transform(X)

        # PCA fit and whiten the data
        if self.do_pca:
            X = self._pca_fit_transform(X)

        # Store dimensions of whitened data
        self._set_dimensions(X)

        # Initialize centroid dictionary
        self._init_centroids()

        # Initialize assignment
        self._init_assignment()

        if self.visualize:
                self._visualize_clusters(X)

        # Iteratively update and normalize centroids
        while self.epoch < self.max_epochs and self.assignment_change > self.assignment_change_eps:
            if memsafe:
                self._update_centroids_memsafe_fast(X)
            # elif cuda:
            #     self._update_centroids_cuda(X)
            else:
                self._update_centroids(X)
            self._normalize_centroids()
            self._compute_assignment_change()
            self.epoch += 1
            # self._report_status()
            if self.visualize:
                self._visualize_clusters(X)


    def fit_minibatch(self, X):
        '''
        Fit k centroids to input data X until convergence or max number of
        epochs reached. Assumes X is a mini-batch from a larger sample set.
        The first batch is used to initialize the algorithm (dimensions).
        '''

        # Normalize data (per sample)
        if self.normalize:
            X = self._normalize_samples(X)

        # If this is the first batch, use it to initialize
        if not self.initialized:
            # Standardize data
            if self.standardize:
                X = self._standardize_fit_transform(X)
            # PCA whiten the data
            X = self._pca_fit_transform(X)
            # Store dimensions of whitened data
            self._set_dimensions(X)
            # Initialize centroid dictionary
            self._init_centroids()
            # Initialize assignment
            self._init_assignment()
        else:
            if self.standardize:
                X = self._standardize_transform(X)
            X = self._pca_transform(X)
            # Reset epochs and assignments
            self.epoch = 0
            self._init_assignment()
            self.assignment_change = np.inf

        # Iteratively update and normalize centroids
        while self.epoch < self.max_epochs and self.assignment_change > self.assignment_change_eps:
            self._update_centroids(X)
            self._normalize_centroids()
            self._compute_assignment_change()
            self.epoch += 1
            self._report_status()
            if self.visualize:
                self._visualize_clusters(X)


    def transform(self, X, rectify=False, nHot=0):
        '''
        Transform samples X (each column is a feature vector) to learned feature
        space
        '''
        # print("DEBUG: entered skm.transform")
        # Normalize data (per sample)
        if self.normalize:
            X = self._normalize_samples(X)

        # Standardize data (across samples)
        if self.standardize:
            X = self._standardize_fit_transform(X)

        # print("DEBUG: skipped normalized/standardize")

        # PCA whiten
        X = self._pca_transform(X)
        # X = np.random.rand(149, 173)
        # print("DEBUG: did PCA")

        # Dot product with learned dictionary
        X = np.dot(X.T, self.D)
        # print("DEBUG: did dot product")

        if rectify:
            X = np.maximum(X, 0)

        # x-hot coding instead of just dot product
        if nHot > 0:
            indices = np.argsort(X)
            for n,x in enumerate(X):
                x[indices[n][0:-nHot]] = 0
                x[indices[n][-nHot:]] = 1

        return X.T


    def save(self, filepath):
        '''
        Save instance of SKM class to given filepath as pickle file
        '''
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
            file.close()


    def load(self, filepath):
        '''
        Load skm instance from disk (saved as pickle file)
        '''
        with open(filepath, 'rb') as file:
            skm = pickle.load(file)
            file.close()

        return skm


    @classmethod
    def load_persistent_npzhack(cls, arg_file, param_file=None):
        """Alternate class constructor, from files.

        Parameters
        ----------
        arg_file : str
            Path to a JSON file of arguments.
        param_file : str, default=None
            Path to a numpy archive (npz) file of parameters, or create a
            similar object without parameters.

        Returns
        -------
        skm : SKM
            The instantiated object.
        """
        args_all = json.load(open(arg_file))
        params_all = np.load(param_file) if param_file else dict()

        skm = cls(**args_all['args'])
        if 'params' in params_all.keys():
            # params = np.asscalar(params_all['params'])
            params = params_all['params'][()]
            for key in params:
                setattr(skm, key, params[key])

        pca = PCA(**args_all['args_pca'])
        if 'params_pca' in params_all.keys():
            # params_pca = np.asscalar(params_all['params_pca'])
            params_pca = params_all['params_pca'][()]
            for key in params_pca:
                setattr(pca, key, params_pca[key])

        skm.pca = pca
        return skm


    @classmethod
    def load_persistent(cls, arg_file, param_file=None):
        """Alternate class constructor, from files.

        Parameters
        ----------
        arg_file : str
            Path to a JSON file of arguments.
        param_file : str, default=None
            Path to a numpy archive (npz) file of parameters, or create a
            similar object without parameters.

        Returns
        -------
        skm : SKM
            The instantiated object.
        """
        json_dict = json.load(open(arg_file))
        npz_dict = np.load(param_file) if param_file else dict()

        # Extract and set all data from JSON file
        skm = cls(**json_dict['args'])
        for key in json_dict['params']:
            setattr(skm, key, json_dict['params'][key])

        pca = PCA(**json_dict['args_pca'])
        for key in json_dict['params_pca']:
            setattr(pca, key, json_dict['params_pca'][key])

        # Extract and set all data from npz file (ndarrays)
        for key in npz_dict.keys():
            if "pca." in key:
                setattr(pca, key.replace("pca.", ""), npz_dict[key])
            else:
                setattr(skm, key, npz_dict[key])

        skm.pca = pca
        return skm


    @property
    def params(self):
        return dict([(k, getattr(self, k)) for k in SKM.__PARAMS__ if hasattr(self, k)])

    @property
    def params_pca(self):
        return dict([(k, getattr(self.pca, k)) for k in SKM.__PARAMSPCA__ if hasattr(self.pca, k)])

    @property
    def args(self):
        return dict([(k, getattr(self, k)) for k in SKM.__ARGS__ if hasattr(self, k)])

    @property
    def args_pca(self):
        return dict([(k, getattr(self.pca, k)) for k in SKM.__ARGSPCA__ if hasattr(self.pca, k)])


    def save_persistent_npzhack(self, arg_file, param_file):
        # save skm arguments and parameters
        with open(arg_file, 'w') as fp:
            d = {}
            d['args'] = self.args
            d['args_pca'] = self.args_pca
            json.dump(d, fp, indent=2)

        p = {}
        p['params'] = self.params
        p['params_pca'] = self.params_pca
        np.savez(param_file, **p)


    def save_persistent(self, arg_file, param_file):

        # Save all nd-arrays into the npz
        # Save all non-ndarray attributes as JSON
        with open(arg_file, 'w') as fp:

            json_dict = {}
            json_dict['args'] = self.args
            json_dict['args_pca'] = self.args_pca

            params_skm = self.params
            params_pca = self.params_pca

            npz_dict = {}

            # find the ndarrays in skm and store them separately
            for k in params_skm.keys():
                if type(params_skm[k]) is np.ndarray:
                    npz_dict[k] = params_skm[k]
                    params_skm.pop(k)

            # store all remaining non-ndarray attributes as json
            json_dict['params'] = params_skm

            # find the ndarrays in pca and store them separately
            for k in params_pca.keys():
                if type(params_pca[k]) is np.ndarray:
                    npz_dict['pca.'+k] = params_pca[k]
                    params_pca.pop(k)

            # store all remaining non-ndarray attributes as json
            json_dict['params_pca'] = params_pca


            # save json_dict to disk
            # fix for json floats
            json_dict['params_pca']['noise_variance_'] = round(float(json_dict['params_pca']['noise_variance_']), 15)
            # print json_dict
            json.dump(json_dict, fp, indent=2)

            # save all ndararys (npz_dict) to disk
            np.savez(param_file, **npz_dict)





def spherical_kmeans_viz(X_raw, k):
    '''
    Given data input X (each column is a datapoint, number of rows if
    dimensionality of the data), and desired number of means k, compute the
    means (dictionary D) and cluster assignment S using the spherical k-means
    algorithm (cf. Coats & NG, 2012)

    Data is assumed to be normalized.
    '''

    # Step 1: whiten inputs using PCA
    n_components = 0.99 # explain 99% of the variance
    pca = PCA(n_components=n_components, copy=True, whiten=True)
    X = pca.fit_transform(X_raw.T)
    X = X.T
    # X = X_raw

    dim, N = X.shape
    # print X.shape

    # Step 2: k-means
    # Step 2.1: initialize dictionary D
    D = np.random.normal(size=[dim,k]) # sample randomly from normal distribution
    D = sklearn.preprocessing.normalize(D, axis=0, norm='l2') # normalize centroids
    # print 'D.shape', D.shape

    # Step 2.2: initialize code vectors (matrix) S

    # Step 2.2: update until convergence
    epoc = 0
    max_epocs = 100
    change = np.inf
    change_eps = 0.001
    prev_index = np.zeros(N)
    while epoc < max_epocs and change > change_eps:
        S = np.dot(D.T,X)
        # print 'S.shape', S.shape
        centroid_index = np.argmax((S), 0) # np.abs? NO!
        # print 'centroid_index.shape', centroid_index.shape
        s_ij = S[centroid_index, np.arange(N)] # dot products already calculated
        S = np.zeros([k,N])
        S[centroid_index, np.arange(N)] = s_ij
        D += np.dot(X,S.T)
        D = sklearn.preprocessing.normalize(D, axis=0, norm='l2') # normalize

        # Compute change
        change = np.mean(centroid_index != prev_index)
        prev_index = centroid_index
        epoc += 1
        print("EPOC: {:d} CHANGE: {:.4f}".format(epoc,change))

        # # Visualize clustering
        # pl.figure()
        # colors = ['b','r','g', 'c', 'm', 'y', 'k']
        # for n,point in enumerate(X.T):
        #     pl.plot(point[0],point[1], colors[centroid_index[n]] + 'o')
        # pl.axhline(0, color='black')
        # pl.axvline(0, color='black')
        # for n,centroid in enumerate(D.T):
        #     # centroid = pca.inverse_transform(centroid)
        #     pl.plot(centroid[0],centroid[1],colors[n] + 'x')
        # pl.show()

