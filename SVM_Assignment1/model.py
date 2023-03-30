import numpy as np
from tqdm import tqdm

class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        # center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # compute the covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T
        
        # sort the eigenvalues and eigenvectors in descending order
        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[index]
        # select the first n_components eigenvectors
        self.components = eigenvectors[0:self.n_components]
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        # center the data using the mean computed during fit
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # project the data onto the principal components
        return np.dot(X_centered, self.components.T)
    
    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        # pass

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            for i in range(X.shape[0]):
                 # sample a random training example
                rd_idx = np.random.randint(0,X.shape[0])
                rd_x = X[rd_idx]
                rd_y = y[rd_idx]
                # hinge loss and its gradient
                loss = rd_y*(np.dot(rd_x, self.w) + self.b)
                margin = max(0, 1-loss)
                if margin > 0:
                    grad_w = self.w - C * np.dot(rd_x, rd_y)
                    grad_b = -1 * C * rd_y
                else:
                    grad_w = self.w
                    grad_b = 0.0
                # update the parameters (weights and bias) using SGD
                val_w = learning_rate * grad_w 
                val_b = learning_rate * grad_b
                self.w = self.w - val_w
                self.b = self.b - val_b
            # raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.sign(np.dot(X, self.w) + self.b)
        # raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, learning_rate: float, num_iters: int, C: float = 1.0) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        for _i in range(self.num_classes):
            y_val = np.zeros(X.shape[0])
            for _j in range(X.shape[0]):
                if y[_j] == _i:
                    y_val[_j] = 1
                else:
                    y_val[_j] = -1
            # then train the 10 SVM models using the preprocessed data for each class
            self.models[_i].fit(X, y_val, learning_rate, num_iters, C)
        # raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        y_prediction = np.zeros(X.shape[0], dtype = int)
        for sample in range(X.shape[0]):
            score = np.NINF
            idx = -1
            for _class in range(self.num_classes):
                test_prediction = np.dot(X[sample], self.models[_class].w) + self.models[_class].b
                if test_prediction > score:
                    score = test_prediction
                    idx = _class
            y_prediction[sample] = idx
        return y_prediction
        # raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp[i] = np.sum((self.predict(X) == i) & (y == i))
            fp[i] = np.sum((self.predict(X) == i) & (y != i))    
            
        precisionScore = tp/(tp+fp)
        return np.mean(precisionScore)
        # raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        tp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp[i] = np.sum((self.predict(X) == i) & (y == i))
            fn[i] = np.sum((self.predict(X) != i) & (y != i))
        
        recallScore = tp/(tp+fn)
        return np.mean(recallScore)
        # raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp[i] = np.sum((self.predict(X) == i) & (y == i))
            fp[i] = np.sum((self.predict(X) == i) & (y != i))
            fn[i] = np.sum((y == i) & (self.predict(X) != i))
        
        precisionScore = tp/(tp+fp)
        recallScore = tp/(tp+fn)
        f1Score = 2 * precisionScore * recallScore / (precisionScore + recallScore)
        return np.mean(f1Score)
        # raise NotImplementedError
