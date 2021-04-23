import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read file

def get_data(filename):
    data = pd.read_csv(filename, header=None, sep=r'\s+')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = np.array(data, dtype=float)
    X = data[:,:15]
    Y = data[:,15:]
    return X,Y

def normalize_and_add_ones(X):
    X_max = np.array([[np.amax(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])
    X_normalize = (X - X_min) / (X_max - X_min)
    return np.concatenate((np.ones((X.shape[0], 1)), X_normalize), axis=1)


class RidgeRegression:
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        X_tp = X_train.transpose()
        w = np.linalg.inv(X_tp.dot(X_train) + LAMBDA * np.identity(X_train.shape[1])).dot(X_tp).dot(Y_train)
        return w

    def fit_gradient(self, X_train, Y_train, LAMBDA, lr, max_num_epoch=10000, batch_size=128):
        W = np.array([np.random.randn(X_train.shape[1])]).T
        last_loss = 10e+8
        for ep in range(max_num_epoch):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
            for i in range(total_minibatch):
                index = i * batch_size
                X_train_sub = X_train[index:index + batch_size]
                Y_train_sub = Y_train[index:index + batch_size]
                grad =  X_train_sub.T.dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                W = W - lr * grad
            new_loss = self.compute_RSS(self.predict(W, X_train), Y_train)
            if (np.abs(new_loss - last_loss) <= 1e-5):
                break
            last_loss = new_loss
        return W

    def predict(self, W, X_new):
        return X_new.dot(W)

    def compute_RSS(self, Y_new, Y_predicted):
        loss = 1 / Y_new.shape[0] * np.sum((Y_new - Y_predicted) ** 2)
        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predicted = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predicted)
            return aver_RSS

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0, minimum_RSS=10000 ** 2, LAMBDA_values=range(50)) #[0,1,...,50]
        LAMBDA_values = [k*1/1000 for k in range(max(0,(best_LAMBDA-1)),(best_LAMBDA+1)*1000,1)] #Step 0.001

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA,minimum_RSS= minimum_RSS,LAMBDA_values=LAMBDA_values)
        return best_LAMBDA

if __name__ == '__main__':
    X,Y = get_data('Data_Deathrate.txt')

    X = normalize_and_add_ones(X)
    X_train, Y_train = X[0:50],Y[:50]
    X_test, Y_test = X[50:],Y[50:]

    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train,Y_train)
    print('Best lambda = ',best_LAMBDA)
    w_learned = ridge_regression.fit(X_train,Y_train,best_LAMBDA)
    w_learned_gd = ridge_regression.fit_gradient(X_train,Y_train,best_LAMBDA,lr = 0.01, max_num_epoch= 10000,batch_size=30)
    Y_predicted = ridge_regression.predict(W = w_learned,X_new=X_test)
    Y_predicted_gd = ridge_regression.predict(W=w_learned_gd, X_new=X_test)
    print("Use inv: ",ridge_regression.compute_RSS(Y_test,Y_predicted))
    print("Use ridge regression",ridge_regression.compute_RSS(Y_test,Y_predicted_gd))



