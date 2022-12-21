import numpy as np

def sigm(x):
    return 1/(1+np.exp(-x))


class ANN:
    def __init__(self, architecture, lr, cutoff_err, max_iter, batch_size, M):
        
        self.lr = lr
        self.cutoff_err = cutoff_err
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weights = []
        architecture = architecture.split(',')
        for idx in range(1, len(architecture)):
            layer_weights = np.random.uniform(-1, 1, (int(architecture[idx-1]) + 1, int(architecture[idx])))
            self.weights.append(layer_weights)
        self.batch_size = batch_size
        self.M = M

    def predict(self, X):
        results = []
        ones = np.ones((np.shape(X)[0], 1))
        X = np.hstack((ones, X))
        tmp = X@self.weights[0]
        tmp = sigm(tmp)
        results.append(tmp)
        for layer_weights in self.weights[1:]:
            ones = np.ones((np.shape(tmp)[0], 1))
            tmp = np.hstack((ones, tmp))
            tmp = tmp@layer_weights
            tmp = sigm(tmp)
            results.append(tmp)
        return results


        
    def fit(self, X, y):
        pred = self.predict(X)
        err = self.calculate_err(pred[-1], y)

        for epoch in range(self.max_iter):

            if epoch % 50 == 0:
                pred = self.predict(X)
                err = self.calculate_err(pred[-1], y)
                print("Greska u {}. iteraciji iznosi: {}".format(epoch, err))
                if err < self.cutoff_err:
                    break
            indexes = np.arange(np.shape(X)[0])

            while indexes.size > 0:
                weight_updates = []
                batch = np.random.choice(indexes, size=self.batch_size, replace=False)
                for value in batch:
                    indexes = np.delete(indexes, np.where(indexes==value))


                for index in batch:#napraviti da se racunaju izlazi za sve odjednom
                    deltas = []
                    izlazi = self.predict(np.reshape(X[index, :], (1, self.M*2)))

                    d = len(izlazi)
                    for i in reversed(range(d)):
                        
                        if i == d - 1:
                            deltas.insert(0, izlazi[i][0] * (1 - izlazi[i][0]) * (y[index, :] - izlazi[i][0]))
                        else:
                            greske = np.multiply(izlazi[i][0] * (1 - izlazi[i][0]), self.weights[i+1][1:, :]@deltas[0])
                            deltas.insert(0, greske)

                    d_updates = len(weight_updates)

                    for i in range(d):
                        if d_updates == d:
                            if i == 0:
                                promjene = np.multiply(np.hstack(([1], X[index, :]))[:, None], deltas[i]) * self.lr
                            else:
                                promjene = np.multiply(np.hstack(([[1]], izlazi[i-1])).T, deltas[i]) * self.lr
                            weight_updates[i] += promjene
                        else:
                            if i == 0:
                                weight_updates.append(np.multiply(np.hstack(([1], X[index, :]))[:, None], deltas[i]) * self.lr)
                            else:
                                weight_updates.append(np.multiply(np.hstack(([[1]], izlazi[i-1])).T, deltas[i]) * self.lr)
                                

                for i in range(len(self.weights)):
                    self.weights[i] += weight_updates[i]

                        
        print("Izlazim!")



    def calculate_err(self, pred, y):
        return np.sum(np.square(y-pred))/(2*np.shape(pred)[0])


