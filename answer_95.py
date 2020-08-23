import numpy as np

np.random.seed(0)


class NN:
    def __init__(self):
        self.nets = [
            [np.random.normal(0, 1, [2, 64]),
             np.random.normal(0, 1, [64])],
            [np.random.normal(0, 1, [64, 64]),
             np.random.normal(0, 1, [64])],
            [np.random.normal(0, 1, [64, 1]),
             np.random.normal(0, 1, [1])],
        ]
        self.lr = 0.1  # 学习率

    def sigmoid(self, x):
        # sigmoid 激活函数
        return 1. / (1. + np.exp(-x))

    def forward(self, x):
        # 前向操作
        result = [x]
        for net in self.nets:
            result.append(
                self.sigmoid(np.dot(result[-1], net[0]) + net[1])
            )
        self.result = result
        return self.result[-1]

    def train(self, t):
        # 反馈
        grad_u = None
        len_net = len(self.nets)
        result = self.result[::-1]

        for i, v in enumerate(result[:-1]):
            if grad_u is None:
                grad_u = (v - t) * v * (1 - v)
            else:
                grad_u = np.dot(grad_u, self.nets[len_net - i][0].T) * v * (1 - v)

            grad_w = np.dot(result[i + 1].T, grad_u)
            grad_b = np.dot(np.ones([grad_u.shape[0]]), grad_u)

            self.nets[len_net - 1 - i][0] -= self.lr * grad_w
            self.nets[len_net - 1 - i][1] -= self.lr * grad_b


# train
def train_nn(nn, train_x, train_t, iteration_N=5000):
    for i in range(iteration_N):
        nn.forward(train_x)
        nn.train(train_t)
    return nn


# test
def test_nn(nn, test_x, test_t):
    for j in range(len(test_x)):
        x = train_x[j]
        t = train_t[j]
        print("in:", x, "pred:", nn.result[-1][j])


# train data
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# train label data
train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

# prepare neural network
nn = NN()

# train
nn = train_nn(nn, train_x, train_t, iteration_N=5000)

# test
test_nn(nn, train_x, train_t)
