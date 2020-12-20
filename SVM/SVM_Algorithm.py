import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")


class SupportVectorMachine:
    def __init__(self, data, visualization=True):
        self.visualization = visualization
        self.colors = {1: "r", -1: "b"}  # color coding
        self.data = data
        self.min_feature_value = 0
        self.max_feature_value = 0
        self.w = np.zeros((2, 1))
        self.b = 0
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self):
        opt_dict = {}
        transforms = [[1, 1],
                      [1, -1],
                      [-1, 1],
                      [-1, -1]]

        self.max_feature_value = max(np.amax(self.data[-1]), np.amax(self.data[1]))
        self.min_feature_value = min(np.amin(self.data[-1]), np.amin(self.data[1]))

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        b_range_multiple = 1  # most computationally expensive
        b_multiple = 5  # b steps are not as small as for w
        latest_optimum = self.max_feature_value * 10  # cutting corners to get starting w

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False  # w_t and b should not fail even for a single data point
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print("Optimised a step")
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker="*", c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])}

svm = SupportVectorMachine(data_dict)
svm.fit()
predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]
for p in predict_us:
    svm.predict(p)
svm.visualize()
