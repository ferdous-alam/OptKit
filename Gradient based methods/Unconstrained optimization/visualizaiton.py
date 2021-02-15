import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
import time


class Visualization:
    def __init__(self, X_opt, cost, lb, ub):
        self.X_opt = X_opt
        self.cost = cost
        self.lb = lb
        self.ub = ub

        self.f = plt.figure(figsize=(12, 8))
        plt.rc('axes', linewidth=2)

        x = np.arange(lb, ub, 0.1)
        self.X, self.Y = np.meshgrid(x, x)
        M = {0: self.X, 1: self.Y}
        self.F = self.cost(M)

    def plot_optimization(self):
        plt.plot([self.X_opt[i][0] for i in range(len(self.X_opt))],
                 [self.X_opt[i][1] for i in range(len(self.X_opt))], '-*', alpha=0.5, color='red', markersize=15, lw=2.0)
        plt.contour(self.X, self.Y, self.F)
        plt.xlabel('$x_1$', size=25)
        plt.ylabel('$x_2$', size=25)
        plt.show()
        # f.savefig("steepest_descent_armijo.pdf", bbox_inches='tight')

    def animate_optimization(self):
        camera = Camera(plt.figure(figsize=(10, 8)))

        # policy
        x1 = [self.X_opt[i][0] for i in range(len(self.X_opt))]
        x2 = [self.X_opt[i][0] for i in range(len(self.X_opt))]
        y = [self.F[i][1] for i in range(len(self.F))]

        for i in range(len(x1)):
            x1data, x2data, ydata = x1[:i + 1], x2[:i + 1], y[:i + 1]
            plt.contour(x1data, x2data, ydata, '-', color='white', markersize=10, lw=2.0)
            plt.contour(x1data[-1], x2data[-1], ydata[-1], 'o', color='red', markersize=10, lw=2.0)
            camera.snap()

        anim = camera.animate(blit=True)
        anim.save('optimization_animation.gif', writer='imagemagick')
        plt.show()

        xdata, ydata = [], []

