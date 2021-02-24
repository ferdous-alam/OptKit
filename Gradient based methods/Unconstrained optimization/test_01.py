from unconstrained_optimization import UnconstrainedOptimization
from visualizaiton import Visualization

cost = lambda x: 3*pow(x[0], 2) + pow(x[1], 4)


x0 = [1.0, -2.0]
algo_params = [0.5, 1, 0.1]
opt = UnconstrainedOptimization(cost, 'newton_method', 'armijo_rule', x0,
                                ['fixed_steps', 25], algo_params)
X_opt, error_cache = opt.run_algorithm()
viz = Visualization(X_opt, cost, -2.0, 2.0)
viz.plot_optimization()
