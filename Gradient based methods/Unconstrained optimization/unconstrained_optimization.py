import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import jacfwd, jacrev
from jax import random
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from visualizaiton import Visualization


class UnconstrainedOptimization:
    def __init__(self, cost, direction_type, stepsize_type, x0,
                 termination_params, algo_params=None):
        self.cost = cost
        self.x0 = jnp.array(x0)
        self.direction_type = direction_type
        self.stepsize_type = stepsize_type
        self.algo_params = algo_params
        self.termination_params = termination_params

    def armijo_condition(self, alpha, sigma, x_current, x_next, direction):
        LHS = self.cost(x_current) - self.cost(x_next)
        RHS = - sigma * alpha * jnp.matmul(grad(self.cost)(x_current), direction)
        condition = True if (LHS >= RHS) else False
        return condition

    def get_step_size(self, x_current, direction):

        if self.stepsize_type == 'minimization_rule':
            # create symbolic step size: alpha
            alpha = sym.Symbol('alpha')
            # convert jax device array to numpy list
            x_current = x_current.tolist()
            direction = direction.tolist()
            sym_vec = []
            for d in range(len(direction)):
                mm = x_current[d] + alpha * direction[d]
                sym_vec.append(mm)
            sym_cost = self.cost(sym_vec)
            sym_derivative = sym.diff(sym_cost, alpha)
            alpha_sym = sym.solveset(sym_derivative, alpha)
            alpha_sym = list(alpha_sym)
            print(alpha_sym)

            # only keep the real number by checking data type
            i = 0
            while True:
                condition = type(alpha_sym[i] ** 2) is sym.numbers.Float
                if condition is True:
                    val = alpha_sym[i]
                    break
                else:
                    i += 1

            alpha = float(val)

        elif self.stepsize_type == 'constant':
            alpha = self.algo_params[2]

        elif self.stepsize_type == 'armijo_rule':
            beta, s, sigma = self.algo_params
            m = 0
            condition = False
            while True:
                alpha = pow(beta, m) * s
                x_next = x_current + alpha * direction
                condition = self.armijo_condition(alpha, sigma, x_current, x_next, direction)

                if condition is True:
                    break
                m += 1
        else:
            raise Exception(
                'Unknown step size type, please input one of the following: minimization_tule, constant, armijo rule')

        return alpha

    def get_gradient_direction(self, x_val):
        # get derivative of the cost function
        derivative = grad(self.cost)(x_val)

        if self.direction_type == 'steepest_descent':
            # D in case of steepest descent is identity matrix
            D = jnp.identity(len(x_val))
            direction = - jnp.matmul(D, derivative)

        elif self.direction_type == 'newton_method':
            hessian = jacfwd(jacrev(self.cost))(x_val)
            D = jnp.linalg.inv(hessian)
            direction = - jnp.matmul(D, derivative)

        else:
            raise Exception(
                'Unknown gradient based method, please input one of the following: '
                'steepest_descent, newton_method')

        return direction

    def update(self, x_current, X_opt, error_val, error_cache):

        # update states
        X_opt.append(np.asarray(x_current, dtype=float))
        direction = self.get_gradient_direction(x_current)
        step_size = self.get_step_size(x_current, direction)
        x_next = x_current + step_size * direction
        error_val = x_current - x_next
        error_norm = jnp.linalg.norm(error_val)
        error_cache.append(error_norm)

        return x_next, X_opt, error_norm, error_cache

    def run_algorithm(self):
        """
        Runs the algorithm provided the user with starting and termination conditions,
        If the termination condition is "convergence" then the algorithm runs until convergence
        is achieved upto the desired value, the other termination option is to run the algorithm for
        a fixed number of timesteps.
        :return: X_opt = a matrix with co-ordinates of all the states obtained from running the algorithm
        """

        # get termination conditions from algo_params
        termination_type, termination_condition = self.termination_params

        x_current = self.x0
        X_opt = []
        error_norm = np.inf
        error_cache = []
        fig = plt.figure()

        if termination_type == "convergence":
            while error_norm > termination_condition:
                x_next, X_opt, error_norm, error_cache = self.update(
                    x_current, X_opt, error_norm, error_cache)
                # update state vector
                x_current = x_next

        elif termination_type == "fixed_steps":
            for t in range(termination_condition):
                x_next, X_opt, error_norm, error_cache = self.update(
                    x_current, X_opt, error_norm, error_cache)
                # update state vector
                x_current = x_next

        return X_opt, error_cache




