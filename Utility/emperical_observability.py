
import numpy as np


def empirical_observability_matrix(system, x0, tsim, eps=1e-4, args=None):
    """ Empirically calculates the observability matrix O for a given system & input.

        Inputs
            system:             simulator object
            x0:                 initial state
            tsim:               simulation time
            usim:               simulation inputs
            eps:                amount to perturb initial state

        Outputs
            O:                  numerically calculated observability matrix
            X                   nominal trajectory simulation data
            deltay:             the difference in perturbed measurements at each time step
                                (basically O stored in a 3D array)
    """

    # Simulate once for nominal trajectory
    X, Y, U = system.simulate(x0, tsim, *args)
    n_state = X.shape[0]  # number of states
    n_output = Y.shape[0]  # number of outputs

    # Calculate O
    w = len(tsim)  # of points in time window
    delta = eps * np.eye(n_state)  # perturbation amount for each state
    deltay = np.zeros((n_output, n_state, w))  # preallocate deltay
    for k in range(n_state):  # each state
        # Perturb initial condition in both directions
        x0plus = x0 + delta[:, k]
        x0minus = x0 - delta[:, k]

        # Simulate measurements from perturbed initial conditions
        _, yplus, _ = system.simulate(x0plus, tsim, *args)
        _, yminus, _ = system.simulate(x0minus, tsim, *args)

        # Calculate the numerical Jacobian & normalize by 2x the perturbation amount
        deltay[:, k, :] = np.array(yplus - yminus) / (2 * eps)

    # Construct O by stacking the 3rd dimension of deltay along the 1st dimension, O is a (p*m x n) matrix
    O = []  # list to store datat at each time point fo O
    for j in range(w):
        O.append(deltay[:, :, j])

    O = np.vstack(O)

    return {'bigO': O, 'X': X, 'deltay': deltay, 'Y': Y, 'U': U}