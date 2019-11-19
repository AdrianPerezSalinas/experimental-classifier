import numpy as np

def partial_matrix(theta, phi):
    """
    This function creates one gate with two parameters: Rz * Ry
    :param theta: Rotation angle in Y
    :param phi: Rotation angle in Z
    :return: matrix of composed rotation
    """
    m = np.array([[np.cos(theta / 2) * np.exp(1j * phi / 2), -np.sin(theta / 2) * np.exp(1j * phi / 2)],
                  [np.sin(theta / 2) * np.exp(-1j * phi / 2), np.cos(theta / 2) * np.exp(-1j * phi / 2)]],
                 dtype=complex)

    return m

def params_translator_2(parameters, x):
    """
    This function translates a flat array of parameters into angles of rotations for a 2-variable function
    :param parameters: weights and biases in a flat array. The lenght must be layers * 4
    :param x: 2-coordinates variable
    :return: angles of rotations
    """
    L = len(parameters) // 4
    vars = parameters.reshape((L, 4))
    theta = vars[:, 0] * x[0] + vars[:, 2]
    phi = vars[:, 1] * x[1] + vars[:, 3]

    return theta, phi


def params_translator_1(parameters, x):
    """
    This function translates a flat array of parameters into angles of rotations for a 2-variable function
    :param parameters: weights and biases in a flat array. The lenght must be layers * 3
    :param x: 1-coordinate variable
    :return: angles of rotations
    """
    L = len(parameters) // 3
    vars = parameters.reshape((L, 3))
    theta = vars[:, 0] * x + vars[:, 2]
    phi = vars[:, 1]

    return theta, phi


def matrix_1(parameters, x):
    """
    This function creates the complete matrix for one one-dimensional variable
    :param parameters: weights and biases in a flat array. The lenght must be layers * 3
    :param x: 1-coordinate variable
    :return: global unitary operation
    """
    theta, phi = params_translator_1(parameters, x)
    M = partial_matrix(theta[0], phi[0])
    i=0
    for t, p in zip(theta[1:], phi[1:]):
        i+=1
        print(i)
        M = partial_matrix(t, p) @ M

    return M


def matrix_2(parameters, x):
    """
    This function creates the complete matrix for one one-dimensional variable
    :param parameters: weights and biases in a flat array. The lenght must be layers * 4
    :param x: 2-coordinates variable
    :return: global unitary operation
    """
    theta, phi = params_translator_2(parameters, x)
    M = partial_matrix(theta[0], phi[0])

    for t, p in zip(theta[1:], phi[1:]):
        M = partial_matrix(t, p) @ M

    return M