import numpy as np
from scipy.optimize import linprog

def solve_optimisation(L=5, D=1, t=0.1, n_elements=20, soil_type='sand', W_top=0, beta_0=0.3, beta_p=0.3, gamma_p=10, gamma_s=65, alpha=0.5, s_u=40e3):

    z_n = np.linspace(0, L, n_elements+1)

    soil_type = [soil_type] * n_elements
    gamma_s = [gamma_s] * n_elements
    alpha = [alpha] * n_elements
    s_u = [s_u] * n_elements

    if soil_type[0] == 'sand':
        k_e = [1e-4] * n_elements
    elif soil_type[0] == 'clay':
        k_e = [1e-9] * n_elements

    h_norm = normalised_hydraulic_head(z_n, k_e, D, t)

    A_1, b_1 = equilibrium_conditions_assembly(D, t, z_n, gamma_s, gamma_p, h_norm)
    A_2, b_2 = discontinuity_conditions_assembly(n_elements)
    A_3, b_3 = boundary_conditions_assembly(D, t, W_top, n_elements)
    A_ub, b_ub = yield_conditions_assembly(soil_type, beta_0, beta_p, alpha, s_u, z_n, gamma_s, h_norm)

    A_eq = np.append(np.append(A_1, A_2, axis=0), A_3, axis=0)
    b_eq = np.append(np.append(b_1, b_2, axis=0), b_3, axis=0)

    c = np.zeros(n_elements * 6 + 1)
    c[-1] = -1

    solution = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(None, None))

    return solution


def normalised_hydraulic_head(z_n, k_e, D, t):

    K = hydraulic_matrix_assembly(z_n, k_e, D, t)

    q = np.zeros(len(z_n))
    q[0] = 1

    h = np.linalg.solve(K, q)
    h_norm = h / h[0]

    return h_norm

def hydraulic_matrix_assembly(z_n, k_e, D, t):

    n = len(z_n)
    K = np.zeros((n, n))

    for i, k in enumerate(k_e):
        z_1 = z_n[i]
        z_2 = z_n[i+1]
        L_e = z_2 - z_1
        K_e = hydraulic_conductivity(k, D, t, L_e)
        K[i:i+2, i:i+2] = K[i:i+2, i:i+2] + K_e

    L = z_n[-1]
    A_i = np.pi / 4 * ((D - 2 * t) ** 2)
    K_base = k_e[-1] * A_i * (5 / D + 1 / L)
    K[-1, -1] = K[-1, -1] + K_base

    return K

def hydraulic_conductivity(k, D, t, L_e):

    A_i = np.pi / 4 * ((D - 2 * t) ** 2)

    K = [[ k * A_i / L_e,-k * A_i / L_e],
         [-k * A_i / L_e, k * A_i / L_e]]

    return K

def equilibrium_conditions_assembly(D, t, z_n, gamma_s, gamma_p, h_norm):

    n_elements = len(z_n) - 1
    A = np.zeros((2 * n_elements, 6 * n_elements + 1))
    b = np.zeros(2 * n_elements)

    for i, gamma_s_e in enumerate(gamma_s):

        z_1 = z_n[i]
        z_2 = z_n[i+1]
        L_e = z_2 - z_1

        p_norm_1 = h_norm[i]
        p_norm_2 = h_norm[i+1]

        A_e1, A_e2, b_e = equilibrium_conditions(L_e, D, t, p_norm_1, p_norm_2, gamma_p, gamma_s_e)

        A[2 * i:2 * i + 2, 6 * i:6 * i + 6] = A_e1
        A[2 * i:2 * i + 2, 6 * n_elements] = A_e2
        b[2 * i:2 * i + 2] = b_e

    return A, b

def discontinuity_conditions_assembly(n_elements):

    A = np.zeros((2 * (n_elements-1), 6 * n_elements + 1))
    b = np.zeros(2 * (n_elements-1))

    for i in range(n_elements-1):
        A_d1, A_d2, b_d = discontinuity_conditions()

        A[2 * i:2 * i + 2, 6 * i:6 * i + 12] = A_d1
        A[2 * i:2 * i + 2, 6 * n_elements] = A_d2
        b[2 * i:2 * i + 2] = b_d

    return A, b

def boundary_conditions_assembly(D, t, W_top, n_elements):

    A = np.zeros((2, 6 * n_elements + 1))
    b = np.zeros(2)
    A_b1, A_b2, b_b = boundary_conditions(D, t, W_top)

    A[:, 0:6] = A_b1
    A[:, 6 * n_elements] = A_b2
    b[:] = b_b

    return A, b

def yield_conditions_assembly(soil_type, beta_0, beta_p, alpha, s_u, z_n, gamma_s, h_norm):

    n_elements = len(gamma_s)
    A = np.zeros((4 * n_elements + 4, 6 * n_elements + 1))
    b = np.zeros(4 * n_elements + 4)

    L_e = z_n[1:] - z_n [:-1]
    sigma_v2 = np.cumsum(gamma_s * L_e)
    sigma_v1 = np.append([0], sigma_v2[:-1])
    sigma_v0 = (sigma_v2 + sigma_v1) / 2

    for i, (soil_type_e, alpha_e, s_u_e, sigma_v0_e) in enumerate(zip(soil_type, alpha, s_u, sigma_v0)):
        A_y1, A_y2, b_y = yield_conditions(soil_type_e, beta_0, beta_p, alpha_e, s_u_e, sigma_v0_e)

        A[4 * i:4 * i + 4, 6 * i:6 * i + 6] = A_y1
        A[4 * i:4 * i + 4, 6 * n_elements] = A_y2
        b[4 * i:4 * i + 4] = b_y

    if soil_type[-1] == 'sand':
        A[4 * n_elements:4 * n_elements + 4, 6 * (n_elements-1):6 * (n_elements-1) + 6] = [
            [0, 1, 0, 0, 0, 0],
            [0,-1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0,-1, 0, 0]]

        b[4 * n_elements:4 * n_elements + 4] = [sigma_v2[-1] * 40, 0, sigma_v2[-1] * 40, 0]

    elif soil_type[-1] == 'clay':
        A[4 * n_elements:4 * n_elements + 4, 6 * (n_elements - 1):6 * (n_elements - 1) + 7] = [
            [0, 1, 0, 0, 0, 0, h_norm[-1]],
            [0, -1, 0, 0, 0, 0, -h_norm[-1]],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0]]

        b[4 * n_elements:4 * n_elements + 4] = [9 * s_u[-1], 0, 9 * s_u[-1], 0]

    return A, b

def boundary_conditions(D, t, W_top):

    A_p = np.pi / 4 * (D ** 2 - (D - 2 * t) ** 2)
    A_i = np.pi / 4 * ((D - 2 * t) ** 2)

    A_b1 = [[0, 0, 1, 0, 0, 0],
         [A_p, 0, 0, 0, 0, 0]]
    A_b2 = [0, A_i]
    b_b = [0, W_top]

    return A_b1, A_b2, b_b

def equilibrium_conditions(L_e, D, t, p_norm_1, p_norm_2, gamma_p, gamma_s):

    A_p = np.pi/4*(D**2-(D-2*t)**2)
    A_i = np.pi / 4 * ((D - 2 * t) ** 2)
    P_o = np.pi*D
    P_i = np.pi*(D-2*t)

    A_e1 = [[-1/L_e, 1/L_e, 0, 0, -P_o/A_p, -P_i/A_p],
         [0, 0, -1/L_e, 1/L_e, 0, P_i/A_i]]
    A_e2 = [0, -p_norm_1/L_e+p_norm_2/L_e]
    b_e = [gamma_p, gamma_s]

    return A_e1, A_e2, b_e

def discontinuity_conditions():

    A_d1 = [[0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0]]
    A_d2 = [0, 0]
    b_d = [0, 0]

    return A_d1, A_d2, b_d

def yield_conditions(soil_type, beta_0, beta_p, alpha, s_u, sigma_v0):

    if soil_type=='sand':
        A_y1 = [[0, 0,-beta_p/2,-beta_p/2, 0, 1],
                [0, 0,-beta_p/2,-beta_p/2, 0,-1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0,-1, 0]]
        A_y2 = [0, 0, 0, 0]
        b_y = [(beta_0-beta_p)*sigma_v0, (beta_0-beta_p)*sigma_v0, beta_0*sigma_v0, beta_0*sigma_v0]

    elif soil_type=='clay':
        A_y1 = [[0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0,-1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0,-1, 0]]
        A_y2 = [0, 0, 0, 0]
        b_y = [alpha*s_u, alpha*s_u, alpha*s_u, alpha*s_u]

    else:
        raise

    return A_y1, A_y2, b_y

if __name__ == '__main__':
    print(solve_optimisation())
