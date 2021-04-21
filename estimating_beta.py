import numpy as np
from env_var import EPIDEMIC
from forward_integration import get_model_parameters
from scipy.linalg import eigvals


def construct_next_generation_matrix(beta_gh, pop_erva_hat, mob_av, age_er):
    T_I = EPIDEMIC['T_I']**(-1)
    beta = 1

    num_ervas, num_age_groups = age_er.shape
    ks = num_ervas
    gs = num_age_groups

    kg = ks*gs
    next_gen_matrix = np.zeros((kg, kg))

    beta_ti_n = age_er*T_I*beta

    for k in range(ks):
        for g in range(gs):
            kg_idx = k*gs + g
            beta_ti_n_kg = beta_ti_n[k, g]
            for l in range(ks):
                for h in range(gs):
                    lh_idx = l*gs + h

                    interaction_term = beta_ti_n_kg*beta_gh[g, h]
                    mobility_term = 0
                    for m in range(ks):
                        mobility_term += mob_av[k, m]*mob_av[l, m]/pop_erva_hat[m]
                    next_gen_matrix[kg_idx, lh_idx] = interaction_term*mobility_term
                    # print("%d,%d" % (kg_idx, lh_idx), end=' ')
            # print('')
    eig_vals = eigvals(next_gen_matrix)
    r_effective = np.abs(np.amax(eig_vals))

    return r_effective


if __name__ == "__main__":
    erva_pop_file = 'stats/erva_population_age_2020.csv'

    number_age_groups = 8
    num_ervas = 5
    N_p = num_ervas
    N_g = number_age_groups
    mob_av, beta_gh, pop_erva_hat, age_er, = get_model_parameters(number_age_groups, num_ervas, erva_pop_file)

    rho = construct_next_generation_matrix(beta_gh, pop_erva_hat, mob_av, age_er)

    beta_09 = 0.9/rho
    print('R=0.9 Beta=%f' % (beta_09, ))
    beta_11 = 1.1/rho
    print('R=1.1 Beta=%f' % (beta_11, ))
    beta_13 = 1.3/rho
    print('R=1.3 Beta=%f' % (beta_13, ))
