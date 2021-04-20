import numpy as np
import pandas as pd
import logging
from fetch_data import (
    static_population_erva_age,
)
from env_var import EPIDEMIC


def forward_integration(u_con, c1, beta, c_gh, T, pop_hat, age_er, t0='2021-04-19', policy_thl=False):
    # number of age groups and ervas
    num_ervas, num_age_groups = age_er.shape
    N_t = T
    N_p = num_ervas
    N_g = num_age_groups

    ####################################################################
    # Time periods for epidemic
    T_E = EPIDEMIC['T_E']
    T_V = EPIDEMIC['T_V']
    T_I = EPIDEMIC['T_I']
    T_q0 = EPIDEMIC['T_q0']
    T_q1 = EPIDEMIC['T_q1']
    T_hw = EPIDEMIC['T_hw']
    T_hc = EPIDEMIC['T_hc']
    T_hr = EPIDEMIC['T_hr']

    # Fraction of nonhospitalized that dies
    mu_q = EPIDEMIC['mu_q'][num_age_groups]
    # Fraction of hospitalized that dies
    mu_w = EPIDEMIC['mu_w'][num_age_groups]
    # Fraction of inds. In critical care that dies
    mu_c = EPIDEMIC['mu_c'][num_age_groups]
    # Fraction of infected needing health care
    p_H = EPIDEMIC['p_H'][num_age_groups]
    # Fraction of hospitalized needing critical care
    p_c = EPIDEMIC['p_c'][num_age_groups]
    alpha = EPIDEMIC['alpha']
    e = EPIDEMIC['e']

    # Reading CSV
    epidemic_csv = pd.read_csv('out/epidemic_finland_8.csv')
    # Getting only date t0
    epidemic_zero = epidemic_csv.loc[epidemic_csv['date'] == t0, :]
    # Removing Ahvenanmaa or Aland
    epidemic_zero = epidemic_zero[~epidemic_zero['erva'].str.contains('land')]

    # Getting the order the ervas have inside the dataframe
    ervas_order = EPIDEMIC['ervas_order']
    ervas_df = list(pd.unique(epidemic_zero['erva']))
    ervas_pd_order = [ervas_df.index(erva) for erva in ervas_order]

    # Droping all the columns we are not using
    epidemic_zero = epidemic_zero.drop(columns=['date', 'erva', 'age', 'First dose', 'Second dose', 'Second dose cumulative', 'infected detected', 'infected undetected'])
    # Converting to numpy
    epidemic_npy = epidemic_zero.values
    # Reshaping to 3d array
    epidemic_npy = epidemic_npy.reshape(N_p, N_g, 5)

    # Rearranging the order of the matrix with correct order
    epidemic_npy = epidemic_npy[ervas_pd_order, :]

    # Allocating space for compartments
    S_g = np.zeros((N_g, N_p, N_t))
    I_g = np.zeros((N_g, N_p, N_t))
    E_g = np.zeros((N_g, N_p, N_t))
    R_g = np.zeros((N_g, N_p, N_t))
    V_g = np.zeros((N_g, N_p, N_t))

    # Adding 1 dimension to age_er to do array division
    age_er_div = age_er[:, :, np.newaxis]
    # Dividing to get the proportion
    epidemic_npy = epidemic_npy/age_er_div

    # Initializing with CSV values
    epidemic_npy = epidemic_npy.transpose(1, 0, 2)
    S_g[:, :, 0] = epidemic_npy[:, :, 0]
    I_g[:, :, 0] = epidemic_npy[:, :, 1]
    E_g[:, :, 0] = epidemic_npy[:, :, 2]
    R_g[:, :, 0] = epidemic_npy[:, :, 3]
    V_g[:, :, 0] = epidemic_npy[:, :, 4]

    # Initializing the rest of compartments to zero
    D_g = np.zeros((N_g, N_p, N_t))
    Q_0g = np.zeros((N_g, N_p, N_t))
    Q_1g = np.zeros((N_g, N_p, N_t))
    H_wg = np.zeros((N_g, N_p, N_t))
    H_cg = np.zeros((N_g, N_p, N_t))
    H_rg = np.zeros((N_g, N_p, N_t))
    S_vg = np.zeros((N_g, N_p, N_t))
    S_xg = np.zeros((N_g, N_p, N_t))

    # I store the values for the force of infection (needed for the adjoint equations)
    L_g = np.zeros((N_g, N_p, N_t))
    # Cummulative number for all age groups and all ervas
    D_d = np.zeros(N_t)

    pop_erva = age_er.sum(axis=1)
    # Initializing all group indicators in the last group
    age_group_indicators = np.array([N_g-1]*N_p)
    delay_check_vacc = EPIDEMIC['delay_check_vacc']
    ws_vacc = EPIDEMIC['ws_vacc']

    # Initialize vaccination rate
    u = np.zeros((N_g, N_p, N_t))

    # Short method to get the normalized metric (infectious or hospitalized)
    # In the lat t-delay period
    def get_metric_erva_weigth(metric, t, delay):
        tot_delay = t - delay
        if tot_delay < 0:
            tot_delay = 0

        metric_t = metric[:, :, tot_delay:t]
        metric_t = metric_t.sum(axis=2)
        metric_t = metric_t*age_er.T
        metric_t_all_ages = metric_t.sum(axis=0)

        if np.allclose(metric_t_all_ages, 0):
            metric_t_all_ages[:] = 1

        metric_t_erva_norm = metric_t_all_ages/np.sum(metric_t_all_ages)

        return metric_t_erva_norm, metric_t_all_ages

    # Forward integration for system of equations (1)
    for j in range(N_t-1):
        D = 0.0
        use_age_groups = age_group_indicators != -1
        # print(use_age_groups)
        # Deciding which policy to use
        if policy_thl:
            hosp_norm, hosp = get_metric_erva_weigth(H_wg+H_cg+H_rg, j, delay_check_vacc)
            infe_norm, infe = get_metric_erva_weigth(I_g, j, delay_check_vacc)
            thl_weight = ws_vacc[0]*pop_erva_prop + ws_vacc[1]*hosp_norm + ws_vacc[2]*infe_norm
            u_erva = u_con*thl_weight
        else:
            # Proportional population of the ERVA
            pops_erva_prop = np.zeros(pop_erva.shape)

            use_pops = pop_erva[use_age_groups]
            use_pops_prop = use_pops/np.sum(use_pops)
            pops_erva_prop[use_age_groups] = use_pops_prop

            u_erva = u_con*pops_erva_prop
        if np.allclose(age_group_indicators, -1):
            print('Time: %d. All ERVAs seem to have finished' % (j, ))
            Susc_t = S_g[:, :, j]
            Susc_t = np.sum(Susc_t)
            print('Susceptibles left: %f.' % (Susc_t, ))
        for g in range(N_g-1, -1, -1):
            for n in range(N_p):
                I_h_force = I_g[:, :, j]
                c_gh_force = c_gh[:, g]
                c_gh_force = c_gh_force[np.newaxis, :]
                mob_k = c1[n, :]
                mob_k = mob_k[:, np.newaxis]

                # Force of infection in equation (4)
                lambda_g = c_gh_force@I_h_force@(c1@mob_k)
                L_g[g, n, j] = lambda_g

                if S_g[g, n, j] - beta*lambda_g*S_g[g, n, j] <= 0:
                    age_group_indicators[n] = g - 1

                # if age_group_indicators[n] == -1:
                #     print('Time: %d. ERVA %d finished with vaccinations.' % (j, n, ))

                age_group_indicator = age_group_indicators[n]
                if age_group_indicator == g:
                    # Assign the vaccines to the current age group indicator and erva
                    u[g, n, j] = u_erva[n]/age_er[n, g]
                    # Check for leftovers in the current age group
                    all_aplied = S_g[g, n, j] - beta*lambda_g*S_g[g, n, j] - u[g, n, j]
                    if all_aplied < 0:
                        left_over = np.abs(all_aplied)
                        # Apply the leftovers to the next age group
                        left_over_real = left_over*age_er[n, g]
                        if g-1 >= 0:
                            u[g-1, n, j] = left_over_real/age_er[n, g-1]

                # Ensures that we do not keep vaccinating after there are no susceptibles left
                u[g, n, j] = min(u[g, n, j], max(0.0, S_g[g, n, j] - beta*lambda_g*S_g[g, n, j]))
                S_g[g, n, j+1] = S_g[g, n, j] - beta*lambda_g*S_g[g, n, j] - u[g, n, j]
                S_vg[g, n, j+1] = S_vg[g, n, j] - beta*lambda_g*S_vg[g, n, j] + u[g, n, j] - T_V*S_vg[g, n, j]
                S_xg[g, n, j+1] = S_xg[g, n, j] - beta*lambda_g*S_xg[g, n, j] + (1.-alpha*e)*T_V*S_vg[g, n, j]
                V_g[g, n, j+1] = V_g[g, n, j] + alpha*e*T_V*S_vg[g, n, j]
                E_g[g, n, j+1] = E_g[g, n, j] + beta*lambda_g*(S_g[g, n, j]+S_vg[g, n, j] + S_xg[g, n, j]) - T_E*E_g[g, n, j]
                I_g[g, n, j+1] = I_g[g, n, j] + T_E*E_g[g, n, j] - T_I*I_g[g, n, j]
                Q_0g[g, n, j+1] = Q_0g[g, n, j] + (1.-p_H[g])*T_I*I_g[g, n, j] - T_q0*Q_0g[g, n, j]
                Q_1g[g, n, j+1] = Q_1g[g, n, j] + p_H[g]*T_I*I_g[g, n, j] - T_q1*Q_1g[g, n, j]
                H_wg[g, n, j+1] = H_wg[g, n, j] + T_q1*Q_1g[g, n, j] - T_hw*H_wg[g, n, j]
                H_cg[g, n, j+1] = H_cg[g, n, j] + p_c[g]*T_hw*H_wg[g, n, j] - T_hc*H_cg[g, n, j]
                H_rg[g, n, j+1] = H_rg[g, n, j] + (1.-mu_c[g])*T_hc*H_cg[g, n, j] - T_hr*H_rg[g, n, j]
                R_g[g, n, j+1] = R_g[g, n, j] + T_hr*H_rg[g, n, j] + (1.-mu_w[g])*(1.-p_c[g])*T_hw*H_wg[g, n, j] + (1.-mu_q[g])*T_q0*Q_0g[g, n, j]
                D_g[g, n, j+1] = D_g[g, n, j] + mu_q[g]*T_q0*Q_0g[g, n, j]+mu_w[g]*(1.-p_c[g])*T_hw*H_wg[g, n, j] + mu_c[g]*T_hc*H_cg[g, n, j]

                D = D + D_g[g, n, j+1]*age_er[n, g]
        D_d[j] = D

    return S_g, S_vg, S_xg, L_g, D_d, D_g, u


def get_model_parameters(number_age_groups, num_ervas, erva_pop_file):
    logger = logging.getLogger()
    pop_ervas_age, _ = static_population_erva_age(logger, erva_pop_file,
                                                  number_age_groups=number_age_groups)
    pop_ervas_age = pop_ervas_age[~pop_ervas_age['erva'].str.contains('All')]
    pop_ervas_age = pop_ervas_age[~pop_ervas_age['erva'].str.contains('land')]
    pop_ervas_age = pop_ervas_age.sort_values(['erva', 'age_group'])
    pop_ervas_npy = pop_ervas_age['Total'].values
    pop_ervas_npy = pop_ervas_npy.reshape(num_ervas, number_age_groups)

    ervas_order = ['HYKS', 'TYKS', 'TAYS', 'KYS', 'OYS']
    ervas_df = list(pd.unique(pop_ervas_age['erva']))
    ervas_pd_order = [ervas_df.index(erva) for erva in ervas_order]
    # Rearrange rows in the correct order
    age_er = pop_ervas_npy[ervas_pd_order, :]

    pop_erva = age_er.sum(axis=1)

    # Contact matrix
    c_gh_3 = EPIDEMIC['contact_matrix'][number_age_groups]

    # Mobility matrix
    m_av = EPIDEMIC['mobility_matrix'][number_age_groups]

    m_av = m_av/pop_erva[:, np.newaxis]

    N_p = num_ervas
    N_g = number_age_groups
    mob_av = np.zeros((N_p, N_p))
    r = EPIDEMIC['r']
    for k in range(N_p):
        for m in range(N_p):
            if k == m:
                mob_av[k, m] = (1.-r) + r*m_av[k, m]
            else:
                mob_av[k, m] = r*m_av[k, m]

    ####################################################################
    # Equation (3) in overleaf (change in population size because of mobility)
    # N_hat_{lg}, N_hat_{l}
    pop_erva_hat = np.zeros(N_p)
    age_er_hat = np.zeros((N_g, N_p))

    for m in range(N_p):
        m_k = 0.0
        for k in range(N_p):
            m_k = m_k + pop_erva[k]*mob_av[k,m]
            for g in range(N_g):
                age_er_hat[g, m] = sum(age_er[:, g]*mob_av[:, m])

        pop_erva_hat[m] = m_k

    ##############################################
    # Population size per age group in all ervas
    age_pop = sum(age_er)

    # Computing beta_gh for force of infection ()
    beta_gh = np.zeros((N_g, N_g))

    for g in range(N_g):
        for h in range(N_g):
            if g == h:
                for m in range(N_p):
                    sum_kg2 = 0.0
                    for k in range(N_p):
                        sum_kg2 = sum_kg2 + age_er[k, g]*mob_av[k, m]*mob_av[k, m]/pop_erva_hat[m]
                sum_kg = sum(age_er_hat[g, :]*age_er_hat[h, :]/pop_erva_hat)
                beta_gh[g, h] = age_pop[g]*c_gh_3[g, h]/(sum_kg-sum_kg2)
            else:
                sum_kg = age_er_hat[g, :]*age_er_hat[h, :]
                beta_gh[g, h] = age_pop[g]*c_gh_3[g, h]/(sum(sum_kg/pop_erva_hat))

    return mob_av, beta_gh, pop_erva_hat, age_er


if __name__ == "__main__":
    erva_pop_file = 'stats/erva_population_age_2020.csv'

    number_age_groups = 8
    num_ervas = 5
    N_p = num_ervas
    N_g = number_age_groups
    mob_av, beta_gh, pop_erva_hat, age_er, = get_model_parameters(number_age_groups, num_ervas, erva_pop_file)

    # number of optimization variables
    N_f = (N_g-3)*N_p
    T = 115
    # transmission parameter
    beta = 0.02
    # Number of vaccines per day
    u = 30000
    t0 = '2021-04-19'
    policy_thl = False

    _, _, _, _, _, D_g, u_g = forward_integration(u, mob_av, beta, beta_gh, T, pop_erva_hat, age_er, t0, policy_thl)
