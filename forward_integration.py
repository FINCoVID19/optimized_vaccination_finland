import numpy as np
import pandas as pd
import logging
from fetch_data import (
    static_population_erva_age,
)
from scipy.linalg import eigvals
from env_var import EPIDEMIC, EXPERIMENTS


def forward_integration(u_con, c1, beta, c_gh, T, pop_hat, age_er,
                        t0, ws_vacc, e, epidemic_npy, init_vacc, checks=False,
                        u_op_file=None):
    # number of age groups and ervas
    num_ervas, num_age_groups = age_er.shape
    N_p = num_ervas
    N_g = num_age_groups
    N_t = T

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

    pop_erva = age_er.sum(axis=1)
    # Initializing all group indicators in the last group
    age_group_indicators = np.array([N_g-1]*N_p)
    delay_check_vacc = EPIDEMIC['delay_check_vacc']

    # Initialize vaccination rate
    if u_op_file is None:
        u = np.zeros((N_g, N_p, N_t))
        assert np.isclose(np.sum(ws_vacc), 0) or np.isclose(np.sum(ws_vacc), 1)
    else:
        u = np.load(u_op_file)

    # Allocating space for compartments
    S_g = np.zeros((N_g, N_p, N_t))
    I_g = np.zeros((N_g, N_p, N_t))
    E_g = np.zeros((N_g, N_p, N_t))
    R_g = np.zeros((N_g, N_p, N_t))
    V_g = np.zeros((N_g, N_p, N_t))
    H_wg = np.zeros((N_g, N_p, N_t))
    H_cg = np.zeros((N_g, N_p, N_t))
    S_xg = np.zeros((N_g, N_p, N_t))
    D_g = np.zeros((N_g, N_p, N_t))
    Q_0g = np.zeros((N_g, N_p, N_t))
    Q_1g = np.zeros((N_g, N_p, N_t))
    H_rg = np.zeros((N_g, N_p, N_t))
    S_vg = np.zeros((N_g, N_p, N_t))

    # Initializing with CSV values
    S_g[:, :, 0] = epidemic_npy[:, :, 0]
    I_g[:, :, 0] = epidemic_npy[:, :, 1]
    E_g[:, :, 0] = epidemic_npy[:, :, 2]
    R_g[:, :, 0] = epidemic_npy[:, :, 3]
    V_g[:, :, 0] = epidemic_npy[:, :, 4]
    S_xg[:, :, 0] = epidemic_npy[:, :, 5]
    H_wg[:, :, 0] = epidemic_npy[:, :, 6]
    H_cg[:, :, 0] = epidemic_npy[:, :, 7]

    hospitalized_incidence = np.zeros((N_g, N_p, N_t))
    infections_incidence = np.zeros((N_g, N_p, N_t))

    # I store the values for the force of infection (needed for the adjoint equations)
    L_g = np.zeros((N_g, N_p, N_t))

    # Function to calculate the force of infection
    def force_of_inf(I_h, c_gh, k, N_g, N_p, mob, pop_hat):
        fi = 0.0
        for h in range(N_g):
            for m in range(N_p):
                for l in range(N_p):
                    fi = fi + (mob[k, m]*mob[l, m]*I_h[h, l]*c_gh[h]*age_er[l, h])/pop_hat[m]
        return fi

    # Short method to get the normalized metric (infectious or hospitalized)
    # In the lat t-delay period
    def get_metric_erva_weigth(metric, t, delay, use_ervas):
        tot_delay = t - delay
        if tot_delay < 0:
            tot_delay = 0

        # Get the counts in the last period
        metric_t = metric[:, :, tot_delay:t]

        assert metric_t.shape[2] <= delay

        # Sum over all time
        metric_t = metric_t.sum(axis=2)
        # Metric_t is a proportion of erva and age group
        # Here transforming to actual number
        metric_t = metric_t*age_er.T
        # Sum over all age groups
        metric_t_all_ages = metric_t.sum(axis=0)

        # If all the values are close to 0 then assign 1 to all ervas
        # This case can happen at the beginning when we have no hospitalizations
        if np.allclose(metric_t_all_ages, 0):
            metric_t_all_ages[:] = 1

        # Preallocate an array with 0s
        metric_t_erva_norm = np.zeros(metric_t_all_ages.shape)
        # Use_ervas is a boolean flag to indicate
        # which ERVAs have not finished vaccination. Use only these to normalize
        use_metric_ervas = metric_t_all_ages[use_ervas]
        metric_t_erva = use_metric_ervas/np.sum(use_metric_ervas)
        # The rest of the ervas will have a count of 0
        metric_t_erva_norm[use_ervas] = metric_t_erva

        return metric_t_erva_norm, metric_t_all_ages

    # Variable to store the spare vaccines from the last timestep
    remain_last = 0
    # Forward integration for system of equations (1)
    for j in range(N_t-1):
        if u_op_file is None:
            # Sum the remaining vaccines from the last timestep if any
            u_con_remain = u_con + remain_last
            remain_last = 0

            # Checking which ervas have still people to be vaccinated
            use_ervas = age_group_indicators != -1

            # Proportional population of the ERVA
            pops_erva_prop = np.zeros(pop_erva.shape)
            # Only getting the missing ervas
            use_pops = pop_erva[use_ervas]
            # Normalizing with the population of the missing ervas
            use_pops_prop = use_pops/np.sum(use_pops)
            pops_erva_prop[use_ervas] = use_pops_prop

            # Get the normalized counts of infected people and hosp
            hosp_norm, hosp = get_metric_erva_weigth(H_wg+H_cg+H_rg, j, delay_check_vacc, use_ervas)
            infe_norm, infe = get_metric_erva_weigth(infections_incidence, j, delay_check_vacc, use_ervas)
            # Construct the final policy
            policy = ws_vacc[0]*pops_erva_prop + ws_vacc[1]*infe_norm + ws_vacc[2]*hosp_norm

            # Get the vaccines assigned to each erva
            u_erva = u_con_remain*policy

        # Go over all ervas
        for n in range(N_p):
            # Go over all age groups starting from the last one
            for g in range(N_g-1, -1, -1):
                # Calculate the force of infection
                lambda_g = force_of_inf(I_g[:, :, j], c_gh[:, g], n, N_g, N_p, c1, pop_hat)
                L_g[g, n, j] = lambda_g

                if u_op_file is None:
                    # Check if we still can vaccinate someone in this age group
                    if S_g[g, n, j] - beta*lambda_g*S_g[g, n, j] <= 0:
                        # Continue to next age group
                        age_group_indicators[n] = g - 1

                        # If we reach here it means  that we have vaccines
                        # that we are not going to use. Distribute to next erva
                        if g == 0 and u_erva[n] != 0:
                            if n+1 < N_p:
                                u_erva[n+1] += u_erva[n]
                            else:
                                # If the next erva is the last one then to next timestep
                                remain_last += u_erva[n]

                    # Assign the vaccines to the current age group and erva
                    age_group_indicator = age_group_indicators[n]
                    if age_group_indicator == g:
                        # Get the total amount of vaccines
                        u[g, n, j] += u_erva[n]/age_er[n, g]

                        # Check for leftovers in the current age group
                        all_aplied = S_g[g, n, j] - beta*lambda_g*S_g[g, n, j] - u[g, n, j]
                        # We have some leftovers
                        if all_aplied < 0:
                            # Indicate that we should continue to next age group
                            age_group_indicators[n] = g - 1
                            # Get the number of leftovers
                            left_over = np.abs(all_aplied)
                            left_over_real = left_over*age_er[n, g]
                            # The next age group will only have the lefotvers
                            if g-1 >= 0:
                                u_erva[n] = left_over_real
                            # If we finnish with the age groups then give to the next erva
                            elif n+1 < N_p:
                                u_erva[n+1] += left_over_real
                            # If it was the last erva then keep the vaccines for next timestep
                            else:
                                remain_last += left_over_real

                # Ensures that we do not keep vaccinating after there are no susceptibles left
                u[g, n, j] = min(u[g, n, j], max(0.0, S_g[g, n, j] - beta*lambda_g*S_g[g, n, j]))

                # Epidemic dynamics
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

                hospitalized_incidence[g, n, j] = T_q1*Q_1g[g, n, j]
                infections_incidence[g, n, j] = T_E*E_g[g, n, j]

    hospitalized_incidence[:, :, T-1] = T_q1*Q_1g[:, :, T-1]
    infections_incidence[:, :, T-1] = T_E*E_g[:, :, T-1]

    if checks:
        # Final check to see that we always vaccinate u_con people
        u_final = u*age_er.T[:, :, np.newaxis]
        u_final = u_final.sum(axis=0)
        u_final = u_final.sum(axis=0)
        print(u_final)
        print(np.where(~np.isclose(u_final, u_con)))

    return S_g, E_g, H_wg, H_cg, H_rg, I_g, D_g, u, hospitalized_incidence, infections_incidence


def read_initial_values(age_er, init_vacc, t0):
    num_ervas, num_age_groups = age_er.shape
    N_p = num_ervas
    N_g = num_age_groups

    if init_vacc:
        csv_name = 'out/epidemic_finland_%d.csv' % (num_age_groups, )
    else:
        csv_name = 'out/epidemic_finland_%d_no_vacc.csv' % (num_age_groups, )

    # Reading CSV
    epidemic_csv = pd.read_csv(csv_name)
    # Getting only date t0
    epidemic_zero = epidemic_csv.loc[epidemic_csv['date'] == t0, :]
    # Removing Ahvenanmaa or Aland
    epidemic_zero = epidemic_zero[~epidemic_zero['erva'].str.contains('land')]

    # Getting the order the ervas have inside the dataframe
    ervas_order = EPIDEMIC['ervas_order']
    ervas_df = list(pd.unique(epidemic_zero['erva']))
    ervas_pd_order = [ervas_df.index(erva) for erva in ervas_order]

    select_columns = ['susceptible',
                      'infected',
                      'exposed',
                      'recovered',
                      'vaccinated',
                      'vaccinated no imm',
                      'ward',
                      'icu']
    # Selecting the columns to use
    epidemic_zero = epidemic_zero[select_columns]
    # Converting to numpy
    epidemic_npy = epidemic_zero.values
    # Reshaping to 3d array
    epidemic_npy = epidemic_npy.reshape(N_p, N_g, len(select_columns))

    # Rearranging the order of the matrix with correct order
    epidemic_npy = epidemic_npy[ervas_pd_order, :]

    # Adding 1 dimension to age_er to do array division
    age_er_div = age_er[:, :, np.newaxis]
    # Dividing to get the proportion
    epidemic_npy = epidemic_npy/age_er_div

    # epidemic_npy has num_ervas first, compartmetns have age first
    # Transposing to epidemic_npy to accomodate to compartments
    epidemic_npy = epidemic_npy.transpose(1, 0, 2)

    return epidemic_npy


def get_model_parameters(number_age_groups, num_ervas, init_vacc, t0, tau):
    logger = logging.getLogger()
    erva_pop_file = 'stats/erva_population_age_2020.csv'
    pop_ervas_age, _ = static_population_erva_age(logger, erva_pop_file,
                                                  number_age_groups=number_age_groups)
    pop_ervas_age = pop_ervas_age[~pop_ervas_age['erva'].str.contains('All')]
    pop_ervas_age = pop_ervas_age[~pop_ervas_age['erva'].str.contains('land')]
    pop_ervas_age = pop_ervas_age.sort_values(['erva', 'age_group'])
    pop_ervas_npy = pop_ervas_age['Total'].values
    pop_ervas_npy = pop_ervas_npy.reshape(num_ervas, number_age_groups)

    ervas_order = EPIDEMIC['ervas_order']

    ervas_df = list(pd.unique(pop_ervas_age['erva']))
    ervas_pd_order = [ervas_df.index(erva) for erva in ervas_order]
    # Rearrange rows in the correct order
    age_er = pop_ervas_npy[ervas_pd_order, :]

    if init_vacc:
        csv_name = 'out/epidemic_finland_%d.csv' % (number_age_groups, )
    else:
        csv_name = 'out/epidemic_finland_%d_no_vacc.csv' % (number_age_groups, )

    # Reading CSV
    epidemic_csv = pd.read_csv(csv_name)
    # Getting only date t0
    epidemic_zero = epidemic_csv.loc[epidemic_csv['date'] == t0, :]
    # Removing Ahvenanmaa or Aland
    epidemic_zero = epidemic_zero[~epidemic_zero['erva'].str.contains('land')]

    # Getting the order the ervas have inside the dataframe
    ervas_df = list(pd.unique(epidemic_zero['erva']))
    ervas_pd_order = [ervas_df.index(erva) for erva in ervas_order]

    select_columns = ['susceptible',
                      'vaccinated no imm']
    # Selecting the columns to use
    epidemic_zero = epidemic_zero[select_columns]
    # Converting to numpy
    epidemic_npy = epidemic_zero.values
    # Reshaping to 3d array
    epidemic_npy = epidemic_npy.reshape(num_ervas, number_age_groups, len(select_columns))
    epidemic_npy = epidemic_npy.astype(np.float64)
    epidemic_sus = epidemic_npy.sum(axis=2)

    epidemic_sus = epidemic_sus.reshape(num_ervas, number_age_groups)

    # Rearranging the order of the matrix with correct order
    epidemic_sus = epidemic_sus[ervas_pd_order, :]
    # age_er = epidemic_sus

    pop_erva = age_er.sum(axis=1)

    # Contact matrix
    c_gh_3 = EPIDEMIC['contact_matrix'][number_age_groups]

    # Mobility matrix
    m_av = EPIDEMIC['mobility_matrix'][num_ervas]

    m_av = m_av/pop_erva[:, np.newaxis]

    # theta_km
    N_p = num_ervas
    N_g = number_age_groups
    mob_av = np.zeros((N_p, N_p))
    for k in range(N_p):
        for m in range(N_p):
            if k == m:
                mob_av[k, m] = (1-tau) + tau*m_av[k, m]
            else:
                mob_av[k, m] = tau*m_av[k, m]

    # Change in population size because of mobility
    # N_hat_{lg}, N_hat_{l}
    pop_erva_hat = np.zeros(N_p)
    age_er_hat = np.zeros((N_g, N_p))

    for m in range(N_p):
        m_k = 0.0
        for k in range(N_p):
            m_k = m_k + pop_erva[k]*mob_av[k, m]
            for g in range(N_g):
                age_er_hat[g, m] = sum(age_er[:, g]*mob_av[:, m])

        pop_erva_hat[m] = m_k

    # Population size per age group in all ervas
    age_pop = sum(age_er)

    # Computing beta_gh for force of infection
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

    # Computing NGM and rho
    T_I = EPIDEMIC['T_I']**(-1)
    ks = num_ervas
    gs = number_age_groups
    kg = ks*gs
    next_gen_matrix = np.zeros((kg, kg))

    # beta_ti_n = age_er*T_I
    beta_ti_n = epidemic_sus*T_I
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

    eig_vals = eigvals(next_gen_matrix)
    rho = np.abs(np.amax(eig_vals))

    return mob_av, beta_gh, pop_erva_hat, age_er, rho
