import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from datetime import datetime
import pandas as pd
from env_var import EPIDEMIC, EXPERIMENTS
import logging
from fetch_data import static_population_erva_age
import os
import time
from multiprocessing import Pool
import datetime as dt


def get_vac(u_con, c1, beta, c_gh, T, pop_hat, age_er):
    num_ervas, num_age_groups = age_er.shape
    # Time periods for epidemic
    T_E = EPIDEMIC['T_E']
    T_V = EPIDEMIC['T_V']
    T_I = EPIDEMIC['T_I']
    T_q0 = EPIDEMIC['T_q0']
    T_q1 = EPIDEMIC['T_q1']
    T_hw = EPIDEMIC['T_hw']
    T_hc = EPIDEMIC['T_hc']
    T_hr = EPIDEMIC['T_hr']
    e = EPIDEMIC['e']

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

    N_g = num_age_groups
    N_p = num_ervas
    N_t = 6

    # Reading CSV
    t0 = EXPERIMENTS['t0']

    # Reading CSV
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_name = 'epidemic_finland_%d.csv' % (num_age_groups, )
    csv_path = os.path.join(dir_path, 'out', csv_name)
    # Reading CSV
    epidemic_csv = pd.read_csv(csv_path)
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

    # Allocating space for compartments
    S_g = np.zeros((N_g, N_p, N_t))
    I_g = np.zeros((N_g, N_p, N_t))
    E_g = np.zeros((N_g, N_p, N_t))
    R_g = np.zeros((N_g, N_p, N_t))
    V_g = np.zeros((N_g, N_p, N_t))
    H_wg = np.zeros((N_g, N_p, N_t))
    H_cg = np.zeros((N_g, N_p, N_t))
    S_xg = np.zeros((N_g, N_p, N_t))

    # Adding 1 dimension to age_er to do array division
    age_er_div = age_er[:, :, np.newaxis]
    # Dividing to get the proportion
    epidemic_npy = epidemic_npy/age_er_div

    # epidemic_npy has num_ervas first, compartmetns have age first
    # Transposing to epidemic_npy to accomodate to compartments
    epidemic_npy = epidemic_npy.transpose(1, 0, 2)

    # Initializing with CSV values
    S_g[:, :, 0] = epidemic_npy[:, :, 0]
    I_g[:, :, 0] = epidemic_npy[:, :, 1]
    E_g[:, :, 0] = epidemic_npy[:, :, 2]
    R_g[:, :, 0] = epidemic_npy[:, :, 3]
    V_g[:, :, 0] = epidemic_npy[:, :, 4]
    S_xg[:, :, 0] = epidemic_npy[:, :, 5]
    H_wg[:, :, 0] = epidemic_npy[:, :, 6]
    H_cg[:, :, 0] = epidemic_npy[:, :, 7]

    # Initializing the rest of compartments to zero
    D_g = np.zeros((N_g, N_p, N_t))
    Q_0g = np.zeros((N_g, N_p, N_t))
    Q_1g = np.zeros((N_g, N_p, N_t))
    H_rg = np.zeros((N_g, N_p, N_t))
    S_vg = np.zeros((N_g, N_p, N_t))

    # force of infection in equation (4)
    def force_of_inf(I_h, c_gh, k, N_g, N_p, mob, pop_hat):
        fi = 0.0
        for h in range(N_g):
            for m in range(N_p):
                for l in range(N_p):
                    fi = fi + (mob[k, m]*mob[l, m]*I_h[h, l]*c_gh[h]*age_er[l, h])/pop_hat[m]
        return fi

    # I store the values for the force of infection (needed for the adjoint equations)
    L_g = np.zeros((N_g, N_p, N_t))
    # cummulative number for all age groups and all ervas
    D_d = np.zeros(N_t)
    V_d = np.zeros(N_t)
    hos = np.zeros((N_g, N_p, N_t))
    u = np.zeros((N_g, N_p, N_t))

    remain_last = 0
    u_con = 30000.
    age_group_indicators = np.array([N_g-1]*N_p)
    for j in range(N_t-1):
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
        u_erva = u_con_remain*pops_erva_prop
        d_hos = 0.0
        d = 0.0
        for g in range(N_g-1, -1, -1):
            for n in range(N_p):
                lambda_c = force_of_inf(I_g[:, :, j], c_gh[:, g], n, N_g, N_p, c1, pop_hat)
                L_g[g, n, j] = beta*lambda_c
                lambda_g = lambda_c

                # Check if we still can vaccinate someone in this age group
                if S_g[g, n, j] - beta*lambda_g*S_g[g, n, j] <= 0:
                    age_group_indicators[n] = g - 1

                    # If we reach here it means  that we have vaccines
                    # that we are not goign to use. Distribute to next erva
                    if g == 0 and u_erva[n] != 0:
                        if n+1 < N_p:
                            u_erva[n+1] += u_erva[n]
                        else:
                            # If the next erva is the last one then next timestep
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

                u[g, n, j] = min(u[g,n, j], max(0.0, S_g[g, n, j] - beta*lambda_g*S_g[g, n, j]))
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

                d_hos = d_hos + T_q1*Q_1g[g, n, j]*age_er[n, g]
                d = d + D_g[g, n, j+1]*age_er[n, g]

        D_d[j] = d_hos

        S0 = S_g[:, :, N_t-1]
        Sv0 = S_vg[:, :, N_t-1]
        Sx0 = S_xg[:, :, N_t-1]
        V0 = V_g[:, :, N_t-1]
        E0 = E_g[:, :, N_t-1]
        I0 = I_g[:, :, N_t-1]
        Q00 = Q_0g[:, :, N_t-1]
        Q01 = Q_1g[:, :, N_t-1]
        Hw0 = H_wg[:, :, N_t-1]
        Hc0 = H_cg[:, :, N_t-1]
        Hr0 = H_rg[:, :, N_t-1]
        Rg0 = R_g[:, :, N_t-1]
        D0 = D_g[:, :, N_t-1]

    return S0, Sv0, Sx0, V0, E0, I0, Q00, Q01, Hw0, Hc0, Hr0, Rg0, D0, sum(D_d)


def sol(u_con, c1, beta, c_gh, T, pop_hat, age_er):
    num_ervas, num_age_groups = age_er.shape
    N_g = num_age_groups
    N_p = num_ervas

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

    N_t = T
    # Allocating space for compartments
    S_g = np.zeros((N_g, N_p, N_t))
    I_g = np.zeros((N_g, N_p, N_t))
    E_g = np.zeros((N_g, N_p, N_t))
    R_g = np.zeros((N_g, N_p, N_t))
    V_g = np.zeros((N_g, N_p, N_t))
    H_wg = np.zeros((N_g, N_p, N_t))
    H_cg = np.zeros((N_g, N_p, N_t))
    D_g = np.zeros((N_g, N_p, N_t))
    Q_0g = np.zeros((N_g, N_p, N_t))
    Q_1g = np.zeros((N_g, N_p, N_t))
    H_rg = np.zeros((N_g, N_p, N_t))
    S_vg = np.zeros((N_g, N_p, N_t))
    S_xg = np.zeros((N_g, N_p, N_t))
    s0, svg0, sxg0, vg0, eg0, ig0, q0, q1, hw0, hc0, hr0, rg0, dg0, D1 = get_vac(30000., c1, beta, c_gh, T, pop_hat, age_er)

    S_g[:, :, 0] = s0
    S_vg[:, :, 0] = svg0
    S_xg[:, :, 0] = sxg0

    I_g[:, :, 0] = ig0
    Q_0g[:, :, 0] = q0
    Q_1g[:, :, 0] = q1
    E_g[:, :, 0] = eg0
    H_wg[:, :, 0] = hw0
    H_cg[:, :, 0] = hc0
    R_g[:, :, 0] = rg0
    H_rg[:, :, 0] = hr0
    V_g[:, :, 0] = vg0
    D_g[:, :, 0] = dg0

    # force of infection in equation (4)
    def force_of_inf(I_h, c_gh, k, N_g, N_p, mob, pop_hat):
        fi = 0.0
        for h in range(N_g):
            for m in range(N_p):
                for l in range(N_p):
                    fi = fi + (mob[k, m]*mob[l, m]*I_h[h, l]*c_gh[h]*age_er[l, h])/pop_hat[m]

        return fi

    # I store the values for the force of infection (needed for the adjoint equations)
    L_g = np.zeros((N_g, N_p, N_t))
    # cummulative number for all age groups and all ervas
    D_d = np.zeros(N_t)
    V_d = np.zeros(N_t)

    u = np.zeros((N_g, N_p, N_t))

    for j in range(N_t-1):
        d_hos = 0.0
        v_0 = 0.0
        for g in range(N_g-1, -1, -1):
            for n in range(N_p):
                lambda_c = force_of_inf(I_g[:, :, j], c_gh[:, g], n, N_g, N_p, c1, pop_hat)
                L_g[g, n, j] = beta*lambda_c
                lambda_g = lambda_c

                u[g, n, j] = min(u_con[g, n, j], max(0.0, S_g[g, n, j] - beta*lambda_g*S_g[g, n, j]))
                v_0 = v_0 + u[g, n, j]*age_er[n, g]
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

                if death_optim:
                    d_hos = d_hos + D_g[g, n, j+1]*age_er[n, g] # T_q1*Q_1g[g, n, j]*age_er[n,g]
                else:
                    d_hos = d_hos + T_q1*Q_1g[g, n, j]*age_er[n, g]

        if death_optim:
            D_d[j+1] = d_hos
        else:
            D_d[j] = d_hos
        V_d[j] = v_0

    if death_optim:
        return S_g, S_vg, S_xg, L_g, D_d.max(), V_d, u
    else:
        Df = 0.0
        for h in range(N_g):
            for k in range(N_p):
                Df = Df + T_q1*Q_1g[h, k, N_t-1]*age_er[k, h]
        D_d[N_t-1] = Df
        # np.save("u0.npy", u)
        return S_g, S_vg, S_xg, L_g, sum(D_d), V_d, u


def back_int(Sg, Sv, Sx, Lg, nu, c_hg, beta, T, age_er, mob, pop_erva, ind):
    num_ervas, num_age_groups = age_er.shape

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

    N_g = 5
    N_p = num_ervas
    N_t = T

    # plt.plot(time,I_all)
    lS = np.zeros((N_g, N_p, N_t))
    lSv = np.zeros((N_g, N_p, N_t))
    lSx = np.zeros((N_g, N_p, N_t))
    lE = np.zeros((N_g, N_p, N_t))
    lI = np.zeros((N_g, N_p, N_t))
    lQ_0 = np.zeros((N_g, N_p, N_t))
    lQ_1 = np.zeros((N_g, N_p, N_t))
    lHw = np.zeros((N_g, N_p, N_t))
    lHc = np.zeros((N_g, N_p, N_t))
    lHr = np.zeros((N_g, N_p, N_t))
    lD = np.zeros((N_g, N_p, N_t))

    dH = np.zeros((N_g, N_p, N_t))
    for i in range(N_t-1, -1, -1):
        for g in range(N_g):
            for n in range(N_p):
                lS[g, n, i-1] = lS[g, n, i] - lS[g, n, i]*Lg[g+ind, n, i] + lE[g, n, i]*Lg[g+ind, n, i]
                lSv[g, n, i-1] = lSv[g, n, i] - lSv[g, n, i]*(Lg[g+ind, n, i] + T_V) + lE[g, n, i]*Lg[g+ind, n, i]
                lSx[g, n, i-1] = lSx[g, n, i] - lSx[g, n, i]*Lg[g+ind, n, i] + lE[g, n, i]*Lg[g+ind, n, i]\
                    + lSv[g, n, i]*(1.-alpha*e)*T_V

                lE[g, n, i-1] = lE[g, n, i] - (lE[g, n, i]-lI[g, n, i])*T_E
                sumh = 0.0
                sumh2 = 0.0
                sumh3 = 0.0
                for h in range(N_g):
                    for k in range(N_p):
                        for m in range(N_p):
                            mob_k = mob[k, m]*mob[n, m]/pop_erva[m]
                            sumh = sumh + beta*c_hg[g+ind, h+ind]*Sg[h+ind, k, i]*mob_k*(lE[h, k, i] - lS[h, k, i])/age_er[k, h+ind]
                            sumh2 = sumh2 + beta*c_hg[g+ind, h+ind]*Sv[h+ind, k, i]*mob_k*(lE[h, k, i] - lSv[h, k, i])/age_er[k, h+ind]
                            sumh3 = sumh + beta*c_hg[g+ind, h+ind]*Sx[h+ind, k, i]*mob_k*(lE[h, k, i] - lSx[h, k, i])/age_er[k, h+ind]

                lI[g, n, i-1] = lI[g, n, i] - T_I*lI[g, n, i] + sumh + sumh2 + sumh3 + lQ_0[g, n, i]*(1.-p_H[g+ind])*T_I \
                    + lQ_1[g, n, i]*p_H[g+ind]*T_I
                lQ_0[g, n, i-1] = lQ_0[g, n, i]*(1. - T_q0) + lD[g, n, i]*mu_q[g+ind]*T_q0

                if death_optim:
                    lQ_1[g, n, i-1] = lQ_1[g, n, i]*(1. - T_q1) + lHw[g, n, i]*T_q1  #+ T_q1
                else:
                    lQ_1[g, n, i-1] = lQ_1[g, n, i]*(1. - T_q1) + lHw[g, n, i]*T_q1 + T_q1

                lHw[g, n, i-1] = lHw[g, n, i]*(1.-T_hw) + lHc[g, n, i]*p_c[g+ind]*T_hw \
                    + lD[g, n, i]*mu_w[g+ind]*(1.-p_c[g+ind])*T_hw
                lHc[g, n, i-1] = lHc[g, n, i]*(1.-T_hc) + lHr[g, n, i]*(1.-mu_c[g+ind])*T_hc \
                    + lD[g, n, i]*mu_c[g+ind]*T_hc
                lHr[g, n, i-1] = lHr[g, n, i]*(1.-T_hr)

                if death_optim:
                    lD[g, n, i-1] = lD[g, n, i] + 1.

                # if lS[g,n,i]>lSv[g,n,i]:
                dH[g, n, i] = -lS[g, n, i] + lSv[g, n, i]
            # else:
            #   dH[g,n,i] = 0.0

    return dH


def ob_fun(x):
    Ng = 5
    N_p = 5
    Nt = 110
    nuc = np.reshape(x, (Ng, N_p, Nt))
    nu2 = np.zeros((2, N_p, Nt))
    nu3 = np.zeros((2, N_p, Nt))
    nuf = np.concatenate((nu2, nuc, nu3))
    Sg, Svg, Sxg, Lg, Dg, Vd, vac = sol(nuf, mob_av, beta, beta_gh, T, pop_erva_hat, age_er)

    l = (Dg)
    print(l)

    J = l
    return J


def der(x):
    Ng = 5
    N_p = 5
    Nt = 110

    nuc = np.reshape(x, (Ng, N_p, Nt))
    nu2 = np.zeros((2, N_p, Nt))
    nu3 = np.zeros((2, N_p, Nt))
    nuf = np.concatenate((nu2, nuc, nu3))
    Sg, Svg, Sxg, Lg, Dg, Vd, vac = sol(nuf, mob_av, beta, beta_gh, T, pop_erva_hat, age_er)
    # calculation of the gradient
    dH = back_int(Sg, Svg, Sxg, Lg, u, beta_gh, beta, T, age_er, mob_av, pop_erva_hat, 2)

    dH2 = np.reshape(dH, (Ng*N_p*Nt))

    return dH2


def bound_f(bound_full, T_i, u_op, kg_pairs):
    Ng = 5
    N_p = 5
    T = 110
    T_old = T_i
    T_temp = T_i
    bound_r = np.reshape(bound_full, (Ng, N_p, T))
    Sg, Svg, Sxg, Lg, Dg, Vd, vac = sol(u_op, mob_av, beta, beta_gh, T, pop_erva_hat, age_er)
    Var = False
    for i in range(T_i+1, T):
        for g in range(Ng-1, -1, -1):
            for k in range(N_p):
                if Sg[g+2, k, i] == 0:
                    if (g, k) not in kg_pairs:
                        T_i = i
                        print(T_i, g, k)
                        bound_r[g, k, i-1] = Sg[g+2, k, i-1] - Lg[g+2, k, i-1]*Sg[g+2, k, i-1]
                        bound_r[g, k, i:T] = 0.0
                        T_temp = i
                        kg_pairs.append((g, k))
                        Var = True
        if Var:
            break

    bound_rf = np.reshape(bound_r, Ng*N_p*T)

    return bound_rf, T_i, kg_pairs


def optimize(filename, beta_sim=0.03559801015581483, r=1.0, death_optim_in=False):
    global beta
    beta = beta_sim

    global death_optim
    death_optim = death_optim_in
    r = r

    # contact matrix
    num_ervas = EXPERIMENTS['num_ervas']
    number_age_groups = EXPERIMENTS['num_age_groups']

    c_gh_3 = EPIDEMIC['contact_matrix'][number_age_groups]

    logger = logging.getLogger()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_name = 'erva_population_age_2020.csv'
    erva_pop_file = os.path.join(dir_path, 'stats', csv_name)
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
    # age structure in each erva
    global age_er
    age_er = pop_ervas_npy[ervas_pd_order, :]

    global pop_erva
    pop_erva = age_er.sum(axis=1)

    # mobility matrix
    m_av = EPIDEMIC['mobility_matrix'][num_ervas]

    m_av = m_av/pop_erva[:, np.newaxis]

    N_p = num_ervas
    Ng = number_age_groups

    global mob_av
    mob_av = np.zeros((N_p, N_p))

    print('tau = %s' % (r, ))
    for k in range(N_p):
        for m in range(N_p):
            if k == m:
                mob_av[k, m] = (1.-r) + r*m_av[k, m]
            else:
                mob_av[k, m] = r*m_av[k, m]

    ####################################################################
    # equation (3) in overleaf (change in population size because of mobility) N_hat_{lg}, N_hat_{l}
    global pop_erva_hat
    pop_erva_hat = np.zeros(N_p)
    age_er_hat = np.zeros((Ng, N_p))

    for m in range(N_p):
        m_k = 0.0
        for k in range(N_p):
            m_k = m_k + pop_erva[k]*mob_av[k, m]
            for g in range(Ng):
                age_er_hat[g, m] = sum(age_er[:, g]*mob_av[:, m])

        pop_erva_hat[m] = m_k

    age_pop = sum(age_er)
    global beta_gh
    beta_gh = np.zeros((Ng, Ng))

    for g in range(Ng):
        for h in range(Ng):
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

    ######################################################################################
    N_p = num_ervas
    Ng = number_age_groups
    # number of optimization variables
    N_f = (Ng-4)*N_p
    global T
    T = 110

    # transmission parameter
    global u
    u = np.zeros((Ng, N_p, T))
    time = np.arange(0, T)
    n_max = 30000

    # constraints
    a1 = np.eye(T)*age_er[0, 2]
    a2 = np.eye(T)*age_er[1, 2]
    a3 = np.eye(T)*age_er[2, 2]
    a4 = np.eye(T)*age_er[3, 2]
    a5 = np.eye(T)*age_er[4, 2]

    b1 = np.eye(T)*age_er[0, 3]
    b2 = np.eye(T)*age_er[1, 3]
    b3 = np.eye(T)*age_er[2, 3]
    b4 = np.eye(T)*age_er[3, 3]
    b5 = np.eye(T)*age_er[4, 3]

    c0 = np.eye(T)*age_er[0, 4]
    c01 = np.eye(T)*age_er[1, 4]
    c02 = np.eye(T)*age_er[2, 4]
    c03 = np.eye(T)*age_er[3, 4]
    c04 = np.eye(T)*age_er[4, 4]

    c1 = np.eye(T)*age_er[0, 5]
    c2 = np.eye(T)*age_er[1, 5]
    c3 = np.eye(T)*age_er[2, 5]
    c4 = np.eye(T)*age_er[3, 5]
    c5 = np.eye(T)*age_er[4, 5]

    c6 = np.eye(T)*age_er[0, 6]
    c7 = np.eye(T)*age_er[1, 6]
    c8 = np.eye(T)*age_er[2, 6]
    c9 = np.eye(T)*age_er[3, 6]
    c10 = np.eye(T)*age_er[4, 6]

    Af = np.concatenate((a1, a2, a3, a4, a5,
                         b1, b2, b3, b4, b5,
                         c0, c01, c02, c03, c04,
                         c1, c2, c3, c4, c5, c6, c7, c8, c9, c10), axis=1)

    print(np.shape(Af))
    print(T*N_f)

    b = n_max*np.ones(T)
    print('first')

    cons = {
            "type": "eq", "fun": lambda x:  Af @ x - b,
            'jac': lambda x: Af
    }

    # bounds for minimum and maximum value for the optimization variable
    bound0 = np.zeros(T*N_f)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    x0 = np.zeros(N_f*T)
    bound0 = np.zeros(N_f*T)

    bound1 = np.zeros(T*N_f)
    ########check this again
    bound1[0:T] = n_max/age_er[0, 2]
    bound1[T:2*T] = n_max/age_er[1, 2]
    bound1[2*T:3*T] = n_max/age_er[2, 2]
    bound1[3*T:4*T] = n_max/age_er[3, 2]
    bound1[4*T:5*T] = n_max/age_er[4, 2]

    bound1[5*T:6*T] = n_max/age_er[0, 3]
    bound1[6*T:7*T] = n_max/age_er[1, 3]
    bound1[7*T:8*T] = n_max/age_er[2, 3]
    bound1[8*T:9*T] = n_max/age_er[3, 3]
    bound1[9*T:10*T] = n_max/age_er[4, 3]

    bound1[10*T:11*T] = n_max/age_er[0, 4]
    bound1[11*T:12*T] = n_max/age_er[1, 4]
    bound1[12*T:13*T] = n_max/age_er[2, 4]
    bound1[13*T:14*T] = n_max/age_er[3, 4]
    bound1[14*T:15*T] = n_max/age_er[4, 4]

    bound1[15*T:16*T] = n_max/age_er[0, 5]
    bound1[16*T:17*T] = n_max/age_er[1, 5]
    bound1[17*T:18*T] = n_max/age_er[2, 5]
    bound1[18*T:19*T] = n_max/age_er[3, 5]
    bound1[19*T:20*T] = n_max/age_er[4, 5]

    bound1[20*T:21*T] = n_max/age_er[0, 6]
    bound1[21*T:22*T] = n_max/age_er[1, 6]
    bound1[22*T:23*T] = n_max/age_er[2, 6]
    bound1[23*T:24*T] = n_max/age_er[3, 6]
    bound1[24*T:25*T] = n_max/age_er[4, 6]

    bounds = Bounds(bound0, bound1)

    res = minimize(ob_fun, x0, method='SLSQP', jac=der,
                   constraints=[cons], options={'maxiter': 5, 'disp': True},
                   bounds=bounds)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    opc_o = np.reshape(res.x, (5, N_p, T))
    nu2_o = np.zeros((2, N_p, T))
    nu3_o = np.zeros((2, N_p, T))
    nuf_o = np.concatenate((nu2_o, opc_o, nu3_o))
    u_old = nuf_o

    T_old = 0
    bound_old = bound1

    old_kg_pairs = []
    for i in range(24):
        bound_new, T_new, new_kg_pairs = bound_f(bound_old, T_old, u_old, old_kg_pairs)
        bound_old = bound_new
        T_old = T_new
        old_kg_pairs = new_kg_pairs
        if len(new_kg_pairs) >= 24:
            nuf = u_old
            break
        bounds = Bounds(bound0, bound_new)

        res = minimize(ob_fun, x0, method='SLSQP', jac=der,
                       constraints=[cons], options={'maxiter': 3, 'disp': True},
                       bounds=bounds)

        opc = np.reshape(res.x, (5, N_p, T))
        nu2 = np.zeros((2, N_p, T))
        nu3 = np.zeros((2, N_p, T))
        nuf = np.concatenate((nu2, opc, nu3))
        u_old = nuf

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    np.save(filename, nuf)
    print('File written to: %s' % (filename, ))


def run_optimize(r, beta_sim, tau, death_optim_in):
    filename = "%ssol_tau%s_deathoptim%s.npy" % (r, tau, death_optim_in)
    try:
        start_time = time.time()
        proc_number = os.getpid()
        print('Starting (%s). R: %s. Tau: %s. Death optim: %s' % (proc_number, r,
                                                                  tau, death_optim_in))

        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, 'out', filename)
        optimize(beta_sim=beta_sim, r=tau, filename=file_path,
                 death_optim_in=death_optim_in)

        elapsed_time = time.time() - start_time
        elapsed_delta = dt.timedelta(seconds=elapsed_time)
        print('Finished (%s). R: %s. Tau: %s. Death optim: %s. Time: %s' % (proc_number,
                                                                            r,
                                                                            tau,
                                                                            death_optim_in,
                                                                            elapsed_delta))
        return filename
    except Exception:
        logger = logging.getLogger()
        numeric_log_level = getattr(logging, "DEBUG", None)
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%d/%m/%Y %H:%M:%S %p',
            level=numeric_log_level,
            handlers=[logging.StreamHandler()]
        )
        logger.exception("Something went wrong in optimization: %s" % (filename, ))
        print("Something went wrong in optimization: %s" % (filename, ))
        return None


def run_parallel_optimizations():
    all_experiments = [
        (0.75,  0.,  0.016577192790495632, True),
        (0.75,  0.5,  0.017420081058752156, True),
        (0.75,  1.0,  0.017799005077907416, True),
        (1.0,  0.,  0.022102923720660844, True),
        (1.0,  0.5,  0.023226774745002877, True),
        (1.0,  1.0,  0.023732006770543223, True),
        (1.25,  0.,  0.027628654650826055, True),
        (1.25,  0.5,  0.029033468431253595, True),
        (1.25,  1.0,  0.02966500846317903, True),
        (1.5,  0.,  0.033154385580991264, True),
        (1.5,  0.5,  0.03484016211750431, True),
        (1.5,  1.0,  0.03559801015581483, True),
        (0.75,  0.,  0.016577192790495632, False),
        (0.75,  0.5,  0.017420081058752156, False),
        (0.75,  1.0,  0.017799005077907416, False),
        (1.0,  0.,  0.022102923720660844, False),
        (1.0,  0.5,  0.023226774745002877, False),
        (1.0,  1.0,  0.023732006770543223, False),
        (1.25,  0.,  0.027628654650826055, False),
        (1.25,  0.5,  0.029033468431253595, False),
        (1.25,  1.0,  0.02966500846317903, False),
        (1.5,  0.,  0.033154385580991264, False),
        (1.5,  0.5,  0.03484016211750431, False),
        (1.5,  1.0,  0.03559801015581483, False),
    ]
    num_cpus = os.cpu_count()
    start_time = time.time()
    num_experiments = len(all_experiments)
    result_filenames = []
    print('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute forward simulation in asynchronous way
        async_res = [pool.apply_async(func=run_optimize,
                                      args=(r, beta_sim, tau, death_optim_in))
                     for r, tau, beta_sim, death_optim_in in all_experiments]

        # Waiting for the values of the async execution
        for res in async_res:
            filename = res.get()
            result_filenames.append(filename)
    elapsed_time = time.time() - start_time
    elapsed_delta = dt.timedelta(seconds=elapsed_time)
    print('Finished experiments. Elapsed: %s' % (elapsed_delta, ))
    print('Resulting filenames: %s' % (result_filenames, ))


if __name__ == "__main__":
    run_parallel_optimizations()
