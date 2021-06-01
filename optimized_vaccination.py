import os
import time
import datetime
import json
import logging
import logging.handlers
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
import pandas as pd
from env_var import EPIDEMIC
from forward_integration import get_model_parameters


def sol(u_con, mob_av, beta, beta_gh, T, pop_hat, age_er, epidemic_npy, return_states):
    T_E = EPIDEMIC['T_E']
    T_V = EPIDEMIC['T_V']
    T_I = EPIDEMIC['T_I']
    T_q0 = EPIDEMIC['T_q0']
    T_q1 = EPIDEMIC['T_q1']
    T_hw = EPIDEMIC['T_hw']
    T_hc = EPIDEMIC['T_hc']
    T_hr = EPIDEMIC['T_hr']

    # Fraction of nonhospitalized that dies
    mu_q = EPIDEMIC['mu_q'][N_g]
    # Fraction of hospitalized that dies
    mu_w = EPIDEMIC['mu_w'][N_g]
    # Fraction of inds. In critical care that dies
    mu_c = EPIDEMIC['mu_c'][N_g]
    # Fraction of infected needing health care
    p_H = EPIDEMIC['p_H'][N_g]
    # Fraction of hospitalized needing critical care
    p_c = EPIDEMIC['p_c'][N_g]
    alpha = EPIDEMIC['alpha']
    e = EPIDEMIC['e']

    # Allocating space for compartments
    S_g = np.zeros((N_g, N_p, T))
    I_g = np.zeros((N_g, N_p, T))
    E_g = np.zeros((N_g, N_p, T))
    R_g = np.zeros((N_g, N_p, T))
    V_g = np.zeros((N_g, N_p, T))
    H_wg = np.zeros((N_g, N_p, T))
    H_cg = np.zeros((N_g, N_p, T))
    S_xg = np.zeros((N_g, N_p, T))
    D_g = np.zeros((N_g, N_p, T))
    Q_0g = np.zeros((N_g, N_p, T))
    Q_1g = np.zeros((N_g, N_p, T))
    H_rg = np.zeros((N_g, N_p, T))
    S_vg = np.zeros((N_g, N_p, T))

    # Initializing with values
    S_g[:, :, 0] = epidemic_npy[:, :, 0]
    I_g[:, :, 0] = epidemic_npy[:, :, 1]
    E_g[:, :, 0] = epidemic_npy[:, :, 2]
    R_g[:, :, 0] = epidemic_npy[:, :, 3]
    V_g[:, :, 0] = epidemic_npy[:, :, 4]
    S_xg[:, :, 0] = epidemic_npy[:, :, 5]
    H_wg[:, :, 0] = epidemic_npy[:, :, 6]
    H_cg[:, :, 0] = epidemic_npy[:, :, 7]

    D_g[:, :, 0] = epidemic_npy[:, :, 8]
    Q_0g[:, :, 0] = epidemic_npy[:, :, 9]
    Q_1g[:, :, 0] = epidemic_npy[:, :, 10]
    H_rg[:, :, 0] = epidemic_npy[:, :, 11]
    S_vg[:, :, 0] = epidemic_npy[:, :, 12]

    # I store the values for the force of infection (needed for the adjoint equations)
    L_g = np.zeros((N_g, N_p, T))

    # number of hospitalizations
    D_d = np.zeros((N_g, N_p, T))

    u = np.zeros((N_g, N_p, T))

    p_H_ages = p_H[:, np.newaxis]
    p_c_ages = p_c[:, np.newaxis]
    mu_c_ages = mu_c[:, np.newaxis]
    mu_w_ages = mu_w[:, np.newaxis]
    mu_q_ages = mu_q[:, np.newaxis]

    age_er_t = age_er.T
    mob_k_pop = mob_av/pop_hat[np.newaxis, :]
    mobility_term = mob_av@mob_k_pop.T
    for j in range(T-1):
        # force of infection in equation (4)
        infect_mobility = (I_g[:, :, j]*age_er_t)@mobility_term.T
        lambda_g = beta_gh.T@infect_mobility

        L_g[:, :, j] = beta*lambda_g

        u[:, :, j] = np.minimum(u_con[:, :, j], np.maximum(0.0, S_g[:, :, j] - beta*lambda_g*S_g[:, :, j]))
        S_g[:, :, j+1] = S_g[:, :, j] - beta*lambda_g*S_g[:, :, j] - u[:, :, j]
        S_vg[:, :, j+1] = S_vg[:, :, j] - beta*lambda_g*S_vg[:, :, j] + u[:, :, j] - T_V*S_vg[:, :, j]
        S_xg[:, :, j+1] = S_xg[:, :, j] - beta*lambda_g*S_xg[:, :, j] + (1.-alpha*e)*T_V*S_vg[:, :, j]
        V_g[:, :, j+1] = V_g[:, :, j] + alpha*e*T_V*S_vg[:, :, j]
        E_g[:, :, j+1] = E_g[:, :, j] + beta*lambda_g*(S_g[:, :, j]+S_vg[:, :, j] + S_xg[:, :, j]) - T_E*E_g[:, :, j]
        I_g[:, :, j+1] = I_g[:, :, j] + T_E*E_g[:, :, j] - T_I*I_g[:, :, j]
        Q_0g[:, :, j+1] = Q_0g[:, :, j] + (1-p_H_ages)*T_I*I_g[:, :, j] - T_q0*Q_0g[:, :, j]
        Q_1g[:, :, j+1] = Q_1g[:, :, j] + p_H_ages*T_I*I_g[:, :, j] - T_q1*Q_1g[:, :, j]
        H_wg[:, :, j+1] = H_wg[:, :, j] + T_q1*Q_1g[:, :, j] - T_hw*H_wg[:, :, j]
        H_cg[:, :, j+1] = H_cg[:, :, j] + p_c_ages*T_hw*H_wg[:, :, j] - T_hc*H_cg[:, :, j]
        H_rg[:, :, j+1] = H_rg[:, :, j] + (1-mu_c_ages)*T_hc*H_cg[:, :, j] - T_hr*H_rg[:, :, j]
        R_g[:, :, j+1] = R_g[:, :, j] + T_hr*H_rg[:, :, j] + (1-mu_w_ages)*(1-p_c_ages)*T_hw*H_wg[:, :, j] + (1-mu_q_ages)*T_q0*Q_0g[:, :, j]
        D_g[:, :, j+1] = D_g[:, :, j] + mu_q_ages*T_q0*Q_0g[:, :, j]+mu_w_ages*(1-p_c_ages)*T_hw*H_wg[:, :, j] + mu_c_ages*T_hc*H_cg[:, :, j]

        D_d[:, :, j] = T_q1*Q_1g[:, :, j]*age_er_t

    D_d[:, :, T-1] = T_q1*Q_1g[:, :, T-1]*age_er_t

    V_d = u*age_er_t[:, :, np.newaxis]
    V_d = V_d.sum(axis=(0, 1))

    if not return_states:
        return S_g, S_vg, S_xg, L_g, D_d.sum(), V_d, u

    new_epidemic_npy = np.zeros((N_g, N_p, 13))
    new_epidemic_npy[:, :, 0] = S_g[:, :, T-1]
    new_epidemic_npy[:, :, 1] = I_g[:, :, T-1]
    new_epidemic_npy[:, :, 2] = E_g[:, :, T-1]
    new_epidemic_npy[:, :, 3] = R_g[:, :, T-1]
    new_epidemic_npy[:, :, 4] = V_g[:, :, T-1]
    new_epidemic_npy[:, :, 5] = S_xg[:, :, T-1]
    new_epidemic_npy[:, :, 6] = H_wg[:, :, T-1]
    new_epidemic_npy[:, :, 7] = H_cg[:, :, T-1]

    new_epidemic_npy[:, :, 8] = D_g[:, :, T-1]
    new_epidemic_npy[:, :, 9] = Q_0g[:, :, T-1]
    new_epidemic_npy[:, :, 10] = Q_1g[:, :, T-1]
    new_epidemic_npy[:, :, 11] = H_rg[:, :, T-1]
    new_epidemic_npy[:, :, 12] = S_vg[:, :, T-1]

    return D_d.sum(), u, new_epidemic_npy


def back_int(S_g, S_vg, S_xg, L_g, beta_gh, beta, T, age_er, mob_av, pop_hat):
    T_E = EPIDEMIC['T_E']
    T_V = EPIDEMIC['T_V']
    T_I = EPIDEMIC['T_I']
    T_q0 = EPIDEMIC['T_q0']
    T_q1 = EPIDEMIC['T_q1']
    T_hw = EPIDEMIC['T_hw']
    T_hc = EPIDEMIC['T_hc']
    T_hr = EPIDEMIC['T_hr']

    # Fraction of nonhospitalized that dies
    mu_q = EPIDEMIC['mu_q'][N_g]
    # Fraction of hospitalized that dies
    mu_w = EPIDEMIC['mu_w'][N_g]
    # Fraction of inds. In critical care that dies
    mu_c = EPIDEMIC['mu_c'][N_g]
    # Fraction of infected needing health care
    p_H = EPIDEMIC['p_H'][N_g]
    # Fraction of hospitalized needing critical care
    p_c = EPIDEMIC['p_c'][N_g]
    alpha = EPIDEMIC['alpha']
    e = EPIDEMIC['e']

    lS = np.zeros((N_g, N_p, T))
    lSv = np.zeros((N_g, N_p, T))
    lSx = np.zeros((N_g, N_p, T))
    lE = np.zeros((N_g, N_p, T))
    lI = np.zeros((N_g, N_p, T))
    lQ_0 = np.zeros((N_g, N_p, T))
    lQ_1 = np.zeros((N_g, N_p, T))
    lHw = np.zeros((N_g, N_p, T))
    lHc = np.zeros((N_g, N_p, T))
    lHr = np.zeros((N_g, N_p, T))
    lD = np.zeros((N_g, N_p, T))

    dH = np.zeros((N_g, N_p, T))

    mob_n = mob_av/pop_hat[np.newaxis, :]
    mob_n = mob_n[:, np.newaxis, :]
    mob_av_exp = mob_av[np.newaxis, :, :]
    mob_k = mob_n*mob_av_exp

    beta_term = beta*beta_gh
    beta_term = beta_term[:, np.newaxis, :]

    p_H_ages = p_H[:, np.newaxis]
    p_c_ages = p_c[:, np.newaxis]
    mu_c_ages = mu_c[:, np.newaxis]
    mu_w_ages = mu_w[:, np.newaxis]
    mu_q_ages = mu_q[:, np.newaxis]

    for i in range(T-1, -1, -1):
        lS[:, :, i-1] = lS[:, :, i] - lS[:, :, i]*L_g[:, :, i] + lE[:, :, i]*L_g[:, :, i]
        lSv[:, :, i-1] = lSv[:, :, i] - lSv[:, :, i]*(L_g[:, :, i] + T_V) + lE[:, :, i]*L_g[:, :, i]
        lSx[:, :, i-1] = lSx[:, :, i] - lSx[:, :, i]*L_g[:, :, i] + lE[:, :, i]*L_g[:, :, i]\
            + lSv[:, :, i]*(1-alpha*e)*T_V

        lE[:, :, i-1] = lE[:, :, i] - (lE[:, :, i]-lI[:, :, i])*T_E

        S_g_term = S_g[:, :, i]*((lE[:, :, i] - lS[:, :, i])/age_er.T)
        S_vg_term = S_vg[:, :, i]*((lE[:, :, i] - lSv[:, :, i])/age_er.T)
        S_xg_term = S_xg[:, :, i]*((lE[:, :, i] - lSx[:, :, i])/age_er.T)

        part_sumh = beta_term*S_g_term.T
        part_sumh = part_sumh.transpose(0, 2, 1)
        sumh = np.einsum('ijk,lkm->iljk', part_sumh, mob_k)
        sumh = sumh.sum(axis=(2, 3))

        part_sumh2 = beta_term*S_vg_term.T
        part_sumh2 = part_sumh2.transpose(0, 2, 1)
        sumh2 = np.einsum('ijk,lkm->iljk', part_sumh2, mob_k)
        sumh2 = sumh2.sum(axis=(2, 3))

        part_sumh3 = beta_term*S_xg_term.T
        part_sumh3 = part_sumh3.transpose(0, 2, 1)
        sumh3 = np.einsum('ijk,lkm->iljk', part_sumh3, mob_k)
        sumh3 = sumh3.sum(axis=(2, 3))

        lI[:, :, i-1] = lI[:, :, i] - T_I*lI[:, :, i] + sumh + sumh2 + sumh3 + lQ_0[:, :, i]*(1-p_H_ages)*T_I \
            + lQ_1[:, :, i]*p_H_ages*T_I
        lQ_0[:, :, i-1] = lQ_0[:, :, i]*(1. - T_q0) + lD[:, :, i]*mu_q_ages*T_q0
        lQ_1[:, :, i-1] = lQ_1[:, :, i]*(1. - T_q1) + lHw[:, :, i]*T_q1 + T_q1
        lHw[:, :, i-1] = lHw[:, :, i]*(1.-T_hw) + lHc[:, :, i]*p_c_ages*T_hw \
            + lD[:, :, i]*mu_w_ages*(1.-p_c_ages)*T_hw
        lHc[:, :, i-1] = lHc[:, :, i]*(1.-T_hc) + lHr[:, :, i]*(1.-mu_c_ages)*T_hc \
            + lD[:, :, i]*mu_c_ages*T_hc
        lHr[:, :, i-1] = lHr[:, :, i]*(1.-T_hr)
        dH[:, :, i] = -lS[:, :, i] + lSv[:, :, i]

    return dH


def ob_fun(x):
    logger = create_logger()
    start_obj = time.time()

    nuf = np.reshape(x, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, D_g, _, _ = sol(nuf, mob_av, beta, beta_gh, T,
                                          pop_hat, age_er, epidemic_npy, False)

    J = (D_g)

    elapsed_time = time.time() - start_obj
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    logger.info('Finished ob_fun function. Value: %s. Elapsed time: %s' % (J, elapsed_delta))

    return J


def der(x):
    logger = create_logger()
    start_der = time.time()

    nuf = np.reshape(x, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, _, _, _ = sol(nuf, mob_av, beta, beta_gh, T,
                                        pop_hat, age_er, epidemic_npy, False)
    # calculation of the gradient
    dH = back_int(S_g, S_vg, S_xg, L_g, beta_gh, beta, T, age_er, mob_av, pop_hat)

    dH2 = np.reshape(dH, (N_g*N_p*T))

    elapsed_time = time.time() - start_der
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    logger.info('Finished der function. Elapsed time: %s' % (elapsed_delta, ))

    return dH2


def bound_f(bound_full_orig, u_op):
    logger = create_logger()
    bound_r = np.reshape(bound_full_orig, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, D_g, _, _ = sol(u_op, mob_av, beta, beta_gh, T,
                                          pop_hat, age_er, epidemic_npy, False)

    kg_pairs = []
    for i in range(T):
        for g in range(N_g-1, -1, -1):
            for k in range(N_p):
                if S_g[g, k, i] <= 0:
                    if (g, k) not in kg_pairs:
                        logger.info('Found KG pair %s at time %s' % ((g, k), i))
                        bound_r[g, k, i-1] = S_g[g, k, i-1] - L_g[g, k, i-1]*S_g[g, k, i-1]
                        bound_r[g, k, i:T] = 0.0
                        kg_pairs.append((g, k))

    bound_rf = np.reshape(bound_r, N_g*N_p*T)

    return bound_rf, kg_pairs, D_g


def log_out_minimize(minimize_result):
    result_str = ('%(message)s\t(Exit mode %(status)s)\n'
                  '\tCurrent function value: %(value)s\n'
                  '\tIterations: %(iter)s\n'
                  '\tFunction evaluations: %(evals)s\n'
                  '\tGradient evaluations: %(grad)s\n') % ({
                    'message': minimize_result.message,
                    'status': minimize_result.status,
                    'value': minimize_result.fun,
                    'evals': minimize_result.nfev,
                    'grad': minimize_result.njev,
                    'iter': minimize_result.nit,
                  })
    return result_str


def optimize(epidemic_npy_complete):
    logger = create_logger()

    # number of optimization variables
    N_f = N_g*N_p

    n_max = 30000

    global epidemic_npy
    epidemic_npy = epidemic_npy_complete
    mult_age_er = age_er.T[:, :, np.newaxis]
    logger.debug('Current population:\n%s' % (epidemic_npy*mult_age_er, ))

    # Constraints
    Af = np.array([]).reshape(T, 0)
    for g in range(N_g):
        for k in range(N_p):
            kg_eye = np.eye(T)*age_er[k, g]
            Af = np.hstack((Af, kg_eye))

    b = n_max*np.ones(T)

    cons = {"type": "eq", "fun": lambda x:  Af @ x - b,
            'jac': lambda x: Af}

    bound0 = np.zeros(N_f*T)
    bound_full_orig = np.zeros(N_f*T)
    idx_t = 0
    for g in range(N_g):
        for k in range(N_p):
            low_idx = int(idx_t*T)
            up_idx = int((idx_t+1)*T)
            bound_full_orig[low_idx:up_idx] = n_max/age_er[k, g]
            idx_t += 1

    init_bounds = Bounds(bound0, bound_full_orig)

    logger.info('Constructed initial bounds.')

    x0 = np.zeros(N_f*T)
    kg_pairs = []

    minimize_iter = 1
    u_op = x0
    bounds = init_bounds
    last_values = np.array([np.inf])
    while True:
        start_iter = time.time()
        logger.info(('Starting minimize. Iteration: %s.\n'
                     'KG pairs: %s') % (minimize_iter, kg_pairs))
        res = minimize(ob_fun, u_op, method='SLSQP', jac=der,
                       constraints=[cons], options={'maxiter': 5},
                       bounds=bounds)

        logger.info('Finished minimize:\n%s\nLooking for KG pairs.' % (log_out_minimize(res), ))
        u_op = np.reshape(res.x, (N_g, N_p, T))
        bound_full, kg_pairs, D_g = bound_f(bound_full_orig, u_op)
        bounds = Bounds(bound0, bound_full)

        elapsed_time = time.time() - start_iter
        elapsed_delta = datetime.timedelta(seconds=elapsed_time)
        logger.info(('Finished minimize. Iteration: %s.\n'
                     'Elapsed time: %s\n'
                     'Last D_g values: %s\n'
                     'Current D_g value: %s') % (minimize_iter, elapsed_delta,
                                                 last_values[-3:], D_g))
        minimize_iter += 1

        if np.allclose(last_values[-3:], D_g):
            logger.info('Last iterations results converged, breaking.')
            break

        last_values = np.concatenate((last_values, [D_g]))

    D_g, u_op, new_epidemic_npy = sol(u_con=u_op,
                                      mob_av=mob_av,
                                      beta=beta,
                                      beta_gh=beta_gh,
                                      T=T,
                                      pop_hat=pop_hat,
                                      age_er=age_er,
                                      epidemic_npy=epidemic_npy,
                                      return_states=True)

    logger.info(('Finished iterations.\n'
                 'Final value: %s.\n'
                 'Final KG pairs: %s.') % (D_g, kg_pairs))
    logger.debug('Final population:\n%s' % (new_epidemic_npy*mult_age_er, ))

    return new_epidemic_npy, u_op, kg_pairs, D_g


def full_optimize(r, beta_sim, tau, time_horizon, init_time,
                  total_time, num_age_groups, num_regions):
    logger = create_logger()

    global beta
    beta = beta_sim

    global t0
    t0 = init_time

    global N_g
    N_g = num_age_groups

    global N_p
    N_p = num_regions

    logger.debug(('Getting parameters with:\n'
                  'beta: %s\n'
                  't0: %s\n'
                  'N_g: %s\n'
                  'N_p: %s') % (beta, t0, N_g, N_p))
    global mob_av, beta_gh, pop_hat, age_er
    mob_av, beta_gh, pop_hat, age_er, rho = get_model_parameters(
                                                number_age_groups=num_age_groups,
                                                num_regions=num_regions,
                                                init_vacc=True,
                                                t0=init_time,
                                                tau=tau
                                            )
    logger.info('Got model parameters.')

    # Reading CSV
    csv_name = 'out/epidemic_finland_9.csv'
    logger.debug('Reading %s for initial state.' % (csv_name, ))
    # Reading CSV
    epidemic_csv = pd.read_csv(csv_name)
    # Getting only date t0
    epidemic_zero = epidemic_csv.loc[epidemic_csv['date'] == t0, :]
    # Removing Ahvenanmaa or Aland
    epidemic_zero = epidemic_zero[~epidemic_zero['erva'].str.contains('land')]

    # Getting the order the ervas have inside the dataframe
    ervas_order = ['HYKS', 'TYKS', 'TAYS', 'KYS', 'OYS']
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
    age_er_ext = age_er.T[:, :, np.newaxis]

    # epidemic_npy has num_ervas first, compartmetns have age first
    # Transposing to epidemic_npy to accomodate to compartments
    epidemic_npy = epidemic_npy.transpose(1, 0, 2)

    # Dividing to get the proportion
    epidemic_npy = epidemic_npy/age_er_ext

    initial_epidemic_npy = np.zeros((N_g, N_p, 13))
    initial_epidemic_npy[:, :, :len(select_columns)] = epidemic_npy
    logger.info('Finished reading inital state.')

    global T
    T = time_horizon
    logger.info('Time horizon for optimization: %s' % (T, ))

    base_name = "R_%s_tau_%s_T_%s" % (r, tau, total_time)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    json_file = "%s.json" % (base_name, )
    json_file_path = os.path.join(dir_path, 'out', json_file)

    epidemic_npy_complete = initial_epidemic_npy
    json_save = {}
    time_done = 0
    u_total = np.array([]).reshape(N_g, N_p, 0)
    while time_done < total_time:
        logger.info('Starting optimize at time: %s/%s' % (time_done, total_time))
        epidemic_npy_complete, u_op, kg_pairs, D_g = optimize(epidemic_npy_complete)
        time_done += time_horizon
        logger.info('Finished optimization, saving results.')

        u_total = np.concatenate((u_total, u_op), axis=2)
        
        json_save[time_done] = {}
        u_op_filename = '%s_%s_u_op.npy' % (base_name, time_done)
        epidemic_npy_filename = '%s_%s_epidemic.npy' % (base_name, time_done)
        
        u_op_file_path = os.path.join(dir_path, 'out', u_op_filename)
        epidemic_file_path = os.path.join(dir_path, 'out', epidemic_npy_filename)

        json_save[time_done]['u_op'] = u_op_file_path
        json_save[time_done]['epidemic'] = epidemic_file_path
        json_save[time_done]['D_g'] = D_g
        json_save[time_done]['kg_pairs'] = kg_pairs

        np.save(u_op_file_path, u_op)
        np.save(epidemic_file_path, epidemic_npy_complete)
        logger.info('File written to: %s and %s' % (u_op_file_path, epidemic_file_path))

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_save, f, indent=2)

    logger.info('Finished with total time, obtaining final results.')

    D_g, u_op, new_epidemic_npy = sol(u_con=u_total,
                                      mob_av=mob_av,
                                      beta=beta,
                                      beta_gh=beta_gh,
                                      T=total_time,
                                      pop_hat=pop_hat,
                                      age_er=age_er,
                                      epidemic_npy=initial_epidemic_npy,
                                      return_states=True)
    u_op_filename = '%s_u_op.npy' % (base_name, )
    u_op_file_path = os.path.join(dir_path, 'out', u_op_filename)
    initial_epi_filename = '%s_initial_epidemic.npy' % (base_name, )
    initial_epi_file_path = os.path.join(dir_path, 'out', initial_epi_filename)
    final_epi_filename = '%s_final_epidemic.npy' % (base_name, )
    final_epi_file_path = os.path.join(dir_path, 'out', final_epi_filename)
    np.save(u_op_file_path, u_op)
    np.save(initial_epi_file_path, initial_epidemic_npy)
    np.save(final_epi_file_path, new_epidemic_npy)

    json_save['initial_epidemic'] = initial_epi_file_path
    json_save['u_op'] = u_op_file_path
    json_save['final_epidemic'] = final_epi_file_path
    json_save['D_g'] = D_g
    json_save['kg_pairs'] = kg_pairs

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_save, f, indent=2)

    logger.info(('Final results obtained.\n'
                 'Complete D_g: %s.\n'
                 'Final KG pairs: %s.\n'
                 'Final shape u_op: %s.\n'
                 'JSON file: %s.\n'
                 'Final populations:\n%s') % (D_g, kg_pairs, u_op.shape,
                                              json_file_path,
                                              new_epidemic_npy*age_er_ext))

    return json_file_path


def run_optimize(r, tau, beta_sim, time_horizon, init_time, total_time):
    logger = create_logger()
    try:
        start_time = time.time()
        logger.info('Starting. R: %s. Tau: %s. T: %s. T0: %s' % (r,
                                                                 tau,
                                                                 total_time,
                                                                 init_time))

        filename = full_optimize(r=r,
                                 beta_sim=beta_sim,
                                 tau=tau,
                                 time_horizon=time_horizon,
                                 init_time=init_time,
                                 total_time=total_time,
                                 num_age_groups=9,
                                 num_regions=5)

        elapsed_time = time.time() - start_time
        elapsed_delta = datetime.timedelta(seconds=elapsed_time)
        logger.info('Finished. R: %s. Tau: %s. T: %s. T0: %s. Time: %s' % (r,
                                                                           tau,
                                                                           total_time,
                                                                           init_time,
                                                                           elapsed_delta))

        return filename
    except Exception:
        logger.exception(('Something went wrong in optimization.\n'
                          'R: %s. Tau: %s. T: %s. T0: %s') % (r,
                                                              tau,
                                                              total_time,
                                                              init_time))
        return None


def create_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(processName)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S %p'
    )
    handler_file = logging.handlers.RotatingFileHandler(
                    'optimized_vaccination.log',
                    maxBytes=1e6,
                    backupCount=3
                    )
    handler_console = logging.StreamHandler()
    handler_file.setFormatter(formatter)
    handler_console.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler_file)
    
    if len(logger.handlers) == 1:
        logger.addHandler(handler_console)

    return logger


def run_parallel_optimizations():
    logger = create_logger()
    logger.info('Logger for optimized_vaccination configured.')
    # all_experiments = [
    #     (0.75,  0.,  0.016577192790495632),
    #     (0.75,  0.5,  0.017420081058752156),
    #     (0.75,  1.0,  0.017799005077907416),
    #     (1.0,  0.,  0.022102923720660844),
    #     (1.0,  0.5,  0.023226774745002877),
    #     (1.0,  1.0,  0.023732006770543223),
    #     (1.25,  0.,  0.027628654650826055),
    #     (1.25,  0.5,  0.029033468431253595),
    #     (1.25,  1.0,  0.02966500846317903),
    #     (1.5,  0.,  0.033154385580991264),
    #     (1.5,  0.5,  0.03484016211750431),
    #     (1.5,  1.0,  0.03559801015581483),
    # ]
    all_experiments = [
        (1.5,  0.5,  0.03484016211750431, 10, '2021-04-18', 25),
    ]
    logger.info('optimized_vaccination experiments:\n%s' % (all_experiments, ))

    num_cpus = os.cpu_count()
    start_time = time.time()
    num_experiments = len(all_experiments)
    result_filenames = []
    logger.info('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute simulations in asynchronous way
        async_res = [pool.apply_async(func=run_optimize,
                                      args=(r, tau, beta_sim,
                                            time_horizon, init_time, total_time))
                     for r, tau, beta_sim,
                     time_horizon, init_time, total_time in all_experiments]

        # Waiting for the values of the async execution
        for res in async_res:
            filename = res.get()
            result_filenames.append(filename)

    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    logger.info('Finished experiments. Elapsed: %s' % (elapsed_delta, ))
    logger.info('Resulting filenames: %s' % (result_filenames, ))


if __name__ == "__main__":
    run_parallel_optimizations()
