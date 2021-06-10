import os
import time
import datetime
import json
import logging
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from utils_optimized_vaccination import (
    log_out_minimize, create_logger, parse_args
)
from env_var import EPIDEMIC
from forward_integration import get_model_parameters, read_initial_values


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

        u[:, :, j] = np.minimum(u_con[:, :, j], np.maximum(0.0, S_g[:, :, j] - L_g[:, :, j]*S_g[:, :, j]))
        S_g[:, :, j+1] = S_g[:, :, j] - L_g[:, :, j]*S_g[:, :, j] - u[:, :, j]
        S_vg[:, :, j+1] = S_vg[:, :, j] - L_g[:, :, j]*S_vg[:, :, j] + u[:, :, j] - T_V*S_vg[:, :, j]
        S_xg[:, :, j+1] = S_xg[:, :, j] - L_g[:, :, j]*S_xg[:, :, j] + (1.-alpha*e)*T_V*S_vg[:, :, j]
        V_g[:, :, j+1] = V_g[:, :, j] + alpha*e*T_V*S_vg[:, :, j]
        E_g[:, :, j+1] = E_g[:, :, j] + L_g[:, :, j]*(S_g[:, :, j]+S_vg[:, :, j] + S_xg[:, :, j]) - T_E*E_g[:, :, j]
        I_g[:, :, j+1] = I_g[:, :, j] + T_E*E_g[:, :, j] - T_I*I_g[:, :, j]
        Q_0g[:, :, j+1] = Q_0g[:, :, j] + (1-p_H_ages)*T_I*I_g[:, :, j] - T_q0*Q_0g[:, :, j]
        Q_1g[:, :, j+1] = Q_1g[:, :, j] + p_H_ages*T_I*I_g[:, :, j] - T_q1*Q_1g[:, :, j]
        H_wg[:, :, j+1] = H_wg[:, :, j] + T_q1*Q_1g[:, :, j] - T_hw*H_wg[:, :, j]
        H_cg[:, :, j+1] = H_cg[:, :, j] + p_c_ages*T_hw*H_wg[:, :, j] - T_hc*H_cg[:, :, j]
        H_rg[:, :, j+1] = H_rg[:, :, j] + (1-mu_c_ages)*T_hc*H_cg[:, :, j] - T_hr*H_rg[:, :, j]
        R_g[:, :, j+1] = R_g[:, :, j] + T_hr*H_rg[:, :, j] + (1-mu_w_ages)*(1-p_c_ages)*T_hw*H_wg[:, :, j] + (1-mu_q_ages)*T_q0*Q_0g[:, :, j]
        D_g[:, :, j+1] = D_g[:, :, j] + mu_q_ages*T_q0*Q_0g[:, :, j] + mu_w_ages*(1-p_c_ages)*T_hw*H_wg[:, :, j] + mu_c_ages*T_hc*H_cg[:, :, j]

        if hosp_optim:
            D_d[:, :, j] = T_q1*Q_1g[:, :, j]
        else:
            D_d[:, :, j+1] = mu_q_ages*T_q0*Q_0g[:, :, j] + mu_w_ages*(1-p_c_ages)*T_hw*H_wg[:, :, j] + mu_c_ages*T_hc*H_cg[:, :, j]

    # Filling the values at the last time step
    infect_mobility = (I_g[:, :, T-1]*age_er_t)@mobility_term.T
    lambda_g = beta_gh.T@infect_mobility
    L_g[:, :, T-1] = beta*lambda_g
    u[:, :, T-1] = np.minimum(u_con[:, :, T-1], np.maximum(0.0, S_g[:, :, T-1] - L_g[:, :, T-1]*S_g[:, :, T-1]))
    
    if hosp_optim:
        D_d[:, :, T-1] = T_q1*Q_1g[:, :, T-1]

    D_d = D_d*age_er_t[:, :, np.newaxis]

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

    final_susc = new_epidemic_npy[:, :, 0]
    empty_susc = np.where(np.isclose(final_susc, 0, atol=1e-2))
    ages = empty_susc[0]
    regions = empty_susc[1]
    kg_pairs = [(k.item(), g.item()) for k, g in zip(ages, regions)]

    return D_d.sum(), u, new_epidemic_npy, kg_pairs


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
        
        if hosp_optim:
            lQ_1[:, :, i-1] = lQ_1[:, :, i]*(1. - T_q1) + lHw[:, :, i]*T_q1 + T_q1
        else:
            lQ_1[:, :, i-1] = lQ_1[:, :, i]*(1. - T_q1) + lHw[:, :, i]*T_q1
            lD[:, :, i-1] = lD[:, :, i] + 1.
        
        lHw[:, :, i-1] = lHw[:, :, i]*(1.-T_hw) + lHc[:, :, i]*p_c_ages*T_hw \
            + lD[:, :, i]*mu_w_ages*(1.-p_c_ages)*T_hw
        lHc[:, :, i-1] = lHc[:, :, i]*(1.-T_hc) + lHr[:, :, i]*(1.-mu_c_ages)*T_hc \
            + lD[:, :, i]*mu_c_ages*T_hc
        lHr[:, :, i-1] = lHr[:, :, i]*(1.-T_hr)
        dH[:, :, i] = -lS[:, :, i] + lSv[:, :, i]

    return dH


def ob_fun(x):
    logger = create_logger(log_file, log_level)
    start_obj = time.time()

    nuf = np.reshape(x, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, D_d, _, _ = sol(nuf, mob_av, beta, beta_gh, T,
                                          pop_hat, age_er, epidemic_npy, False)

    J = (D_d)

    elapsed_time = time.time() - start_obj
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    logger.info('ob_fun function. Value: %s. Elapsed time: %s' % (J, elapsed_delta))

    return J


def der(x):
    logger = create_logger(log_file, log_level)
    start_der = time.time()

    nuf = np.reshape(x, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, _, _, _ = sol(nuf, mob_av, beta, beta_gh, T,
                                        pop_hat, age_er, epidemic_npy, False)
    # calculation of the gradient
    dH = back_int(S_g, S_vg, S_xg, L_g, beta_gh, beta, T, age_er, mob_av, pop_hat)

    dH2 = np.reshape(dH, (N_g*N_p*T))

    elapsed_time = time.time() - start_der
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    logger.info('der function. Elapsed time: %s' % (elapsed_delta, ))

    return dH2


def bound_f(bound_full_orig, u_op):
    logger = create_logger(log_file, log_level)
    bound_r = np.reshape(bound_full_orig, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, D_d, _, _ = sol(u_op, mob_av, beta, beta_gh, T,
                                          pop_hat, age_er, epidemic_npy, False)

    kg_pairs = []
    for i in range(T):
        for g in range(N_g-1, -1, -1):
            for k in range(N_p):
                # If Susc below 0 or close to 0
                if S_g[g, k, i] <= 0 or np.isclose(S_g[g, k, i], 0, atol=1e-2):
                    if (g, k) not in kg_pairs:
                        logger.info('Found KG pair %s at time %s' % ((g, k), i))
                        bound_r[g, k, i:] = 0.0
                        kg_pairs.append((g, k))

    bound_rf = np.reshape(bound_r, N_g*N_p*T)

    return bound_rf, kg_pairs, D_d


def optimize(epidemic_npy_complete, max_execution_hours):
    logger = create_logger(log_file, log_level)

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
    bound_full_orig = bound_full_orig.reshape(N_g, N_p, T)
    # Do not vaccinate two first age groups
    bound_full_orig[:2, :, :] = 0
    bound_full_orig = bound_full_orig.reshape(N_f*T)

    init_bounds = Bounds(bound0, bound_full_orig)

    logger.info('Constructed initial bounds.')

    x0 = np.zeros(N_f*T)
    kg_pairs = []

    minimize_iter = 1
    # Optimization variables
    u_op = x0
    bounds = init_bounds
    last_values = np.array([np.inf])
    # Time variables
    start_optim = time.time()
    max_time_secs = max_execution_hours*3600
    while True:
        start_iter = time.time()
        logger.info(('Starting minimize. Iteration: %s.\n'
                     'KG pairs: %s') % (minimize_iter, kg_pairs))
        res = minimize(ob_fun, u_op, method='SLSQP', jac=der,
                       constraints=[cons], options={'maxiter': 5},
                       bounds=bounds)

        logger.info('minimize done:\n%s\nLooking for KG pairs.' % (log_out_minimize(res), ))
        u_op = np.reshape(res.x, (N_g, N_p, T))
        bound_full, kg_pairs, D_d = bound_f(bound_full_orig.copy(), u_op)
        bounds = Bounds(bound0, bound_full)

        elapsed_time = time.time() - start_iter
        elapsed_delta = datetime.timedelta(seconds=elapsed_time)
        logger.info(('Finished minimize. Iteration: %s.\n'
                     'Elapsed time: %s\n'
                     'KG pairs: %s\n'
                     'Last D_d values: %s\n'
                     'Current D_d value: %s') % (minimize_iter, elapsed_delta,
                                                 kg_pairs,
                                                 last_values[-3:], D_d))
        minimize_iter += 1
        
        elapsed_optim = time.time() - start_optim
        elapsed_delta_optim = datetime.timedelta(seconds=elapsed_optim)
        logger.info('Elapsed time in optimization: %s' % (elapsed_delta_optim, ))
        if elapsed_delta_optim.total_seconds() > max_time_secs:
            logger.warning('Breaking optimization because of exceeded time.')
            break

        # Tolerance to 3 decimal places
        if np.allclose(last_values[-3:], D_d, atol=1e-3):
            logger.info('Last iterations results converged, breaking.')
            break

        last_values = np.concatenate((last_values, [D_d]))

    D_d, u_op, new_epidemic_npy, kg_pairs = sol(u_con=u_op,
                                                mob_av=mob_av,
                                                beta=beta,
                                                beta_gh=beta_gh,
                                                T=T,
                                                pop_hat=pop_hat,
                                                age_er=age_er,
                                                epidemic_npy=epidemic_npy,
                                                return_states=True)
    logger.info(('Finished iterations.\n'
                 'value: %s.\n'
                 'KG pairs: %s.') % (D_d, kg_pairs))
    logger.debug('Population:\n%s' % (new_epidemic_npy*mult_age_er, ))
    u_op_day = u_op*mult_age_er
    u_op_day = u_op_day.sum(axis=(0, 1))
    logger.debug('Vaccination/day:\n%s' % (u_op_day, ))

    return new_epidemic_npy, u_op, kg_pairs, D_d


def full_optimize(r, tau, time_horizon, init_time, max_execution_hours,
                  total_time, num_age_groups, region, hosp_optim_in,
                  results_folder):
    logger = create_logger(log_file, log_level)

    global t0
    t0 = init_time

    global N_g
    N_g = num_age_groups

    global hosp_optim
    hosp_optim = hosp_optim_in

    logger.debug(('Getting parameters with:\n'
                  'R: %s\n'
                  'tau: %s\n'
                  't0: %s\n'
                  'N_g: %s\n'
                  'Region: %s') % (r, tau, t0, N_g, region))
    global mob_av, beta_gh, pop_hat, age_er
    mob_av, beta_gh, pop_hat, age_er, rho = get_model_parameters(
                                                number_age_groups=num_age_groups,
                                                region=region,
                                                init_vacc=True,
                                                t0=init_time,
                                                tau=tau
                                            )
    num_regions, _ = age_er.shape
    global N_p
    N_p = num_regions
    logger.info('Number of regions: %s' % (num_regions, ))

    global beta
    beta = r/rho

    logger.info('Got model parameters.\nBeta: %s' % (beta, ))

    # Adding 1 dimension to age_er to do data manipulation
    age_er_ext = age_er.T[:, :, np.newaxis]

    epidemic_npy = read_initial_values(age_er=age_er,
                                       region=region,
                                       init_vacc=True,
                                       t0=t0)
    _, _, columns = epidemic_npy.shape

    initial_epidemic_npy = np.zeros((N_g, N_p, 13))
    initial_epidemic_npy[:, :, :columns] = epidemic_npy
    logger.info('Initial state read.')

    global T
    T = time_horizon
    logger.info('Time horizon for optimizations: %s' % (T, ))

    base_name = '%s_R_%s_tau_%s_t0_%s_T_%s' % (region, r, tau, t0, total_time)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_path = os.path.join(dir_path, 'out', results_folder)
    os.makedirs(results_path, exist_ok=True)

    json_file = "%s.json" % (base_name, )
    json_file_path = os.path.join(results_path, json_file)

    epidemic_npy_complete = initial_epidemic_npy
    json_save = {}
    time_done = 0
    u_total = np.array([]).reshape(N_g, N_p, 0)
    while time_done < total_time:
        logger.info('Starting optimize at time: %s/%s' % (time_done, total_time))
        epidemic_npy_complete, u_op, kg_pairs, D_d = optimize(epidemic_npy_complete,
                                                              max_execution_hours)
        time_done += time_horizon
        logger.info('Finished optimization, saving results.')

        u_total = np.concatenate((u_total, u_op), axis=2)
        
        json_save[time_done] = {}
        u_op_filename = '%s_%s_u_op.npy' % (base_name, time_done)
        epidemic_npy_filename = '%s_%s_epidemic.npy' % (base_name, time_done)
        
        u_op_file_path = os.path.join(results_path, u_op_filename)
        epidemic_file_path = os.path.join(results_path, epidemic_npy_filename)

        json_save[time_done]['u_op'] = u_op_file_path
        json_save[time_done]['epidemic'] = epidemic_file_path
        json_save[time_done]['D_d'] = D_d
        json_save[time_done]['kg_pairs'] = kg_pairs

        np.save(u_op_file_path, u_op)
        np.save(epidemic_file_path, epidemic_npy_complete)
        logger.info('File written to: %s and %s' % (u_op_file_path, epidemic_file_path))

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_save, f, indent=2)

    logger.info('All times finished, obtaining final results.')

    u_op_day = u_total*age_er_ext
    u_op_day = u_op_day.sum(axis=(0, 1))
    logger.debug('u_total vaccination/day:\n%s' % (u_op_day, ))

    D_d, u_op, new_epidemic_npy, kg_pairs = sol(u_con=u_total,
                                                mob_av=mob_av,
                                                beta=beta,
                                                beta_gh=beta_gh,
                                                T=total_time,
                                                pop_hat=pop_hat,
                                                age_er=age_er,
                                                epidemic_npy=initial_epidemic_npy,
                                                return_states=True)
    u_op_filename = '%s_u_op.npy' % (base_name, )
    u_op_file_path = os.path.join(results_path, u_op_filename)
    initial_epi_filename = '%s_initial_epidemic.npy' % (base_name, )
    initial_epi_file_path = os.path.join(results_path, initial_epi_filename)
    final_epi_filename = '%s_final_epidemic.npy' % (base_name, )
    final_epi_file_path = os.path.join(results_path, final_epi_filename)
    np.save(u_op_file_path, u_op)
    np.save(initial_epi_file_path, initial_epidemic_npy)
    np.save(final_epi_file_path, new_epidemic_npy)

    json_save['initial_epidemic'] = initial_epi_file_path
    json_save['u_op'] = u_op_file_path
    json_save['final_epidemic'] = final_epi_file_path
    json_save['D_d'] = D_d
    json_save['r'] = r
    json_save['tau'] = tau
    json_save['t0'] = t0
    json_save['T'] = T
    json_save['total_time'] = total_time
    json_save['kg_pairs'] = kg_pairs

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_save, f, indent=2)

    logger.info(('Final results obtained.\n'
                 'Complete D_d: %s.\n'
                 'Final KG pairs: %s.\n'
                 'Final shape u_op: %s.\n'
                 'JSON file: %s.') % (D_d, kg_pairs, u_op.shape,
                                      json_file_path))
    logger.debug('Final populations:\n%s' % (new_epidemic_npy*age_er_ext, ))
    u_op_day = u_op*age_er_ext
    u_op_day = u_op_day.sum(axis=(0, 1))
    logger.debug('Final vaccination/day:\n%s' % (u_op_day, ))

    return json_file_path


def run_optimize(r, tau, time_horizon, init_time, total_time, log_level_in,
                 hosp_optim, max_execution_hours, log_file_in, region,
                 num_age_groups, results_folder):
    multiprocessing.current_process().name = 'Worker-%s-R_%s-Tau_%s' % (region, r, tau)
    global log_level
    log_level = log_level_in
    global log_file
    log_file = log_file_in
    logger = create_logger(log_file, log_level)
    try:
        start_time = time.time()
        logger.info('Starting. R: %s. Tau: %s. T: %s. T0: %s' % (r,
                                                                 tau,
                                                                 total_time,
                                                                 init_time))

        filename = full_optimize(r=r,
                                 tau=tau,
                                 time_horizon=time_horizon,
                                 init_time=init_time,
                                 total_time=total_time,
                                 hosp_optim_in=hosp_optim,
                                 num_age_groups=num_age_groups,
                                 region=region,
                                 results_folder=results_folder,
                                 max_execution_hours=max_execution_hours)

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


def run_parallel_optimizations(r_experiments, taus, num_age_groups, region,
                               init_time, total_time, time_horizon, hosp_optim,
                               results_folder, log_file, log_level_str,
                               max_execution_hours):
    log_level = getattr(logging, log_level_str, None)
    logger = create_logger(log_file, log_level)
    logger.info('Logger for optimized_vaccination configured.')

    all_experiments = []
    for tau in taus:
        for r in r_experiments:
            all_experiments.append((r, tau))
    logger.info(('Script parameters:\n'
                 'R_effs: %(rs)s\n'
                 'Taus: %(taus)s\n'
                 'T0: %(t0)s\n'
                 'T: %(T)s\n'
                 'part_time: %(part_time)s\n'
                 'Region: %(region)s\n'
                 'Number of age groups: %(num_age_groups)s\n'
                 'Hospitalization optimized: %(hosp_optim)s\n'
                 'all_experiments: %(all_experiments)s\n'
                 'Results folder: %(results_folder)s\n'
                 'Log file: %(log_file)s\n'
                 'Log level: %(log_level)s\n'
                 'Max execution (hours): %(max_exec)s') % {
                    'rs': r_experiments,
                    'taus': taus,
                    't0': init_time,
                    'T': total_time,
                    'part_time': time_horizon,
                    'num_age_groups': num_age_groups,
                    'region': region,
                    'hosp_optim': hosp_optim,
                    'all_experiments': all_experiments,
                    'log_level': log_level_str,
                    'log_file': log_file,
                    'results_folder': results_folder,
                    'max_exec': max_execution_hours
                })

    num_cpus = os.cpu_count()
    start_time = time.time()
    num_experiments = len(all_experiments)
    result_filenames = []
    logger.info('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute simulations in asynchronous way
        async_res = [
                        pool.apply_async(
                            func=run_optimize,
                            kwds={
                                'r': r,
                                'tau': tau,
                                'time_horizon': time_horizon,
                                'region': region,
                                'num_age_groups': num_age_groups,
                                'init_time': init_time,
                                'total_time': total_time,
                                'log_level_in': log_level,
                                'hosp_optim': hosp_optim,
                                'max_execution_hours': max_execution_hours,
                                'log_file_in': log_file,
                                'results_folder': results_folder
                            })
                        for r, tau in all_experiments
                    ]

        # Waiting for the values of the async execution
        for res in async_res:
            filename = res.get()
            result_filenames.append(filename)

    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    logger.info('Finished experiments. Elapsed: %s' % (elapsed_delta, ))
    logger.info('Resulting filenames: %s' % (result_filenames, ))


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        run_parallel_optimizations(
            time_horizon=10,
            init_time='2021-04-18',
            total_time=25,
            taus=[0.5],
            r_experiments=[1.5],
            hosp_optim=False,
            region='erva',
            num_age_groups=9,
            results_folder='foo',
            log_file=args.log_file,
            log_level_str=args.log_level,
            max_execution_hours=args.max_execution_hours
        )
    else:
        run_parallel_optimizations(
            time_horizon=args.part_time,
            init_time=args.t0,
            total_time=args.T,
            taus=args.taus,
            r_experiments=args.r_experiments,
            hosp_optim=args.hosp_optim,
            region=args.region,
            num_age_groups=args.num_age_groups,
            results_folder=args.results_folder,
            log_file=args.log_file,
            log_level_str=args.log_level,
            max_execution_hours=args.max_execution_hours
        )
