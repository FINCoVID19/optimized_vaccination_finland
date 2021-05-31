import os
import time
import datetime
import json
import logging
from multiprocessing import Pool
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
import pandas as pd
from env_var import EPIDEMIC
from forward_integration import get_model_parameters


def sol(u_con, mob_av, beta, beta_gh, T, pop_hat, age_er, epidemic_npy):
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

    return S_g, S_vg, S_xg, L_g, D_d.sum(), V_d, u


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
    start_obj = time.time()

    nuf = np.reshape(x, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, D_g, _, _ = sol(nuf, mob_av, beta, beta_gh, T,
                                          pop_hat, age_er, epidemic_npy)

    J = (D_g)

    elapsed_time = time.time() - start_obj
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished ob_fun function. Value: %s. Elapsed time: %s' % (J, elapsed_delta))

    return J


def der(x):
    start_der = time.time()

    nuf = np.reshape(x, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, _, _, _ = sol(nuf, mob_av, beta, beta_gh, T,
                                        pop_hat, age_er, epidemic_npy)
    # calculation of the gradient
    dH = back_int(S_g, S_vg, S_xg, L_g, beta_gh, beta, T, age_er, mob_av, pop_hat)

    dH2 = np.reshape(dH, (N_g*N_p*T))

    elapsed_time = time.time() - start_der
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished der function. Elapsed time: %s' % (elapsed_delta, ))

    return dH2


def bound_f(bound_full_orig, u_op):
    bound_r = np.reshape(bound_full_orig, (N_g, N_p, T))
    S_g, S_vg, S_xg, L_g, D_g, _, _ = sol(u_op, mob_av, beta, beta_gh, T,
                                          pop_hat, age_er, epidemic_npy)

    kg_pairs = []
    for i in range(T):
        for g in range(N_g-1, -1, -1):
            for k in range(N_p):
                if S_g[g, k, i] <= 0:
                    if (g, k) not in kg_pairs:
                        print('Found KG pair %s at time %s' % ((g, k), i))
                        bound_r[g, k, i-1] = S_g[g, k, i-1] - L_g[g, k, i-1]*S_g[g, k, i-1]
                        bound_r[g, k, i:T] = 0.0
                        kg_pairs.append((g, k))

    bound_rf = np.reshape(bound_r, N_g*N_p*T)

    return bound_rf, kg_pairs, D_g


def optimize(file_npy, file_json, epidemic_npy_complete):
    # number of optimization variables
    N_f = N_g*N_p

    n_max = 30000

    global epidemic_npy
    epidemic_npy = epidemic_npy_complete

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

    print('Constructed initial bounds.')

    x0 = np.zeros(N_f*T)
    kg_pairs = []

    minimize_iter = 1
    u_op = x0
    bounds = init_bounds
    last_value = np.inf
    while True:
        print(('Starting minimize %d iteration.\n'
               'KG pairs: %s') % (minimize_iter, kg_pairs))
        res = minimize(ob_fun, u_op, method='SLSQP', jac=der,
                       constraints=[cons], options={'maxiter': 5, 'disp': True},
                       bounds=bounds)

        print('Finished minimize, looking for KG pairs.')
        u_op = np.reshape(res.x, (N_g, N_p, T))
        bound_full, kg_pairs, D_g = bound_f(bound_full_orig, u_op)
        bounds = Bounds(bound0, bound_full)

        print(('Finished minimize %d iteration.\n'
               'Last D_g value: %s\n'
               'Current D_g value: %s') % (minimize_iter, last_value, D_g))
        minimize_iter += 1

        if np.isclose(last_value, D_g):
            print('Last iteration results converged, breaking.')
            break

        last_value = D_g

    print('Finished iterations. Final KG pairs: %s' % (kg_pairs, ))
    json_save = {
        'kg_pairs': kg_pairs
    }

    with open(file_json, 'w', encoding='utf-8') as f:
        json.dump(json_save, f, indent=2)

    np.save(file_npy, u_op)
    print('File written to: %s and %s' % (file_npy, file_json))


def full_optimize(beta_sim, tau, file_npy, file_json, time_horizon, init_time,
                  num_age_groups, num_regions):
    global beta
    beta = beta_sim

    global t0
    t0 = init_time

    global N_g
    N_g = num_age_groups

    global N_p
    N_p = num_regions

    global T
    T = time_horizon

    global mob_av, beta_gh, pop_hat, age_er
    mob_av, beta_gh, pop_hat, age_er, rho = get_model_parameters(
                                                number_age_groups=num_age_groups,
                                                num_regions=num_regions,
                                                init_vacc=True,
                                                t0=init_time,
                                                tau=tau
                                            )
    print('Got model parameters.')

    # Reading CSV
    csv_name = 'out/epidemic_finland_9.csv'
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
    age_er_div = age_er[:, :, np.newaxis]
    # Dividing to get the proportion
    epidemic_npy = epidemic_npy/age_er_div

    # epidemic_npy has num_ervas first, compartmetns have age first
    # Transposing to epidemic_npy to accomodate to compartments
    epidemic_npy = epidemic_npy.transpose(1, 0, 2)

    epidemic_npy_complete = np.zeros((N_g, N_p, 13))
    epidemic_npy_complete[:, :, :len(select_columns)] = epidemic_npy
    print('Finished reading inital state.')

    optimize(file_npy, file_json, epidemic_npy_complete)


def run_optimize(r, tau, beta_sim, time_horizon, init_time):
    filename = "R_%s_tau_%s_T_%s" % (r, tau, time_horizon)
    npy_filename = filename + '.npy'
    json_filename = filename + '.json'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    npy_file_path = os.path.join(dir_path, 'out', npy_filename)
    json_file_path = os.path.join(dir_path, 'out', json_filename)

    try:
        start_time = time.time()
        proc_number = os.getpid()
        print('Starting (%s). R: %s. Tau: %s. T: %s. T0: %s' % (proc_number,
                                                                r,
                                                                tau,
                                                                time_horizon,
                                                                init_time))

        full_optimize(beta_sim=beta_sim,
                      tau=tau,
                      file_npy=npy_file_path,
                      file_json=json_file_path,
                      time_horizon=time_horizon,
                      init_time=init_time,
                      num_age_groups=9,
                      num_regions=5)

        elapsed_time = time.time() - start_time
        elapsed_delta = datetime.timedelta(seconds=elapsed_time)
        print('Finished (%s). R: %s. Tau: %s. T: %s. T0: %s. Time: %s' % (proc_number,
                                                                          r,
                                                                          tau,
                                                                          time_horizon,
                                                                          init_time,
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
        (1.5,  0.5,  0.03484016211750431, 20, '2021-04-18'),
    ]
    num_cpus = os.cpu_count()
    start_time = time.time()
    num_experiments = len(all_experiments)
    result_filenames = []
    print('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute simulations in asynchronous way
        async_res = [pool.apply_async(func=run_optimize,
                                      args=(r, tau, beta_sim, time_horizon, init_time))
                     for r, tau, beta_sim, time_horizon, init_time in all_experiments]

        # Waiting for the values of the async execution
        for res in async_res:
            filename = res.get()
            result_filenames.append(filename)
    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished experiments. Elapsed: %s' % (elapsed_delta, ))
    print('Resulting filenames: %s' % (result_filenames, ))


if __name__ == "__main__":
    run_parallel_optimizations()
