from forward_integration import (
    forward_integration, get_model_parameters, read_initial_values
)
import numpy as np
from multiprocessing import Pool
import os
import time
import datetime


def get_experiments_results(num_age_groups, num_ervas, e, taus, u_offset,
                            init_vacc, strategies, u, T, r_experiments, t0):
    experiments_params = {
        'num_ervas': num_ervas,
        'num_age_groups': num_age_groups,
        'u': u,
        't0': t0,
        'T': T,
        'e': e,
        'r_experiments': r_experiments,
        'init_vacc': init_vacc,
        'taus': taus,
        'strategies': strategies
    }
    print(('Beginning experiments.\n'
           'Parameters:\n'
           'Number of age ervas: {num_ervas}.\n'
           'Number of age groups: {num_age_groups}.\n'
           'Number of vaccines per day: {u}.\n'
           't0: {t0}.\n'
           'T: {T}.\n'
           'Vaccine efficacy (e): {e}.\n'
           'Rs to try: {r_experiments}.\n'
           'Taus to try: {taus}.\n'
           'Initialize with vaccinated people: {init_vacc}.\n'
           'Strategies:\n{strategies}.\n').format(**experiments_params))

    tau_params = {tau: {} for tau in taus}
    for tau in taus:
        mob_av, beta_gh, pop_erva_hat, age_er, rho = get_model_parameters(num_age_groups,
                                                                          num_ervas,
                                                                          init_vacc,
                                                                          t0,
                                                                          tau)
        tau_params[tau]['mob_av'] = mob_av
        tau_params[tau]['beta_gh'] = beta_gh
        tau_params[tau]['pop_erva_hat'] = pop_erva_hat
        tau_params[tau]['rho'] = rho

    epidemic_npy = read_initial_values(age_er, init_vacc, t0)

    age_er_prop = age_er.T
    age_er_prop = age_er_prop[:, :, np.newaxis]

    experiments = {r: {tau: {} for tau in taus} for r in r_experiments}
    num_experiments = 0
    for tau in taus:
        for r in r_experiments:
            beta = r/tau_params[tau]['rho']
            for ws, label in strategies:
                # ws is None mean go to optimized strategy
                if type(ws) is not list:
                    dir_path = os.path.dirname(os.path.realpath(__file__))
                    u_op_file = "%ssol_tau%s_deathoptim%s.npy" % (r, tau, ws)
                    u_op_file_path = os.path.join(dir_path, 'out', u_op_file)
                    if os.path.isfile(u_op_file_path):
                        exec_experiment = True
                        print('Found file: %s' % (u_op_file_path, ))
                    else:
                        print('File not found: %s' % (u_op_file_path, ))
                        u_op_file_path = None
                        exec_experiment = False
                    # Forcing T to be 110 to match optimized files
                    T_forward = T - u_offset
                else:
                    exec_experiment = True
                    u_op_file_path = None
                    T_forward = T

                if exec_experiment:
                    num_experiments += 1
                    parameters = {
                        'u_con': u,
                        'c1': tau_params[tau]['mob_av'],
                        'beta': beta,
                        'c_gh': tau_params[tau]['beta_gh'],
                        'T': T_forward,
                        'pop_hat': tau_params[tau]['pop_erva_hat'],
                        'age_er': age_er,
                        't0': t0,
                        'ws_vacc': ws,
                        'e': e,
                        'init_vacc': init_vacc,
                        'epidemic_npy': epidemic_npy,
                        'u_op_file': u_op_file_path,
                        'num_exp': num_experiments,
                        'r': r,
                        'tau': tau,
                        'label': label
                    }
                    experiments[r][tau][label] = {}
                    experiments[r][tau][label]['parameters'] = parameters

    num_cpus = os.cpu_count()
    start_time = time.time()
    print('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute forward simulation in asynchronous way
        async_res = []
        for r, r_level in experiments.items():
            for tau, tau_level in r_level.items():
                for label, label_level in tau_level.items():
                    policy_params = label_level['parameters']
                    async_res.append(
                        pool.apply_async(execute_parallel_forward,
                                         kwds=policy_params)
                    )

        # Waiting for the values of the async execution
        for res in async_res:
            r, tau, label, results = res.get()
            # Adding the results to our dictionary
            experiments[r][tau][label]['results'] = results
    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished experiments. Elapsed: %s' % (elapsed_delta, ))

    return experiments


def execute_parallel_forward(**params):
    # Function that executes in parallel the forward simulations
    start_time = time.time()
    proc_number = os.getpid()
    num_exp = params.pop('num_exp')
    r = params.pop('r')
    tau = params.pop('tau')
    label = params.pop('label')
    print('Start (%s). Exp: %s. R: %s. tau: %s. Policy: %s' % (proc_number,
                                                               num_exp,
                                                               r,
                                                               tau,
                                                               label))

    _, _, H_wg, H_cg, H_rg, I_g, D_g, u_g, hops_i, infs_i = forward_integration(**params)

    age_er_prop = params['age_er'].T
    age_er_prop = age_er_prop[:, :, np.newaxis]

    total_hosp = H_wg + H_cg + H_rg
    deaths_incidence = D_g.copy()
    deaths_incidence[:, :, 1:] -= D_g[:, :, :-1]

    assert np.all(deaths_incidence.cumsum(axis=2) == D_g)

    # Getting the results in the absolute scale (not age region proportioned)
    results = {
        'total hospitalizations': total_hosp*age_er_prop,
        'infectious people': I_g*age_er_prop,
        'infections': infs_i*age_er_prop,
        'deaths': deaths_incidence*age_er_prop,
        'vaccinations': u_g*age_er_prop,
        'vaccinations_raw': u_g,
        'new hospitalizations': hops_i*age_er_prop,
    }

    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished (%s). Exp: %s. Elapsed: %s' % (proc_number,
                                                   num_exp,
                                                   elapsed_delta))

    return r, tau, label, results


def search_best_ws_r_metric(num_age_groups, num_ervas, init_vacc,
                            u, T, r_experiments, t0, e, taus, search_num_ws):
    # Constructing ws. Endpoint=False avoids the case 1, 0, 0
    w1 = np.linspace(0, 1, search_num_ws)

    tau_params = {tau: {} for tau in taus}
    for tau in taus:
        mob_av, beta_gh, pop_erva_hat, age_er, rho = get_model_parameters(num_age_groups,
                                                                          num_ervas,
                                                                          init_vacc,
                                                                          t0,
                                                                          tau)
        tau_params[tau]['mob_av'] = mob_av
        tau_params[tau]['beta_gh'] = beta_gh
        tau_params[tau]['pop_erva_hat'] = pop_erva_hat
        tau_params[tau]['rho'] = rho

    epidemic_npy = read_initial_values(age_er, init_vacc, t0)

    age_er_prop = age_er.T
    age_er_prop = age_er_prop[:, :, np.newaxis]

    experiments = {r: {tau: {} for tau in taus} for r in r_experiments}
    num_experiments = 0
    for tau in taus:
        for r in r_experiments:
            beta = r/tau_params[tau]['rho']
            for w in w1:
                num_experiments += 1
                ws = [w, 1-w, 0]
                label = str(w)
                parameters = {
                    'u_con': u,
                    'c1': tau_params[tau]['mob_av'],
                    'beta': beta,
                    'c_gh': tau_params[tau]['beta_gh'],
                    'T': T,
                    'pop_hat': tau_params[tau]['pop_erva_hat'],
                    'age_er': age_er,
                    't0': t0,
                    'ws_vacc': ws,
                    'e': e,
                    'init_vacc': init_vacc,
                    'epidemic_npy': epidemic_npy,
                    'u_op_file': None,
                    'num_exp': num_experiments,
                    'r': r,
                    'tau': tau,
                    'label': label
                }
                experiments[r][tau][label] = {}
                experiments[r][tau][label]['parameters'] = parameters

    # Running on all the available CPUs of the computer
    num_cpus = os.cpu_count()
    start_time = time.time()
    print('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute forward simulation in asynchronous way
        async_res = []
        for r, r_level in experiments.items():
            for tau, tau_level in r_level.items():
                for label, label_level in tau_level.items():
                    policy_params = label_level['parameters']
                    async_res.append(
                        pool.apply_async(execute_parallel_forward,
                                         kwds=policy_params)
                    )

        # Waiting for the values of the async execution
        for res in async_res:
            r, tau, label, results = res.get()
            # Adding the results to our dictionary
            experiments[r][tau][label]['results'] = results
    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished experiments. Elapsed: %s' % (elapsed_delta, ))

    return experiments
