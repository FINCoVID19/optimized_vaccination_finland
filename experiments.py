from forward_integration import (
    forward_integration, get_model_parameters, read_initial_values
)
import numpy as np
from env_var import EPIDEMIC, EXPERIMENTS
from multiprocessing import Pool
import os
import time
import datetime
import json


def get_experiments_results(num_age_groups, num_ervas, e, taus,
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
                if label == 'Optimal':
                    u_op_file = 'out/R_%s_op_sol.npy' % (r, )
                    if os.path.isfile(u_op_file):
                        exec_experiment = True
                    else:
                        u_op_file = None
                        exec_experiment = False
                else:
                    exec_experiment = True
                    u_op_file = None

                if exec_experiment:
                    num_experiments += 1
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
                        'u_op_file': u_op_file,
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
        'infectious': I_g*age_er_prop,
        'infections': infs_i*age_er_prop,
        'deaths': D_g*age_er_prop,
        'new deaths': deaths_incidence*age_er_prop,
        'vaccinations': u_g*age_er_prop,
        'new hospitalizations': hops_i*age_er_prop,
    }

    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished (%s). Exp: %s. Elapsed: %s' % (proc_number,
                                                   num_exp,
                                                   elapsed_delta))

    return r, tau, label, results


def search_best_ws_r_metric(filename, search_num=10):
    # Get the common parameters for all the experiments
    num_age_groups = EXPERIMENTS['num_age_groups']
    num_ervas = EXPERIMENTS['num_ervas']
    T = EXPERIMENTS['simulate_T']
    init_vacc = EXPERIMENTS['init_vacc']
    inc_mob = EXPERIMENTS['inc_mob']
    u = EXPERIMENTS['vaccines_per_day']
    r_experiments = EXPERIMENTS['r_effs']
    t0 = EXPERIMENTS['t0']
    e = EPIDEMIC['e']

    # Calculate the values with the parameters
    mob_av, beta_gh, pop_erva_hat, age_er, rho = get_model_parameters(num_age_groups,
                                                                      num_ervas,
                                                                      init_vacc,
                                                                      t0,
                                                                      inc_mob)
    # Constructing ws. Endpoint=False avoids the case 1, 0, 0
    w1 = np.linspace(0, 1, search_num, endpoint=False)
    w2 = np.linspace(0, 1-w1, search_num)
    # Constructs a dictionary with all betas and ws pairs to try
    all_params = {}
    for r in r_experiments:
        beta = r/rho
        for i in range(search_num):
            w1_i = w1[i]
            for j in range(search_num):
                # Getting w3 as a function of the other 2 ws
                w2_i = w2[j, i]
                w3_i = 1 - w1_i - w2_i
                ws_i = [w1_i, w2_i, w3_i]

                key = (str(r), str(ws_i))
                parameters = {
                    'u_con': u,
                    'c1': mob_av,
                    'beta': beta,
                    'c_gh': beta_gh,
                    'T': T,
                    'pop_hat': pop_erva_hat,
                    'age_er': age_er,
                    't0': t0,
                    'ws_vacc': ws_i,
                    'e': e,
                    'init_vacc': init_vacc,
                    'r': str(r)
                }
                all_params[key] = parameters
        # Adding the case manually
        ws_i = [1, 0, 0]
        key = (str(r), str(ws_i))
        parameters = {
            'u_con': u,
            'c1': mob_av,
            'beta': beta,
            'c_gh': beta_gh,
            'T': T,
            'pop_hat': pop_erva_hat,
            'age_er': age_er,
            't0': t0,
            'ws_vacc': ws_i,
            'e': e,
            'init_vacc': init_vacc,
            'r': str(r)
        }
        all_params[key] = parameters

    # Running on all the available CPUs of the computer
    num_experiments = len(all_params.keys())
    num_cpus = os.cpu_count()
    start_time = time.time()
    print('Running %s experiments with %s CPUS.' % (num_experiments, num_cpus))
    with Pool(processes=num_cpus) as pool:
        # Calling the function to execute forward simulation in asynchronous way
        async_res = [pool.apply_async(execute_parallel_forward, kwds=params)
                     for params in all_params.values()]
        # Waiting for the values of the async execution
        for res in async_res:
            r_str, ws, total_hosp, i_g, deaths_inc, hops_i, infs_i = res.get()
            # Adding the results to our dictionary
            all_params[(r_str, ws)]['results'] = {
                'total_hosp': total_hosp,
                'infectious': i_g,
                'deaths_inc': deaths_inc,
                'hosp_inc': hops_i,
                'infections': infs_i
            }
    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished experiments. Elapsed: %s' % (elapsed_delta, ))

    # Constructing a dictionary to store the best results for r, metric and ws
    first_key = list(all_params.keys())[0]
    results_keys = all_params[first_key]['results'].keys()
    best_results = {key: {} for key in results_keys}
    for metric, empty_dic in best_results.items():
        for r in r_experiments:
            r_str = str(r)
            empty_dic[r_str] = {
                'best': np.inf,
                'ws': None
            }

    # Searching for the best results
    for key_beta_ws, params_results in all_params.items():
        r_str, _ = key_beta_ws
        for metric, values in best_results.items():
            # Getting the result of the metric
            res_beta_ws = params_results['results'][metric]
            overall_result = res_beta_ws.sum()
            # Checking if we did better than before
            if overall_result < values[r_str]['best']:
                values[r_str]['best'] = overall_result
                values[r_str]['ws'] = params_results['ws_vacc']

    # Storing the results in a JSON file
    with open(filename, 'w') as f:
        json.dump(best_results, f, indent=2)
    print('Results written to file: %s' % (filename, ))


if __name__ == "__main__":
    search_best_ws_r_metric('out/best_ws.json')
