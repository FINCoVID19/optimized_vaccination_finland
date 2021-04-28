from forward_integration import forward_integration, get_model_parameters
import numpy as np
from env_var import EPIDEMIC, EXPERIMENTS
from multiprocessing import Pool
import os
import time
import datetime
import json


def get_experiments_results(num_age_groups, num_ervas, e,
                            init_vacc, strategies, u, T, r_experiments, t0, u_op_file):
    mob_av, beta_gh, pop_erva_hat, age_er, rho = get_model_parameters(num_age_groups,
                                                                      num_ervas,
                                                                      init_vacc,
                                                                      t0)
    age_er_prop = age_er.T
    age_er_prop = age_er_prop[:, :, np.newaxis]
    complete_results = {}
    total_rs = len(r_experiments)

    experiments_params = {
        'num_ervas': num_ervas,
        'num_age_groups': num_age_groups,
        'u': u,
        'rho': rho,
        't0': t0,
        'T': T,
        'e': e,
        'r_experiments': r_experiments,
        'init_vacc': init_vacc,
        'strategies': strategies
    }
    print(('Beginning experiments.\n'
           'Parameters:\n'
           'Number of age ervas: {num_ervas}.\n'
           'Number of age groups: {num_age_groups}.\n'
           'Number of vaccines per day: {u}.\n'
           'rho: {rho}.\n'
           't0: {t0}.\n'
           'T: {T}.\n'
           'Vaccine efficacy (e): {e}.\n'
           'Rs to try: {r_experiments}.\n'
           'Initialize with vaccinated people: {init_vacc}.\n'
           'Strategies:\n{strategies}.\n').format(**experiments_params))

    for i, r in enumerate(r_experiments):
        beta = r/rho
        results_label = []
        j = 1
        total_strategies = len(strategies)
        for ws, label in strategies:
            _, _, H_wg, H_cg, H_rg, I_g, D_g, u_g, hops_i = forward_integration(
                                                                u_con=u,
                                                                c1=mob_av,
                                                                beta=beta,
                                                                c_gh=beta_gh,
                                                                T=T,
                                                                pop_hat=pop_erva_hat,
                                                                age_er=age_er,
                                                                t0=t0,
                                                                ws_vacc=ws,
                                                                e=e,
                                                                init_vacc=init_vacc
                                                            )
            print('Finished R: %s. Beta: %s %d/%d. Policy: %s. %d/%d' % (r,
                                                                         beta,
                                                                         i+1,
                                                                         total_rs,
                                                                         label,
                                                                         j,
                                                                         total_strategies))
            total_hosp = H_wg + H_cg + H_rg
            deaths_incidence = D_g.copy()
            deaths_incidence[:, :, 1:] -= D_g[:, :, :-1]

            assert np.all(deaths_incidence.cumsum(axis=2) == D_g)

            results = {
                'hospitalizations': total_hosp*age_er_prop,
                'infections': I_g*age_er_prop,
                'deaths': D_g*age_er_prop,
                'new deaths': deaths_incidence*age_er_prop,
                'vaccinations': u_g*age_er_prop,
                'new hospitalizations': hops_i*age_er_prop,
            }
            result_pairs = (label, results)
            results_label.append(result_pairs)
            j += 1
        if r == 1.5:
            _, _, H_wg, H_cg, H_rg, I_g, D_g, u_g, hops_i = forward_integration(
                                                                u_con=u,
                                                                c1=mob_av,
                                                                beta=beta,
                                                                c_gh=beta_gh,
                                                                T=T,
                                                                pop_hat=pop_erva_hat,
                                                                age_er=age_er,
                                                                t0=t0,
                                                                ws_vacc=ws,
                                                                e=e,
                                                                init_vacc=init_vacc,
                                                                u_op_file=u_op_file
                                                            )
            total_hosp = H_wg + H_cg + H_rg
            deaths_incidence = D_g.copy()
            deaths_incidence[:, :, 1:] -= D_g[:, :, :-1]

            assert np.all(deaths_incidence.cumsum(axis=2) == D_g)

            # print((u_g*age_er_prop).sum(axis=(0, 1)))
            results = {
                'hospitalizations': total_hosp*age_er_prop,
                'infections': I_g*age_er_prop,
                'deaths': D_g*age_er_prop,
                'new deaths': deaths_incidence*age_er_prop,
                'vaccinations': u_g*age_er_prop,
                'new hospitalizations': hops_i*age_er_prop,
            }
            result_pairs = ('Optimal', results)
            results_label.append(result_pairs)
        complete_results[r] = results_label

    return complete_results


def execute_parallel_forward(**params):
    # Function that executes in parallel the forward simulations
    start_time = time.time()
    proc_number = os.getpid()
    r_str = params.pop('r')
    print('Start execution in process: %s.\nR: %s. Ws: %s' % (proc_number,
                                                              r_str,
                                                              params['ws_vacc']))

    _, _, H_wg, H_cg, H_rg, I_g, D_g, _, hops_i = forward_integration(**params)

    age_er_prop = params['age_er'].T
    age_er_prop = age_er_prop[:, :, np.newaxis]

    total_hosp = H_wg + H_cg + H_rg
    deaths_incidence = D_g.copy()
    deaths_incidence[:, :, 1:] -= D_g[:, :, :-1]

    assert np.all(deaths_incidence.cumsum(axis=2) == D_g)

    # Getting the results in the absolute scale (not age region proportioned)
    total_hosp = total_hosp*age_er_prop
    i_g = I_g*age_er_prop
    deaths_incidence = deaths_incidence*age_er_prop
    hops_i = hops_i*age_er_prop

    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished execution in process: %s. Elapsed: %s' % (proc_number,
                                                              elapsed_delta))

    return r_str, str(params['ws_vacc']), total_hosp, i_g, deaths_incidence, hops_i


def search_best_ws_r_metric(filename, search_num=10):
    # Get the common parameters for all the experiments
    num_age_groups = EXPERIMENTS['num_age_groups']
    num_ervas = EXPERIMENTS['num_ervas']
    T = EXPERIMENTS['simulate_T']
    init_vacc = EXPERIMENTS['init_vacc']
    u = EXPERIMENTS['vaccines_per_day']
    r_experiments = EXPERIMENTS['r_effs']
    t0 = EXPERIMENTS['t0']
    e = EPIDEMIC['e']

    # Calculate the values with the parameters
    mob_av, beta_gh, pop_erva_hat, age_er, rho = get_model_parameters(num_age_groups,
                                                                      num_ervas,
                                                                      init_vacc,
                                                                      t0)
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
            r_str, ws, total_hosp, i_g, deaths_inc, hops_i = res.get()
            # Adding the results to our dictionary
            all_params[(r_str, ws)]['results'] = {
                'total_hosp': total_hosp,
                'infections': i_g,
                'deaths_inc': deaths_inc,
                'hosp_inc': hops_i
            }
    elapsed_time = time.time() - start_time
    elapsed_delta = datetime.timedelta(seconds=elapsed_time)
    print('Finished experiments. Elapsed: %s' % (elapsed_delta, ))

    # Constructing a dictionary to store the best results for r, metric and ws
    best_results = {
        'total_hosp': {
        },
        'infections': {
        },
        'deaths_inc': {
        },
        'hosp_inc': {
        }
    }
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
