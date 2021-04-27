import datetime
from forward_integration import forward_integration, get_model_parameters
import numpy as np


def get_experiments_results(num_age_groups, num_ervas, e,
                            init_vacc, strategies, u, T, r_experiments, t0):
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
        for policy, ws, label in strategies:
            _, _, H_wg, H_cg, H_rg, I_g, D_g, u_g, hops_i = forward_integration(
                                                                u_con=u,
                                                                c1=mob_av,
                                                                beta=beta,
                                                                c_gh=beta_gh,
                                                                T=T,
                                                                pop_hat=pop_erva_hat,
                                                                age_er=age_er,
                                                                t0=t0,
                                                                policy=policy,
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
        complete_results[r] = results_label

    return complete_results


def plot_age_groups(ax, D_g, t0, T, age_labels, plot_subject):
    begin = datetime.datetime.strptime(t0, '%Y-%m-%d')

    num_ages, num_ervas, days = D_g.shape
    assert num_ages == len(age_labels)

    x = [begin + datetime.timedelta(days=day) for day in range(T)]
    deaths = D_g.sum(axis=1)
    assert deaths.shape[0] == num_ages
    assert deaths.shape[1] == T

    for age_i in range(num_ages):
        ax.plot(x, deaths[age_i, :], label=age_labels[age_i])

    ax.set_xlabel('Date')
    ax.set_ylabel('Number of %s' % (plot_subject, ))
    ax.set_title('Number of %s per age group' % (plot_subject, ))

    ax.legend()

    return ax
