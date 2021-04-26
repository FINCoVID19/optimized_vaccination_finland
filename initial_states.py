import os
import logging
from logging import handlers
import pandas as pd
import numpy as np
from env_var import EPIDEMIC
from fetch_data import (
    construct_cases_age_erva_daily, static_population_erva_age,
    construct_thl_vaccines_erva_daily, construct_hs_hosp_age_erva
)


def compartment_values_daily(logger, erva_pop_file, filename=None,
                             number_age_groups=9):
    logger.info('Calculating epidemic compartments')
    cases_by_age_erva = construct_cases_age_erva_daily(logger,
                                                       number_age_groups=number_age_groups)

    inf_period = (EPIDEMIC['T_I'])**(-1)
    inf_period = int(inf_period)
    lat_period = (EPIDEMIC['T_E'])**(-1)
    lat_period = int(lat_period)
    logger.info('Infectious period: %d. Latent period: %d' % (inf_period,
                                                              lat_period))
    a = EPIDEMIC['unreported_exponent']

    cases_by_age_erva.sort_values(['Time', 'erva'])
    dates = pd.unique(cases_by_age_erva['Time'])
    ervas = pd.unique(cases_by_age_erva['erva'])
    num_ervas = len(ervas)
    logger.debug(ervas)
    ages_names = cases_by_age_erva.columns[2:]

    cases_erva_age_npy = cases_by_age_erva.values
    cases_erva_age_npy = cases_erva_age_npy[:, 2:]
    dates_ervas, ages = cases_erva_age_npy.shape
    assert dates_ervas % num_ervas == 0
    days = int(dates_ervas/num_ervas)
    cases_erva_age_npy = cases_erva_age_npy.reshape(days, num_ervas, ages)

    assert len(ages_names) == ages
    assert len(dates) == days

    infectious_detected = np.zeros_like(cases_erva_age_npy)
    recovered_detected = np.zeros_like(cases_erva_age_npy)
    lookback_period = inf_period + lat_period
    for day_t in range(days):
        omega = day_t - lookback_period + 1
        if omega < 0:
            omega = 0
        cases_in_period = cases_erva_age_npy[omega:day_t+1, ]
        # Get the total infected in the period and assign to time t
        infectious_detected[day_t, ] = cases_in_period.sum(axis=0)

        recovered_period = cases_erva_age_npy[:omega, ]
        recovered_detected[day_t, ] = recovered_period.sum(axis=0)

    k = np.arange(ages) + 1
    upscale_factor = 1 + 9*k**(-a)

    # Broadcasting operation
    upscale_factor = upscale_factor[np.newaxis, np.newaxis, :]

    logger.debug('Multiplied fraction: %s' % (upscale_factor, ))
    infectious_undetected = infectious_detected * upscale_factor
    recovered_undetected = recovered_detected * upscale_factor

    infected_total = infectious_detected + infectious_undetected
    recovered_total = recovered_detected + recovered_undetected

    infected_real = (inf_period/lookback_period)*infected_total
    exposed_real = (lat_period/lookback_period)*infected_total

    # Getting the population to get the final Susceptibles
    pop_ervas, _ = static_population_erva_age(logger, erva_pop_file,
                                              number_age_groups=number_age_groups)
    pop_ervas = pop_ervas[~pop_ervas['erva'].str.contains('All')]
    pop_ervas = pop_ervas.sort_values(['erva', 'age_group'])
    pop_ervas_npy = pop_ervas['Total'].values
    pop_ervas_npy = pop_ervas_npy.reshape(num_ervas, ages)

    # To prepare for broadcasting operation
    pop_ervas_npy = pop_ervas_npy[np.newaxis, :]

    susceptible = np.zeros_like(cases_erva_age_npy)
    susceptible = pop_ervas_npy - exposed_real - infected_real - recovered_total

    complete_dataframe = pd.DataFrame()
    for erva_i, erva_name in enumerate(ervas):
        for age_i, age_name in enumerate(ages_names):
            dataframe_data = {
                'date': dates,
                'erva': [erva_name]*days,
                'age': [age_name]*days,
                'susceptible': susceptible[:, erva_i, age_i],
                'infected detected': infectious_detected[:, erva_i, age_i],
                'infected undetected': infectious_undetected[:, erva_i, age_i],
                'infected': infected_real[:, erva_i, age_i],
                'exposed': exposed_real[:, erva_i, age_i],
                'recovered': recovered_total[:, erva_i, age_i],
            }
            erva_age_dataframe = pd.DataFrame(data=dataframe_data)
            complete_dataframe = complete_dataframe.append(erva_age_dataframe)

    if filename is not None:
        complete_dataframe.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return complete_dataframe


def full_epidemic_state_finland(logger, erva_pop_file, filename=None,
                                number_age_groups=9, init_vacc=True):
    logger.info('Getting complete state of epidemic with '
                'epidemic compartments, vaccines and hospitalizations')
    logger.info('Number of age groups: %d' % (number_age_groups))
    compart_df = compartment_values_daily(logger, erva_pop_file,
                                          number_age_groups=number_age_groups)
    logger.info('Epidemic compartments gotten.')
    vacc_df = construct_thl_vaccines_erva_daily(logger,
                                                number_age_groups=number_age_groups)
    logger.info('Number of vaccinated gotten.')
    hosp_df = construct_hs_hosp_age_erva(logger, number_age_groups=number_age_groups)
    logger.info('Number of hospitalizations gotten.')
    epidemic_state = pd.merge(compart_df, vacc_df,
                              on=['date', 'erva', 'age'],
                              how='left')
    epidemic_state = pd.merge(epidemic_state, hosp_df,
                              on=['date', 'erva', 'age'],
                              how='left')
    # Merge will left missing values with NaNs. Filled them with 0
    epidemic_state = epidemic_state.fillna(0)

    if init_vacc:
        epidemic_state['First dose cumulative'] = epidemic_state.groupby(['erva', 'age'])['First dose'].cumsum()
        epidemic_state['Second dose cumulative'] = epidemic_state.groupby(['erva', 'age'])['Second dose'].cumsum()
    else:
        epidemic_state['First dose cumulative'] = 0
        epidemic_state['Second dose cumulative'] = 0
        epidemic_state['susceptible'] = epidemic_state['susceptible'] + epidemic_state['recovered']
        epidemic_state['recovered'] = 0

    # Removing from susceptible data for vaccinated and hospitalized
    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['First dose cumulative']

    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['ward']
    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['icu']

    if filename is not None:
        epidemic_state.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return epidemic_state


if __name__ == "__main__":
    # Select data directory
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # Get a logger of the events
    logfile = os.path.join(curr_dir, 'logs_initial_states.log')
    numeric_log_level = getattr(logging, "INFO", None)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S %p',
        level=numeric_log_level,
        handlers=[
            # Max store 300MB of logs
            handlers.RotatingFileHandler(logfile,
                                         maxBytes=100e6,
                                         backupCount=3),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info('Logger ready. Logging to file: %s' % (logfile))

    # Starting with the tasks (main loop)
    try:
        stats_dir = os.path.join(curr_dir, 'stats')
        out_dir = os.path.join(curr_dir, 'out')
        os.makedirs(out_dir, exist_ok=True)

        erva_pop_file = os.path.join(stats_dir, 'erva_population_age_2020.csv')

        out_csv_filename = os.path.join(out_dir, 'epidemic_finland_9.csv')
        full_epidemic_state_finland(logger, erva_pop_file, out_csv_filename,
                                    number_age_groups=9)

        out_csv_filename = os.path.join(out_dir, 'epidemic_finland_8.csv')
        full_epidemic_state_finland(logger, erva_pop_file, out_csv_filename,
                                    number_age_groups=8)

        out_csv_filename = os.path.join(out_dir, 'epidemic_finland_8_no_vacc.csv')
        full_epidemic_state_finland(logger, erva_pop_file, out_csv_filename,
                                    number_age_groups=8, init_vacc=False)

        out_csv_filename = os.path.join(out_dir, 'epidemic_finland_9_no_vacc.csv')
        full_epidemic_state_finland(logger, erva_pop_file, out_csv_filename,
                                    number_age_groups=9, init_vacc=False)
    except Exception:
        logger.exception("Fatal error in main loop")
