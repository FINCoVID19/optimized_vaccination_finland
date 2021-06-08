import os
import sys
import logging
from logging import handlers
import argparse
import pandas as pd
import numpy as np
from env_var import EPIDEMIC
from fetch_data import (
    construct_cases_age_region_daily, static_population_region_age,
    construct_thl_vaccines_erva_daily, construct_hs_hosp_age_erva
)


def compartment_values_daily(logger, region_pop_file, number_age_groups,
                             region, filename=None):
    logger.info('Calculating epidemic compartments')
    cases_by_age_region = construct_cases_age_region_daily(logger,
                                                           region=region,
                                                           number_age_groups=number_age_groups)

    inf_period = (EPIDEMIC['T_I'])**(-1)
    inf_period = int(inf_period)
    lat_period = (EPIDEMIC['T_E'])**(-1)
    lat_period = int(lat_period)
    logger.info('Infectious period: %d. Latent period: %d' % (inf_period,
                                                              lat_period))
    a = EPIDEMIC['unreported_exponent']

    cases_by_age_region.sort_values(['Time', 'region'])
    dates = pd.unique(cases_by_age_region['Time'])
    regions = pd.unique(cases_by_age_region['region'])
    num_regions = len(regions)
    logger.debug('Constructing cases for: %s' % (regions, ))
    ages_names = cases_by_age_region.columns[2:]

    cases_region_age_npy = cases_by_age_region.values
    cases_region_age_npy = cases_region_age_npy[:, 2:]
    dates_region, ages = cases_region_age_npy.shape
    assert dates_region % num_regions == 0
    days = int(dates_region/num_regions)
    cases_region_age_npy = cases_region_age_npy.reshape(days, num_regions, ages)

    assert len(ages_names) == ages
    assert len(dates) == days

    infectious_detected = np.zeros_like(cases_region_age_npy)
    recovered_detected = np.zeros_like(cases_region_age_npy)
    lookback_period = inf_period + lat_period
    for day_t in range(days):
        omega = day_t - lookback_period + 1
        if omega < 0:
            omega = 0
        cases_in_period = cases_region_age_npy[omega:day_t+1, ]
        # Get the total infected in the period and assign to time t
        infectious_detected[day_t, ] = cases_in_period.sum(axis=0)

        recovered_period = cases_region_age_npy[:omega, ]
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
    pop_regions, _ = static_population_region_age(logger,
                                                  csv_file=region_pop_file,
                                                  number_age_groups=number_age_groups)
    pop_regions = pop_regions[~pop_regions['region'].str.contains('All')]
    pop_regions = pop_regions.sort_values(['region', 'age_group'])
    pop_regions_npy = pop_regions['Total'].values
    pop_regions_npy = pop_regions_npy.reshape(num_regions, ages)

    # To prepare for broadcasting operation
    pop_regions_npy = pop_regions_npy[np.newaxis, :]

    susceptible = np.zeros_like(cases_region_age_npy)
    susceptible = pop_regions_npy - exposed_real - infected_real - recovered_total

    complete_dataframe = pd.DataFrame()
    for reg_i, reg_name in enumerate(regions):
        for age_i, age_name in enumerate(ages_names):
            dataframe_data = {
                'date': dates,
                'region': [reg_name]*days,
                'age': [age_name]*days,
                'susceptible': susceptible[:, reg_i, age_i],
                'infected detected': infectious_detected[:, reg_i, age_i],
                'infected undetected': infectious_undetected[:, reg_i, age_i],
                'infected': infected_real[:, reg_i, age_i],
                'exposed': exposed_real[:, reg_i, age_i],
                'recovered': recovered_total[:, reg_i, age_i],
            }
            region_age_dataframe = pd.DataFrame(data=dataframe_data)
            complete_dataframe = complete_dataframe.append(region_age_dataframe)

    if filename is not None:
        complete_dataframe.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return complete_dataframe


def full_epidemic_state_finland(logger, region_pop_file, region, filename,
                                number_age_groups, init_vacc, e=EPIDEMIC['e']):
    logger.info('Getting complete state of epidemic with '
                'epidemic compartments, vaccines and hospitalizations')
    logger.info('Number of age groups: %d' % (number_age_groups))
    compart_df = compartment_values_daily(logger,
                                          region_pop_file=region_pop_file,
                                          region=region,
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
        epidemic_state['vaccinated'] = e*epidemic_state['First dose cumulative']
        epidemic_state['vaccinated no imm'] = (1-e)*epidemic_state['First dose cumulative']
        epidemic_state['Second dose cumulative'] = epidemic_state.groupby(['erva', 'age'])['Second dose'].cumsum()
    else:
        epidemic_state['First dose cumulative'] = 0
        epidemic_state['Second dose cumulative'] = 0
        epidemic_state['susceptible'] = epidemic_state['susceptible'] + epidemic_state['recovered']
        epidemic_state['recovered'] = 0
        epidemic_state['vaccinated'] = 0
        epidemic_state['vaccinated no imm'] = 0

    # Removing from susceptible data for vaccinated and hospitalized
    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['vaccinated']
    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['vaccinated no imm']
    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['ward']
    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['icu']

    if filename is not None:
        epidemic_state.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return epidemic_state


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Get initial states for the epidemic in Finland."
    )

    parser.add_argument('--num_age_groups', type=int,
                        default=9,
                        choices=[8, 9],
                        help="Get the initial states for 'num_age_groups'.")

    parser.add_argument("--region", type=str,
                        default='erva',
                        choices=["erva", "hcd"],
                        help="Get the initial states for this 'region'.")

    parser.add_argument('--init_vacc', action='store_false',
                        help=('If set the initial states will be given '
                              'without vaccination.'))

    parser.add_argument("--log_file", type=str,
                        default='logs_initial_states.log',
                        help="Logging file.")

    parser.add_argument("--log_level", type=str,
                        default='INFO',
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="Set logging level.")

    return parser.parse_args(args)


def initial_states():
    args = parse_args()
    # Select data directory
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # Get a logger of the events
    logfile = os.path.join(curr_dir, args.log_file)
    numeric_log_level = getattr(logging, args.log_level, None)
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

    num_age_groups = args.num_age_groups
    region = args.region
    init_vacc = args.init_vacc
    logger.info(('Constructing initial states with parameters:'
                 'Age groups: %s'
                 'Region: %s'
                 'init_vacc: %s') % (num_age_groups, region, init_vacc))

    stats_dir = os.path.join(curr_dir, 'stats')
    out_dir = os.path.join(curr_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Starting with the tasks (main loop)
    try:
        pop_filename = '%s_population_age_2020.csv' % (region, )
        region_pop_file = os.path.join(stats_dir, pop_filename)

        if init_vacc:
            csv_name = 'epidemic_finland_%s_%s.csv' % (num_age_groups, region)
        else:
            csv_name = 'epidemic_finland_%s_%s_no_vacc.csv' % (num_age_groups,
                                                               region)
        out_csv_filename = os.path.join(out_dir, csv_name)
        full_epidemic_state_finland(logger,
                                    region_pop_file=region_pop_file,
                                    filename=out_csv_filename,
                                    number_age_groups=num_age_groups,
                                    init_vacc=init_vacc,
                                    region=region)
    except Exception:
        logger.exception("Fatal error in main loop")


if __name__ == "__main__":
    initial_states()
