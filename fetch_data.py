import os
import requests
import json
from io import StringIO
import pandas as pd
import numpy as np
import logging
import datetime
from logging import handlers
from env_var import REQUESTS


def transform_thl_week_datetime(thl_time):
    week_number = int(thl_time[-2:])
    year = thl_time[5:9]
    # Making correction to match Python date to THL dates
    if year == '2020':
        week_number = week_number - 1
    new_time = 'Year %s Week %02.f-1' % (year, week_number)
    # Get monday date based on the week number
    # This needs to be done bc THL does not provide daily counts only weekly
    monday_of_week = datetime.datetime.strptime(new_time, "Year %Y Week %W-%w")

    return monday_of_week


def divide_equally_week_values(week_values):
    assert len(week_values.shape) == 1
    day_values = np.zeros((len(week_values), 7))

    for idx, value in enumerate(week_values):
        remaining = int(value % 7)
        if remaining == 0:
            day_values[idx, :] = value/7
        else:
            day_values[idx, :] = np.floor(value/7)
            day_values[idx, :remaining] += 1

    day_values = day_values.astype(np.int32)

    return day_values


def static_population_erva(logger):
    population_data = {
        "HYKS": 2198182,
        "TYKS": 869004,
        "TAYS": 902681,
        "KYS": 797234,
        "OYS": 736563,
        "Åland": 30129,
    }
    # Add the population of all Finland
    population_data['All'] = sum(population_data.values())
    logger.info('Returned population data for 2019')
    return population_data


def static_population_erva_age(logger, csv_file, number_age_groups=9):
    population_age_df = pd.read_csv(csv_file, sep=";", encoding='utf-8')
    logger.debug('Constructed pandas dataframe')

    population_age_df = population_age_df.drop(columns=['Males', 'Females'])

    population_age_df = population_age_df[~population_age_df['Age'].str.contains('Total')]
    population_age_df['Total'] = population_age_df['Total'].astype('int32')

    age_group_mapping = REQUESTS['age_group_mappings'][number_age_groups]['age_groups_mapping_population']
    population_age_df['age_group'] = population_age_df.apply(lambda row: age_group_mapping[row['Age']], axis=1)
    population_age_df = population_age_df.groupby(by=['erva', 'age_group'],
                                                  as_index=False).sum()
    pop_age_prop = population_age_df.copy()
    ervas = pd.unique(pop_age_prop['erva'])
    for erva in ervas:
        erva_counts = pop_age_prop.loc[pop_age_prop['erva'] == erva, 'Total'].values
        tot_erva = np.sum(erva_counts)
        pop_age_prop.loc[pop_age_prop['erva'] == erva, 'Total'] /= tot_erva

    return population_age_df, pop_age_prop


def fetch_thl_vaccines_erva_weekly(logger, filename=None, number_age_groups=9):
    logger.debug('Getting THL vaccination statistics')

    # Select the appropriate URL
    url = REQUESTS['vaccination']
    headers = {
        'User-Agent': 'Me'
    }
    logger.debug(('Sending the request..\n'
                  'URL: %s\n'
                  'Headers: %s\n') % (url,
                                      json.dumps(headers, indent=1)))
    # Load data from THL's API as CSV
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(response.content)
        raise RuntimeError("THL's API failed!")

    # Tell to the response what is the correct encoding
    response.encoding = response.apparent_encoding
    logger.info('Got THL reported cases!')

    buffer_for_pandas = StringIO(response.text)
    vaccinated_df = pd.read_csv(buffer_for_pandas, sep=";")
    logger.debug('Constructed pandas dataframe')

    vaccinated_df = vaccinated_df.fillna(0)
    logger.debug('Filled NaNs with zeros')

    hcd_erva_mapping = REQUESTS['hcd_erva_mapping']
    vaccinated_df['erva'] = vaccinated_df.apply(lambda row: hcd_erva_mapping[row['Area']], axis=1)
    logger.debug('Augmented data with erva')

    age_group_mapping = REQUESTS['age_group_mappings'][number_age_groups]['age_group_mapping']
    vaccinated_df['age group'] = vaccinated_df.apply(lambda row: age_group_mapping[row['Age']], axis=1)
    logger.debug('Augmented data with age groups')

    columns_agg_erva = ['age group',
                        'Vaccination dose',
                        'Time',
                        'erva']
    vaccinated_erva = vaccinated_df.groupby(by=columns_agg_erva, as_index=False).sum()

    vaccinated_erva = vaccinated_erva[['erva', 'Time', 'age group', 'Vaccination dose', 'val']]
    vaccinated_erva = vaccinated_erva.sort_values(by=['erva', 'Time', 'age group', 'Vaccination dose'])
    logger.debug('Sorting by ERVA')

    # Removing total counts in Finland
    vaccinated_erva = vaccinated_erva[~vaccinated_erva['Time'].str.contains('All')]
    vaccinated_erva = vaccinated_erva[~vaccinated_erva['erva'].str.contains('All')]
    vaccinated_erva = vaccinated_erva[~vaccinated_erva['age group'].str.contains('All')]
    vaccinated_erva = vaccinated_erva[~vaccinated_erva['Vaccination dose'].str.contains('All')]

    if filename is not None:
        vaccinated_erva.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return vaccinated_erva


def fetch_hs_hospitalizations(logger):
    logger.debug('Getting HS hospitalizations')

    # Select the appropriate URL
    url = REQUESTS['hospitalizations']
    headers = {
        'User-Agent': 'Me'
    }
    logger.debug(('Sending the request..\n'
                  'URL: %s\n'
                  'Headers: %s\n') % (url,
                                      json.dumps(headers, indent=1)))
    # Load data from THL's API as CSV
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(response.content)
        raise RuntimeError("HS API failed!")

    # Tell to the response what is the correct encoding
    response.encoding = response.apparent_encoding

    response_json = response.json()
    hospitalizations = response_json['hospitalised']
    hospital_df = pd.DataFrame(hospitalizations)

    hospital_df = hospital_df[~hospital_df['area'].str.contains('Finland')]
    hospital_df.columns = ['date', 'erva', 'hospitalized', 'ward', 'icu', 'dead']

    hospital_df['date'] = hospital_df.apply(
                            lambda row: row['date'].split('T')[0],
                            axis=1
                          )
    hospital_df = hospital_df.sort_values(['date', 'erva'])

    return hospital_df


def construct_thl_vaccines_erva_daily(logger, filename=None, number_age_groups=9):
    vaccinated_weekly = fetch_thl_vaccines_erva_weekly(logger,
                                                       number_age_groups=number_age_groups)
    vaccinated_list = vaccinated_weekly.values

    dates = np.unique(vaccinated_list[:, 1])
    ervas = np.unique(vaccinated_list[:, 0])

    age_groups = pd.unique(vaccinated_weekly['age group'])

    header = 'date;erva;age;First dose;Second dose'
    final_lines = [header, ]
    for date in dates:
        for erva in ervas:
            vacc_week = vaccinated_list[np.where(
                            (vaccinated_list[:, 1] == date) & (vaccinated_list[:, 0] == erva)
                        )]
            vacc_day = np.copy(vacc_week)
            vacc_day_vals = divide_equally_week_values(vacc_day[:, 4])

            monday_of_week = transform_thl_week_datetime(date)
            # Augment by day, start on monday and finish sunday (7 days)
            for day in range(7):
                date_i = monday_of_week + datetime.timedelta(days=day)
                date_str = date_i.strftime('%Y-%m-%d')
                line_counter = 0
                for age_g in age_groups:
                    line = [date_str,
                            erva,
                            age_g,
                            str(vacc_day_vals[line_counter, day]),
                            str(vacc_day_vals[line_counter+1, day])]
                    line_str = ';'.join(line)
                    final_lines.append(line_str)
                    line_counter += 2

    complete_csv = '\n'.join(final_lines)

    buffer_for_pandas = StringIO(complete_csv)
    vaccinated_daily = pd.read_csv(buffer_for_pandas, sep=";")

    logger.debug('Constructed pandas dataframe')
    if filename is not None:
        vaccinated_daily.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return vaccinated_daily


def fetch_thl_cases_erva_daily(logger):
    logger.debug('Getting THL reported cases')

    # Select the appropriate URL
    url = REQUESTS['cases_by_day']
    headers = {
        'User-Agent': 'Me'
    }
    logger.debug(('Sending the request..\n'
                  'URL: %s\n'
                  'Headers: %s\n') % (url,
                                      json.dumps(headers, indent=1)))
    # Load data from THL's API as CSV
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(response.content)
        raise RuntimeError("THL's API failed!")

    # Tell to the response what is the correct encoding
    response.encoding = response.apparent_encoding
    logger.info('Got THL reported cases!')

    buffer_for_pandas = StringIO(response.text)
    new_cases_df = pd.read_csv(buffer_for_pandas, sep=";")
    logger.debug('Constructed pandas dataframe')

    new_cases_df = new_cases_df[~new_cases_df['Time'].str.contains('Week|All')]
    logger.debug('Removed total counts and weekly counts')

    new_cases_df = new_cases_df.fillna(0)
    logger.debug('Filled NaNs with zeros')

    new_cases_df = new_cases_df.drop(columns=['Measure'])
    logger.debug('Removed unecessary Measure column')

    hcd_erva_mapping = REQUESTS['hcd_erva_mapping']
    new_cases_df['erva'] = new_cases_df.apply(lambda row: hcd_erva_mapping[row['Area']], axis=1)
    logger.debug('Augmented data with erva')

    reported_cases_erva = new_cases_df.groupby(by=['Time', 'erva'], as_index=False).sum()
    logger.debug('Keeping only ervas')

    # Remove the counts for Finland
    reported_cases_erva = reported_cases_erva[~reported_cases_erva['erva'].str.contains('All')]

    return reported_cases_erva


def fetch_thl_cases_erva_weekly(logger):
    logger.debug('Getting THL reported cases')

    # Select the appropriate URL
    url = REQUESTS['cases_by_hcd']
    headers = {
        'User-Agent': 'Me'
    }
    logger.debug(('Sending the request..\n'
                  'URL: %s\n'
                  'Headers: %s\n') % (url,
                                      json.dumps(headers, indent=1)))
    # Load data from THL's API as CSV
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(response.content)
        raise RuntimeError("THL's API failed!")

    # Tell to the response what is the correct encoding
    response.encoding = response.apparent_encoding
    logger.info('Got THL reported cases!')

    buffer_for_pandas = StringIO(response.text)
    new_cases_df = pd.read_csv(buffer_for_pandas, sep=";")
    logger.debug('Constructed pandas dataframe')

    new_cases_df = new_cases_df.fillna(0)
    logger.debug('Filled NaNs with zeros')

    hcd_erva_mapping = REQUESTS['hcd_erva_mapping']
    new_cases_df['erva'] = new_cases_df.apply(lambda row: hcd_erva_mapping[row['Area']], axis=1)
    logger.debug('Augmented data with erva')

    reported_cases_erva = new_cases_df.groupby(by=['Time', 'erva'], as_index=False).sum()
    logger.debug('Keeping only ervas')

    return reported_cases_erva


def fetch_finland_cases_age_weekly(logger, number_age_groups=9):
    logger.debug('Getting THL reported cases')

    # Select the appropriate URL
    url = REQUESTS['cases_by_age']
    headers = {
        'User-Agent': 'Me'
    }
    logger.debug(('Sending the request..\n'
                  'URL: %s\n'
                  'Headers: %s\n') % (url,
                                      json.dumps(headers, indent=1)))
    # Load data from THL's API as CSV
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(response.content)
        raise RuntimeError("THL's API failed!")

    # Tell to the response what is the correct encoding
    response.encoding = response.apparent_encoding
    logger.info('Got THL reported cases!')

    buffer_for_pandas = StringIO(response.text)
    new_cases_df = pd.read_csv(buffer_for_pandas, sep=";")
    logger.debug('Constructed pandas dataframe')

    new_cases_df = new_cases_df[~new_cases_df['Time'].str.contains('All')]
    logger.debug('Removed total counts and weekly counts')

    new_cases_df.loc[new_cases_df['val'] == '..', 'val'] = 0

    new_cases_df = new_cases_df.fillna(0)
    logger.debug('Filled NaNs with zeros')

    new_cases_df['val'] = new_cases_df['val'].astype('int32')

    age_group_mapping = REQUESTS['age_group_mappings'][number_age_groups]['age_group_mapping_cases']
    new_cases_df['age_group'] = new_cases_df.apply(lambda row: age_group_mapping[row['Age']], axis=1)
    logger.debug('Augmented data age groups')

    cases_age = new_cases_df.groupby(by=['Time', 'age_group'], as_index=False).sum()
    logger.debug('Keeping only age groups')

    age_groups = REQUESTS['age_group_mappings'][number_age_groups]['age_groups']
    cases_age_prop = cases_age.copy()
    all_times = pd.unique(cases_age['Time'])
    for time in all_times:
        date_df = cases_age.loc[cases_age['Time'] == time, ]
        tot_reported = date_df[date_df['age_group'].str.contains('All')]
        age_reported = date_df[~date_df['age_group'].str.contains('All')]
        tot_val = tot_reported['val'].values[0]
        age_val = np.sum(age_reported['val'].values)
        # Checking if there's a difference between sum in age groups and total
        if age_val != tot_val:
            missing_cases = tot_val - age_val
            logger.debug('%s. Missing: %d' % (time, missing_cases))
            add_cases = np.floor(missing_cases / len(age_groups))
            cases_age.loc[cases_age['Time'] == time, 'val'] += add_cases
            left_cases = missing_cases % len(age_groups)
            if left_cases != 0:
                age_i = 0
                while left_cases > 0:
                    select_age_group = age_groups[age_i]
                    condition = (cases_age['Time'] == time) & (cases_age['age_group'] == select_age_group)
                    cases_age.loc[condition, 'val'] += 1
                    age_i += 1
                    left_cases -= 1

        # Getting the propostion of the age wrt total cases
        cases_age_prop.loc[cases_age['Time'] == time, 'val'] = cases_age.loc[cases_age['Time'] == time, 'val'] / tot_val

    cases_age_prop = cases_age_prop.fillna(0)
    # Remove the column of total age counts
    cases_age = cases_age[~cases_age['age_group'].str.contains('All')]
    cases_age_prop = cases_age_prop[~cases_age_prop['age_group'].str.contains('All')]

    return cases_age, cases_age_prop


def construct_finland_age_cases_daily(logger, number_age_groups=9):
    cases, cases_prop = fetch_finland_cases_age_weekly(logger,
                                                       number_age_groups=number_age_groups)
    cases_list = cases.values
    cases_prop_list = cases_prop.values
    dates = np.unique(cases_list[:, 0])

    age_groups = REQUESTS['age_group_mappings'][number_age_groups]['age_groups']
    header = 'Time'
    for age_i in age_groups:
        header += ';%s' % age_i
    final_lines_day = [header, ]
    final_lines_prop = [header, ]
    for date in dates:
        cases_week = cases_list[np.where(cases_list[:, 0] == date)]
        cases_day = np.copy(cases_week)
        cases_daily_vals = divide_equally_week_values(cases_day[:, 2])

        cases_week_prop = cases_prop_list[np.where(cases_prop_list[:, 0] == date)]

        monday_of_week = transform_thl_week_datetime(date)
        # Augment by day, start on monday and finish sunday (7 days)
        for day in range(7):
            date_i = monday_of_week + datetime.timedelta(days=day)
            date_str = date_i.strftime('%Y-%m-%d')
            line_day = [date_str, ]
            line_prop = [date_str, ]
            for i, age_i in enumerate(age_groups):
                line_day.append(str(cases_daily_vals[i, day]))
                line_prop.append(str(cases_week_prop[i, 2]))
            line_str_day = ';'.join(line_day)
            line_str_prop = ';'.join(line_prop)
            final_lines_day.append(line_str_day)
            final_lines_prop.append(line_str_prop)

    complete_csv_day = '\n'.join(final_lines_day)
    complete_csv_prop = '\n'.join(final_lines_prop)

    buffer_for_pandas = StringIO(complete_csv_day)
    age_cases_daily = pd.read_csv(buffer_for_pandas, sep=";")
    buffer_for_pandas = StringIO(complete_csv_prop)
    age_cases_prop = pd.read_csv(buffer_for_pandas, sep=";")
    logger.debug('Constructed pandas dataframe')

    return age_cases_daily, age_cases_prop


def construct_cases_age_erva_daily(logger, number_age_groups=9):
    _, cases_age_prop = construct_finland_age_cases_daily(logger,
                                                          number_age_groups=number_age_groups)
    cases_erva = fetch_thl_cases_erva_daily(logger)

    cases_age_prop_list = cases_age_prop.values

    cases_by_age_erva = []
    for row_age_prop in cases_age_prop_list:
        time, *probs = row_age_prop

        cases_erva_date = cases_erva.loc[cases_erva['Time'] == time, ]
        cases_erva_date = cases_erva_date.values
        for cases_line in cases_erva_date:
            _, erva, cases = cases_line
            cases_ages = cases*np.array(probs)

            new_line = [time, erva, *cases_ages]
            cases_by_age_erva.append(new_line)

    age_groups = REQUESTS['age_group_mappings'][number_age_groups]['age_groups']
    columns = ['Time', 'erva']
    for age_i in age_groups:
        columns.append(age_i)
    cases_by_age_erva = pd.DataFrame(data=cases_by_age_erva, columns=columns)

    return cases_by_age_erva


def compartment_values_daily(logger, erva_pop_file, filename=None,
                             inf_period=7, a=2.46, lat_period=2,
                             number_age_groups=9):
    cases_by_age_erva = construct_cases_age_erva_daily(logger,
                                                       number_age_groups=number_age_groups)

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

    for day_t in range(days):
        omega = day_t-inf_period
        if omega < 0:
            omega = 0
        cases_in_period = cases_erva_age_npy[omega:day_t, ]
        # Get the total infected in the period and assign to time t
        infectious_detected[day_t, ] = cases_in_period.sum(axis=0)

        recovered_period = cases_erva_age_npy[:omega, ]
        # Get the total recovered and assign them to time t
        recovered_detected[day_t, :] = recovered_period.sum(axis=0)

    k = np.arange(ages) + 1
    upscale_factor = 1 + 9*k**(-a)
    # Broadcasting operation
    upscale_factor = upscale_factor[np.newaxis, np.newaxis, :]

    logger.debug('Multiplied fraction: %s' % (upscale_factor, ))
    infectious_undetected = infectious_detected * upscale_factor
    recovered_undetected = recovered_detected * upscale_factor

    infected_total = infectious_detected + infectious_undetected
    recovered_total = recovered_detected + recovered_undetected

    exposed_total = np.zeros_like(cases_erva_age_npy)
    for day_t in range(days):
        if day_t+lat_period >= days:
            break
        exposed_total[day_t, :] = infected_total[day_t+lat_period, :]

    # Getting the population to get the final Susceptibles
    pop_ervas, _ = static_population_erva_age(logger, erva_pop_file)
    pop_ervas = pop_ervas[~pop_ervas['erva'].str.contains('All')]
    pop_ervas = pop_ervas.sort_values(['erva'])
    pop_ervas_npy = pop_ervas['Total'].values
    pop_ervas_npy = pop_ervas_npy.reshape(num_ervas, ages)

    # To prepare for broadcasting operation
    pop_ervas_npy = pop_ervas_npy[np.newaxis, :]

    susceptible = np.zeros_like(cases_erva_age_npy)
    susceptible = pop_ervas_npy - exposed_total - infected_total - recovered_total

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
                'infected': infected_total[:, erva_i, age_i],
                'exposed': exposed_total[:, erva_i, age_i],
                'recovered': recovered_total[:, erva_i, age_i],
            }
            erva_age_dataframe = pd.DataFrame(data=dataframe_data)
            complete_dataframe = complete_dataframe.append(erva_age_dataframe)

    if filename is not None:
        complete_dataframe.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return complete_dataframe


def full_epidemic_state_finland(logger, erva_pop_file, filename=None,
                                number_age_groups=9):
    compart_df = compartment_values_daily(logger, erva_pop_file,
                                          number_age_groups=number_age_groups)
    vacc_df = construct_thl_vaccines_erva_daily(logger,
                                                number_age_groups=number_age_groups)
    epidemic_state = pd.merge(compart_df, vacc_df,
                              on=['date', 'erva', 'age'],
                              how='left')
    # Merge will left missing values with NaNs. Filled them with 0
    epidemic_state = epidemic_state.fillna(0)
    epidemic_state = epidemic_state.astype({'First dose': 'int32',
                                            'Second dose': 'int32'})

    epidemic_state['First dose cumulative'] = epidemic_state.groupby(['erva', 'age'])['First dose'].cumsum()
    epidemic_state['Second dose cumulative'] = epidemic_state.groupby(['erva', 'age'])['Second dose'].cumsum()

    epidemic_state['susceptible'] = epidemic_state['susceptible'] - epidemic_state['First dose cumulative']

    if filename is not None:
        epidemic_state.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return epidemic_state


if __name__ == "__main__":
    # Select data directory
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # Get a logger of the events
    logfile = os.path.join(curr_dir, 'logs_fetch_data.log')
    numeric_log_level = getattr(logging, "DEBUG", None)
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
        full_epidemic_state_finland(logger, erva_pop_file, out_csv_filename, number_age_groups=9)

        out_csv_filename = os.path.join(out_dir, 'epidemic_finland_8.csv')
        full_epidemic_state_finland(logger, erva_pop_file, out_csv_filename, number_age_groups=8)
    except Exception:
        logger.exception("Fatal error in main loop")
