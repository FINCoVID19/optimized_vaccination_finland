import os
import requests
import json
from io import StringIO
import pandas as pd
import numpy as np
import logging
import datetime
# from scipy.stats import multinomial
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


def static_population_erva_age(logger, csv_file):
    population_age_df = pd.read_csv(csv_file, sep=";", encoding='utf-8')
    logger.debug('Constructed pandas dataframe')

    population_age_df = population_age_df.drop(columns=['Males', 'Females'])

    population_age_df = population_age_df[~population_age_df['Age'].str.contains('Total')]
    population_age_df['Total'] = population_age_df['Total'].astype('int32')

    pop_age_prop = population_age_df.copy()
    ervas = pd.unique(pop_age_prop['erva'])
    for erva in ervas:
        erva_counts = pop_age_prop.loc[pop_age_prop['erva'] == erva, 'Total'].values
        tot_erva = np.sum(erva_counts)
        pop_age_prop.loc[pop_age_prop['erva'] == erva, 'Total'] /= tot_erva

    return population_age_df, pop_age_prop


def fetch_thl_vaccines_erva_weekly(logger, filename=None):
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

    age_group_mapping = REQUESTS['age_group_mapping']
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

    if filename is not None:
        vaccinated_erva.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return vaccinated_erva


def fetch_thl_vaccines_erva_daily(logger, filename=None):
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

    hcd_erva_mapping = REQUESTS['hcd_erva_mapping']
    age_group_mapping = REQUESTS['age_group_mapping']

    all_lines = response.text.split('\n')
    final_lines = []
    headers = all_lines[0].strip()
    headers = headers + ';erva'
    final_lines.append(headers)
    # Reading line by line to fix daily dates manually
    for i in range(1, len(all_lines)):
        line = all_lines[i].strip().split(';')
        # If the line is empty skip iteration
        if len(line) == 1:
            continue
        age, vacc_dose, time, hcd_name, val = line
        # If the line is a global computation skip iteration
        if time == 'All times':
            continue

        # Get the hcd code for the area
        erva = str(hcd_erva_mapping[hcd_name])
        age_group = str(age_group_mapping[age])

        week_number = int(time[-2:])
        year = time[5:9]
        # Making correction to match Python date to THL dates
        if year == '2020':
            week_number = week_number - 1
        new_time = 'Year %s Week %02.f-1' % (year, week_number)
        # Get monday date based on the week number
        # This needs to be done bc THL does not provide daily counts only weekly
        monday_of_week = datetime.datetime.strptime(new_time, "Year %Y Week %W-%w")

        # Get the value of the week
        if val == '':
            val = 0
        else:
            val = int(val)

        # Get an estimation of the daily vaccinations with the average
        avg_val_day = val/7
        avg_str = str(avg_val_day)

        # Augment by day, start on monday and finish sunday (7 days)
        for day in range(7):
            date = monday_of_week + datetime.timedelta(days=day)
            date_str = date.strftime('%Y-%m-%d')
            line = [age_group, vacc_dose, date_str, hcd_name, avg_str, erva]
            line_str = ';'.join(line)
            final_lines.append(line_str)

    complete_csv = '\n'.join(final_lines)
    buffer_for_pandas = StringIO(complete_csv)
    vaccinated_df = pd.read_csv(buffer_for_pandas, sep=";")
    logger.debug('Constructed pandas dataframe')

    columns_agg_erva = ['Age',
                        'Vaccination dose',
                        'Time',
                        'erva']
    vaccinated_erva = vaccinated_df.groupby(by=columns_agg_erva, as_index=False).sum()

    vaccinated_erva = vaccinated_erva[['erva', 'Time', 'Age', 'Vaccination dose', 'val']]
    vaccinated_erva = vaccinated_erva.sort_values(by=['erva', 'Time', 'Age', 'Vaccination dose'])
    logger.debug('Sorting by ERVA')

    if filename is not None:
        vaccinated_erva.to_csv(filename, index=False)
        logger.info('Results written to: %s' % (filename, ))

    return vaccinated_erva


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


def fetch_finland_cases_age_weekly(logger):
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

    age_group_mapping = REQUESTS['age_group_mapping_cases']
    new_cases_df['age_group'] = new_cases_df.apply(lambda row: age_group_mapping[row['Age']], axis=1)
    logger.debug('Augmented data age groups')

    cases_age = new_cases_df.groupby(by=['Time', 'age_group'], as_index=False).sum()
    logger.debug('Keeping only age groups')

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
            # Adding difference in missing to group 15-64
            condition = (cases_age['Time'] == time) & (cases_age['age_group'] == '15-64')
            cases_age.loc[condition, 'val'] += missing_cases

        # Getting the propostion of the age wrt total cases
        cases_age_prop.loc[cases_age['Time'] == time, 'val'] = cases_age.loc[cases_age['Time'] == time, 'val'] / tot_val

    cases_age_prop = cases_age_prop.fillna(0)
    # Remove the column of total age counts
    cases_age = cases_age[~cases_age['age_group'].str.contains('All')]
    cases_age_prop = cases_age_prop[~cases_age_prop['age_group'].str.contains('All')]

    return cases_age, cases_age_prop


def fetch_finland_age_cases_daily(logger):
    cases, cases_prop = fetch_finland_cases_age_weekly(logger)
    cases_list = cases.values
    cases_prop_list = cases_prop.values
    dates = np.unique(cases_list[:, 0])

    header = 'Time;0-14;15-64;65+'
    final_lines_day = [header, ]
    final_lines_prop = [header, ]
    for date in dates:
        cases_week = cases_list[np.where(cases_list[:, 0] == date)]
        cases_day = np.copy(cases_week)
        cases_day[:, 2] = cases_day[:, 2]/7

        cases_week_prop = cases_prop_list[np.where(cases_prop_list[:, 0] == date)]

        monday_of_week = transform_thl_week_datetime(date)
        # Augment by day, start on monday and finish sunday (7 days)
        for day in range(7):
            date_i = monday_of_week + datetime.timedelta(days=day)
            date_str = date_i.strftime('%Y-%m-%d')
            line_day = [date_str,
                        str(cases_day[0, 2]),
                        str(cases_day[1, 2]),
                        str(cases_day[2, 2])]
            line_prop = [date_str,
                         str(cases_week_prop[0, 2]),
                         str(cases_week_prop[1, 2]),
                         str(cases_week_prop[2, 2])]
            line_str_day = ';'.join(line_day)
            line_str_prop = ';'.join(line_prop)
            logger.debug(line_str_prop)
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


def construct_cases_age_erva_daily(logger):
    _, cases_age_prop = fetch_finland_age_cases_daily(logger)
    cases_erva = fetch_thl_cases_erva_daily(logger)

    cases_age_prop_list = cases_age_prop.values

    cases_by_age_erva = []
    for row_age_prop in cases_age_prop_list:
        time, *probs = row_age_prop

        cases_erva_date = cases_erva.loc[cases_erva['Time'] == time, ]
        cases_erva_date = cases_erva_date.values
        for cases_line in cases_erva_date:
            _, erva, cases = cases_line
            sampled_cases_ages = np.random.multinomial(cases, probs)

            new_line = [time, erva, *sampled_cases_ages]
            cases_by_age_erva.append(new_line)

    columns = ['Time', 'erva', '0-14', '15-64', '65+']
    cases_by_age_erva = pd.DataFrame(data=cases_by_age_erva, columns=columns)

    return cases_by_age_erva


def seed_epidemic_erva(logger, filename, seed_period=7, a=0.5, lat_period=2):
    matrix_cases, matrix_ervas, matrix_dates = fetch_thl_cases_erva_daily(logger)
    infectious_detected = np.zeros_like(matrix_cases)
    recovered_detected = np.zeros_like(matrix_cases)

    days = len(matrix_dates)
    for day_t in range(days):
        omega = day_t-seed_period
        if omega < 0:
            omega = 0
        cases_in_period = matrix_cases[omega:day_t, ]
        # Get the total infected in the period and assign to time t
        infectious_detected[day_t, :] = cases_in_period.sum(axis=0)

        recovered_period = matrix_cases[:omega, ]
        # Get the total recovered and assign them to time t
        recovered_detected[day_t, :] = recovered_period.sum(axis=0)

    logger.debug('Multiplied fraction: %f' % ((1-a)/a, ))
    infectious_undetected = np.round(((1-a)/a)*infectious_detected)
    recovered_undetected = np.round(((1-a)/a)*recovered_detected)

    infected_total = infectious_detected + infectious_undetected
    recovered_total = recovered_detected + recovered_undetected

    exposed_total = np.zeros_like(matrix_cases)
    for day_t in range(days):
        if day_t+lat_period >= days:
            break
        exposed_total[day_t, :] = infected_total[day_t+lat_period, :]

    pop_ervas = static_population_erva(logger)
    pop_ervas_ordered = [pop_ervas[erva] for erva in matrix_ervas]
    pop_ervas_ordered = np.array(pop_ervas_ordered)
    # To prepare for broadcasting operation
    pop_ervas_ordered = pop_ervas_ordered[np.newaxis, :]

    susceptible = np.zeros_like(matrix_cases)
    susceptible = pop_ervas_ordered - exposed_total - infected_total - recovered_total

    complete_dataframe = pd.DataFrame()
    for erva_idx in range(len(matrix_ervas)):
        dataframe_data = {
            'date': matrix_dates,
            'erva': [matrix_ervas[erva_idx]]*len(matrix_dates),
            'susceptible': susceptible[:, erva_idx].astype(np.int32),
            'infected detected': infectious_detected[:, erva_idx].astype(np.int32),
            'infected undetected': infectious_undetected[:, erva_idx].astype(np.int32),
            'infected': infected_total[:, erva_idx].astype(np.int32),
            'exposed': exposed_total[:, erva_idx].astype(np.int32),
            'recovered detected': recovered_detected[:, erva_idx].astype(np.int32),
            'recovered undetected': recovered_undetected[:, erva_idx].astype(np.int32),
            'recovered': recovered_total[:, erva_idx].astype(np.int32)
        }
        erva_dataframe = pd.DataFrame(data=dataframe_data)
        complete_dataframe = complete_dataframe.append(erva_dataframe)

    complete_dataframe.to_csv(filename, index=False)
    logger.info('Results written to: %s' % (filename, ))


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

        # out_csv_filename = os.path.join(stats_dir, 'erva_seeds.csv')
        # seed_epidemic_erva(logger, out_csv_filename)

        # out_csv_filename = os.path.join(stats_dir, 'erva_vaccinations.csv')
        # fetch_thl_vaccines_erva_weekly(logger, out_csv_filename)

        # out_csv_filename = os.path.join(stats_dir, 'erva_vaccinations_daily.csv')
        # fetch_thl_vaccines_erva_daily(logger, out_csv_filename)

        # fetch_thl_cases_erva_weekly(logger)

        # age_file = os.path.join(stats_dir, 'erva_population_age_2020.csv')
        # static_population_erva_age(logger, age_file)

        construct_cases_age_erva_daily(logger)
    except Exception:
        logger.exception("Fatal error in main loop")