import sys
import logging
import logging.handlers
import multiprocessing
import argparse
from env_var import EXPERIMENTS


def log_out_minimize(minimize_result):
    result_str = ('%(message)s\t(Exit mode %(status)s)\n'
                  '\tCurrent function value: %(value)s\n'
                  '\tIterations: %(iter)s\n'
                  '\tFunction evaluations: %(evals)s\n'
                  '\tGradient evaluations: %(grad)s') % ({
                    'message': minimize_result.message,
                    'status': minimize_result.status,
                    'value': minimize_result.fun,
                    'evals': minimize_result.nfev,
                    'grad': minimize_result.njev,
                    'iter': minimize_result.nit,
                  })
    return result_str


def create_logger(log_file, log_level=logging.DEBUG):
    logger = multiprocessing.get_logger()
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(processName)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S %p'
    )
    handler_file = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=100e6,
                    backupCount=5
                    )
    handler_console = logging.StreamHandler()
    handler_file.setFormatter(formatter)
    handler_console.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler_file)
    
    if len(logger.handlers) == 1:
        logger.addHandler(handler_console)

    return logger


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Get an optimized vaccination strategy."
    )

    parser.add_argument('--r_experiments', type=float, nargs='+',
                        default=EXPERIMENTS['r_effs'],
                        help='List of R_effs to run the experiments.')

    parser.add_argument('--taus', type=float, nargs='+',
                        default=EXPERIMENTS['taus'],
                        help='List of taus to run the experiments.')

    parser.add_argument('--t0', type=str,
                        default=EXPERIMENTS['t0'],
                        help='Initial date to run the experiments.')

    parser.add_argument('--T', type=int,
                        default=EXPERIMENTS['simulate_T'],
                        help='Time horizon to run the experiments.')

    parser.add_argument('--part_time', type=int,
                        default=EXPERIMENTS['simulate_T'],
                        help=('The total time horizon is going to be divided'
                              ' in  batches part_time days.'))

    parser.add_argument('--num_age_groups', type=int,
                        default=9,
                        choices=[8, 9],
                        help="Get the optimized vaccination for 'num_age_groups'.")

    parser.add_argument("--region", type=str,
                        default='erva',
                        choices=["erva", "hcd"],
                        help="Get the optimized vaccination for this 'region'.")

    parser.add_argument('--hosp_optim', action='store_true',
                        help='If set the number of hospitalizations is optimized.')

    parser.add_argument('--max_execution_hours', type=int,
                        default=24,
                        help='Maximum time in hours to run optimization.')

    parser.add_argument('--test', action='store_true',
                        help='If set just a quick execution is ran.')

    parser.add_argument("--log_file", type=str,
                        default='logs_optimized_vaccination.log',
                        help="Logging file.")

    parser.add_argument("--log_level", type=str,
                        default='INFO',
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="Set logging level.")

    return parser.parse_args(args)
