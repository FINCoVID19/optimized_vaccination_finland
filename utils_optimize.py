import logging
import logging.handlers
import multiprocessing


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


def create_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(processName)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S %p'
    )
    handler_file = logging.handlers.RotatingFileHandler(
                    'optimized_vaccination.log',
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
