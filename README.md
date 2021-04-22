# Vaccination strategies in Finland.

The purpose of this repository is to generate the initial states of the epidemic in Finland. Then use the generated initial states to run forward simulations with different vaccination strategies.

Organization of the repository:
- `fetch_data.py`: API calls and parsing of the data to construct the initial state of the epidemic.
- `initial_states.py`: Uses the functions in `fetch_data.py` to generate CSV files with the final epidemic state.
- `forward_integration.py`: Code to calculate the parameters of the model (except for `beta`) and run the forward simulations with a vaccination strategy.
- `estimating_beta.py`: Constructs the NGM and calculates the `beta` parameter given a `R_eff` value.
- `plot_forward_simulations.ipynb`: Uses the code in `forward_integration.py` to simulate, plot and compare different vaccination strategies.
- `env_var.py`: The static parameters of the epidemic as well as some other parameters to construct the initial states are stored here.
- `read_data.ipynb`: Small example on how the generated CSV files by `initial_states.py` can be used.

Data:
- `out/epidemic_finaland_*.csv`: CSV files with the inital state of the epidemic for 8 and 9 age groups.
- `stats/erva_population_age_2020.csv`: The population by ERVA and age group in 2020. In this file the age groups are of 5 years, it is aggregated to get the final counts for each specific number of age groups (8 or 9).

## Requirements
Developed and tested under
```sh
Python 3.7.4
```

## Installation
1. (Optional) It is recommended to create a virtual environment.
2. Install the packages dependencies.
```sh
pip install -r requirements.txt
```

## Usage
To generate the initial state of epidemic for 8 and 9 age groups
```sh
python initial_states.py 
```

Check out `plot_forward_simulations.ipynb` to some nice visualizations.
