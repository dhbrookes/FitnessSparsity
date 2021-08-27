# Code for "On the sparsity of fitness functions and implications for learning" (2021)

This code runs the analyses presenting in the pre-print, which is available [here](https://www.biorxiv.org/content/10.1101/2021.05.24.445506v1).
In order to run any of the code, you must install the dependencies via

```
$ conda env create -f environment.yml
```
which will create an anaconda environment named `fitness_env`.

## Plotting scripts

All plots presented in the main text and SI of the paper are made by running the scripts in the `scripts` folder. 
Many of these scripts load pre-calculated results that are stored in the `results` folder.

## Running analyses

There are two major analyses that one may wish to re-run, rather than loading the pre-calculated results. First, LASSO estimates of 
the empirical and quasi-empirical fitness functons can be re-calculated using the `scripts/run_lasso_exp.py` script, which is run as:
```
$ python run_lasso_exp.py <data_name>
```
where `<data_name>` can be either `tagbfp`, `his3p_small`, `his3p_big`, or `rna`. 

Next, the numerical tests to calculate a suitable value of the $C$ constant can be re-run using the `scripts/run_C_calculation.py` script. Please
run 
```
$ python run_C_calculation.py --help
```
for information on the command line arguments required to run the script.

## Sampling GNK fitness functions

Although it is not used in any of our analyses, the function `sample_gnk_fitness_function` in `src/gnk_model.py` may be of interest to some. 
This function samples a fitness function from the GNK model given the sequence length `L`, alphabet size `q`, and a set of neighborhoods `V`. 
`V` can be set to one of the string values `"random"`, `"adjacent"` or `"block"` if one wishes to use one of the standard neighborhood schemes; in this case,
the optional `K` parameter must be also be input. Otherwise, `V` is a list of 1-indexed lists where the $i$-th list corresponds to the neighborhood
of position $i$. An example where we sample a $L=3$, $q=2$ fitness function with custom neighborhoods is below:
```
V = [[1, 2], [2], [1, 2, 3]]
f = sample_gnk_fitness_function(3, 2, V=V)
```
