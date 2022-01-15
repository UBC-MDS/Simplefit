# simplefit

A python package that cleans the data, does basic EDA and provides an insight to baseline and basic Classification and Regression models, namely Logistic Regression and Ridge respectively
<br>

### Overview
This package helps data scientists to clean the data, perform basic EDA, visualize graphical interpretations and analyse performance of the baseline model and basic Classification or Regression models, namely Logistic Regression, Ridge on their data.
<br>

### Functions
---
<br>
| Function Name | Input | Output | Description |
|-----------|------------|---------------|------------------|
| cleaner   | `dataframe`  |  list of 3 dataframes | Loads and cleans the dataset, removes NA rows, strip extra white spaces, etc|
| get_eda | `dataframe`, `dist_cols`, `pair_cols`, `corr_method`,`text_col`,`class_label` | list of plot objects from EDA | Creates common exploratory analysis visualizations on numeric and categorical columns in the dataset which are provided to it. |
|regressor| `train_df`, `target_col`, `numeric_feats`, `categorical_feats`, `text_col`, `cv`| `dataframe` |Preprocesses the data, fits baseline model(`Dummy Regressor`) and `Ridge` with default setup and returns model scores  |
|classifier|`train_df`, `target_col`, `numeric_feats`, `categorical_feats`, `text_col`, `cv` |`dataframe`|Preprocesses the data, fits baseline model(`Dummy Classifier`) and `Logistic Regression` with default setup and returns model scores|

<br>
### Our Package in the Python Ecosystem
---
## Installation

```bash
$ pip install simplefit
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

This python package was developed by Mohammadreza Mirzazadeh, Zihan Zhou, Navya Dahiya, and Sanchit Singh. The team is from the Master of Data Science program at the University of the British Columbia.

## License

`simplefit` was created by Reza Zoe Navya Sanchit. It is licensed under the terms of the MIT license.

## Credits

`simplefit` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
