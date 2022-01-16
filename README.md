# simplefit

A python package that cleans the data, does basic EDA and returns scores for basic classification and regression models
<br>

### Overview
This package helps data scientists to clean the data, perform basic EDA, visualize graphical interpretations and analyse performance of the baseline model and basic Classification or Regression models, namely Logistic Regression, Ridge on their data.


### Functions
---
| Function Name | Input                                                                                      | Output                        | Description                                                                                                                          |
|---------------|--------------------------------------------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| cleaner       | `dataframe`                                                                                | list of 3 dataframes          | Loads and cleans the dataset, removes NA rows, strip extra white spaces, etc  and returns clean dataframe along with `data.info()` , `data.describe()` as dataframes                                                     |
| get_eda       | `dataframe`, `dist_cols`, `pair_cols`, `corr_method`,`text_col`,`class_label`              | list of plot objects from EDA | Creates common exploratory analysis visualizations on numeric and categorical columns in the dataset which are provided to it  and returns a list of plots requested by the user      |
| regressor     | `train_df`, `target_col`, `numeric_feats`, `categorical_feats`, `text_col`, `cv`           | `dataframe`                   | Preprocesses the data, fits baseline model(`Dummy Regressor`) and `Ridge` with default setup and returns model scores in the form of a dataframe               |
| classifier    | `train_df` ,  `target_col` ,  `numeric_feats` ,  `categorical_feats` ,  `text_col` ,  `cv` | `dataframe`                   | Preprocesses the data, fits baseline model(`Dummy Classifier`) and `Logistic Regression` with default setup and returns model scores in the form of a dataframe|



### Our Package in the Python Ecosystem
---
There exists a subset of our package as standalone packages, namely [auto-eda](https://pypi.org/project/auto-eda/), [eda-report](https://pypi.org/project/eda-report/), [quick-eda](https://pypi.org/project/quick-eda/), [s11-classifier](https://pypi.org/project/s11-classifier/). But these packages only do the EDA or just the classification using `XGBoostClassifier`. But with our package, we aim to do all the basic steps of a ML pipeline and save the data scientist's time and effort by cleaning, preprocessing, returning grpahical visualisations from EDA and providing an insight about the basic model performances, after which the user can decide which other models to use.


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
