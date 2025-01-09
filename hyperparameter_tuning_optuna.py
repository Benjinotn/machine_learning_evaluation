# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: SAPE
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Hyper Parameter tuning
#
# In XGBoost models hyper parameters seem to have much more of an effect than when using Random Forrest algo's 
#
# here are some good artiles
#
# - [The Ultimate Guide to XGBoost Parameter tuning](https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/)
# - [XGBoost parameter docs](https://xgboost.readthedocs.io/en/release_1.7.0/parameter.html)
# ___________
# # Hyper Parameter Tuning using Optuna 
#
# I think ```pip install jupyter``` is a good way to remove any warnings
#

# %%
import pandas as pd
import yaml
import xgboost as xgb
from pathlib import Path
import optuna
import time

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %% [markdown]
# To use any functions from out reference libraries we need to gain access as follows:

# %% [markdown]
# ___________
# ## Loading and preparing data
#
# For this optimisation I will work soely on the 80/20 LOSA on LOSA training. Although with a slight difference.
#
# I will use 3 datasets in training the model:
# - train
# - validataion
# - test
#
# The idea is that we need to keep some data (test) untouched after running through so many options for hyper paramter tuning. So we will test hyper paramters on train and validation, then when it comes to testing the model we can use (train+validation) and test to get our final scores.

# %%
# LOAD IN X Y DATA HERE

# AND TEST TRAIN SPLITS

# %% [markdown]
# We will use the first train_test_split to get our training set (60% of the data)

# %% [markdown]
# The second train_test_split on the remaining data will give us our validataion and testing datasets.

# %% [markdown]
# Now we need to convert the datasets into DMatrix's so they are compatible with the functions we will use for hyper parameter training

# %%
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_val, enable_categorical=True)

# to be used as the train dataset with the untouched test dataset
dtrainval = xgb.DMatrix(
    pd.concat([X_train, X_val]),
    label=pd.concat([y_train, y_val]),
    enable_categorical=True,
)


# %% [markdown]
# Finally this is a scoring function for the model written using the XGBoost api (not the sklearn api) and will help us to evaluate our model later.

# %%
def score_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    y_true = dmat.get_label()
    y_pred = model.predict(dmat)
    return mean_squared_error(y_true, y_pred, squared=False)


# %% [markdown]
# ______________
# ## Hyper paramter Tuning actual
#
# In XGBoost models there are two sets of hyperparameters, 'Tree Paramters' these affect the descision trees used and 'boost' paramters these affect the boosting. These parameters should be independent of each other (I haven't tested this, taking on word), so the idea is to optimize the tree paramters with fixed boosting parameters and then to optimize the boosting parameters when we know optimal tree parameters.
#
# ### Stage 1: Tuning Tree Parameters
#
# Initally we will need to set the boost parameters that don't change

# %%
# set out parameters that wont change over parameter search
metric = "rmse"
learning_rate = 0.005
base_params = {
    "objective": "reg:squarederror",
    "eval_metric": metric,
}


# %% [markdown]
# Now we create an objective function, this will be used by optuna to train a model based upon the suggested parameters given with the aim of finding the parameters which give the best model score!
#
#

# %%
# objective function using XGBoost API
def objective(trial):
    params = {
        "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 250),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 25, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
    }

    num_boost_round = 10000
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, f"valid-{metric}"
    )

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=50,
        verbose_eval=0,
        callbacks=[pruning_callback],
    )

    trial.set_user_attr("best_iteration", model.best_iteration)

    return model.best_score


# %% [markdown]
# Now it's time to tune the hyper paramters, we can use an optuna study for this:

# %%
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)

# this sets a timer for how long we should run the optimisations
tic = time.time()
while time.time() - tic < 300:
    study.optimize(objective, n_trials=1)

# %% [markdown]
# We can read the results from this using this handy print out from the article author:
#
# or here for prosperity
# tree_method : hist
# max_depth : 7
# min_child_weight : 56
# subsample : 0.9778608052726199
# colsample_bynode : 0.8493981381526909
# reg_lambda : 0.003983884028288949

# %%
print("Stage 1 ==============================")
print(f"best score = {study.best_trial.value}")
print("boosting params ---------------------------")
print(f"fixed learning rate: {learning_rate}")
# print(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
print("best tree params --------------------------")
for k, v in study.best_trial.params.items():
    print(k, ":", v)

# %%
study.best_trial

# %% [markdown]
# ### Stage 2: Intensifying Boost parameters
#
# here we take the best tree paramters and then figure out the best boosting paramters

# %%
low_learning_rate = 0.001

# setting up our model parameters using the base params and the params from the best trail run
params = {}
params.update(base_params)
params.update(study.best_trial.params)
params["learning_rate"] = low_learning_rate

model_stage2 = xgb.train(
    params=params,
    dtrain=dtrain,
    evals=[(dtrain, "train"), (dval, "valid")],
    early_stopping_rounds=50,
    verbose_eval=0,
)

# %%
print("Stage 2 ==============================")
print(f"best score = {score_model(model_stage2, dval)}")
print("boosting params ---------------------------")
print(f'fixed learning rate: {params["learning_rate"]}')
print(f"best boosting round: {model_stage2.best_iteration}")

# %% [markdown]
# _______
# # Final optimised model

# %%
model_final = xgb.train(
    params=params,
    dtrain=dtrainval,
    num_boost_round=model_stage2.best_iteration,
    verbose_eval=0,
)

# %%
model_stage2.best_iteration

# %%
print("Final Model ==========================")
print(f"test score = {score_model(model_final, dtest)}")
print("parameters ---------------------------")
for k, v in params.items():
    print(k, ":", v)
print(f"num_boost_round: {model_stage2.best_iteration}")
