#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-02-05
# Purpose: Print results of the holdout cross-validation
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import glob
import csv
from scipy.special import xlogy
import numpy as np

##########################################################################################
# Load data
##########################################################################################

costs_train = {}
accuracies_train = {}
tprs_train = {}
tnrs_train = {}
for path in glob.glob('bin/*_ps_train.csv'):
  with open(path, 'r') as stream:
    reader = csv.DictReader(stream)
    costs = []
    accuracies = []
    tprs = []
    tnrs = []
    for row in reader:
      label = float(row['Label'])
      weight = float(row['Weight'])
      prediction = float(row['Prediction'])
      costs.append(
        weight*(-xlogy(label, prediction)-xlogy(1.0-label, 1.0-prediction))/np.log(2.0)
      )
      accuracies.append(
        100.0*weight*np.equal(np.round(label), np.round(prediction)).astype(np.float64)
      )
      if label == 1.0:
        tprs.append(
          100.0*np.equal(np.round(label), np.round(prediction)).astype(np.float64)
        )
      elif label == 0.0:
        tnrs.append(
          100.0*np.equal(np.round(label), np.round(prediction)).astype(np.float64)
        )
      else:
        print('ERROR: Unrecognized value in the label.')
        exit()
  filename = path.split('/')[-1].split('.')[0]
  _, step, _, _ = filename.split('_')
  i = int(step)
  if i not in costs_train:
    costs_train[i] = []
  costs_train[i].append(
    np.sum(costs)
  )
  if i not in accuracies_train:
    accuracies_train[i] = []
  accuracies_train[i].append(
    np.sum(accuracies)
  )
  if len(tprs) > 0:
    if i not in tprs_train:
      tprs_train[i] = []
    tprs_train[i].append(
      np.mean(tprs)
    )
  if len(tnrs) > 0:
    if i not in tnrs_train:
      tnrs_train[i] = []
    tnrs_train[i].append(
      np.mean(tnrs)
    )

costs_val = {}
accuracies_val = {}
tprs_val = {}
tnrs_val = {}
for path in glob.glob('bin/*_ps_val.csv'):
  with open(path, 'r') as stream:
    reader = csv.DictReader(stream)
    costs = []
    accuracies = []
    tprs = []
    tnrs = []
    for row in reader:
      label = float(row['Label'])
      weight = float(row['Weight'])
      prediction = float(row['Prediction'])
      costs.append(
        weight*(-xlogy(label, prediction)-xlogy(1.0-label, 1.0-prediction))/np.log(2.0)
      )
      accuracies.append(
        100.0*weight*np.equal(np.round(label), np.round(prediction)).astype(np.float64)
      )
      if label == 1.0:
        tprs.append(
          100.0*np.equal(np.round(label), np.round(prediction)).astype(np.float64)
        )
      elif label == 0.0:
        tnrs.append(
          100.0*np.equal(np.round(label), np.round(prediction)).astype(np.float64)
        )
      else:
        print('ERROR: Unrecognized value in the label.')
        exit()
  filename = path.split('/')[-1].split('.')[0]
  _, step, _, _ = filename.split('_')
  i = int(step)
  if i not in costs_val:
    costs_val[i] = []
  costs_val[i].append(
    np.sum(costs)
  )
  if i not in accuracies_val:
    accuracies_val[i] = []
  accuracies_val[i].append(
    np.sum(accuracies)
  )
  if len(tprs) > 0:
    if i not in tprs_val:
      tprs_val[i] = []
    tprs_val[i].append(
      np.mean(tprs)
    )
  if len(tnrs) > 0:
    if i not in tnrs_val:
      tnrs_val[i] = []
    tnrs_val[i].append(
      np.mean(tnrs)
    )

##########################################################################################
# Results
##########################################################################################

print(
  'Step',
  'Accuracy_Train', 'TRP_Train', 'TNR_Train', 'Cost_train',
  'Accuracy_Val', 'TRP_Val', 'TNR_Val', 'Cost_Val',
  sep=','
)

for i in sorted(accuracies_train.keys()):
  print(
    i,
    np.mean(accuracies_train[i]), np.mean(tprs_train[i]), np.mean(tnrs_train[i]), np.mean(costs_train[i]),
    np.mean(accuracies_val[i]), np.mean(tprs_val[i]), np.mean(tnrs_val[i]), np.mean(costs_val[i]),
    sep=','
  )