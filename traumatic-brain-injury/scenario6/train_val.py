#!/usr/bin/env python3
##########################################################################################
# Base Modle Author: Jared L. Ostmeyer
# Date Started: 2021-11-16
# Purpose: Train and validate a classifier for immune repertoires

# Updated for TBI Project by: Annie Needs
# Updates started: 2025-12-01
# Purpose: update the msm scripts to work with the TBI BCR datasets
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################
import sys
sys.path.append('..') # adds higher directory to python modules path

import argparse
import csv
import glob
import dataplumbing as dp
import dataset as ds
import numpy as np
import torch
import copy

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--holdouts', help='Holdout samples', type=str, nargs='+', required=True)
parser.add_argument('--restart', help='Basename for restart files', type=str, default=None)
parser.add_argument('--output', help='Basename for output files', type=str, required=True)
parser.add_argument('--seed', help='Seed value for randomly initializing fits', type=int, default=1)
parser.add_argument('--device', help='Examples are cuda:0 or cpu', type=str, default='cuda:0')
parser.add_argument('--num_fits', help='Number of fits to the training data', type=int, default=2**17)
args = parser.parse_args()

##########################################################################################
# Assemble sequences
##########################################################################################

# Settings
#
trim_front = 0
trim_rear = 0

kmer_size = 6

# Defining control and case subjects
#
Control_cases = ['TBI07PX01', 'TBI08PX01', 'TBI10PX01', 'TBI11PX01', 'TBI12PX01', 'TBI17PX01', 'TBI19PX01', 'TBI24PX01', 'TBI25PX01', 'TBI27PX01']

# To hold sequences from each subject
#
cases = {}
controls = {}

# Load immune repertoires
#
for path in glob.glob('../dataset_d2/IGH/*.tsv'):
  cdr3s = dp.load_cdr3s(path, min_length=kmer_size+trim_front+trim_rear, max_length=32)
  cdr3s = dp.trim_cdr3s(cdr3s, trim_front=trim_front, trim_rear=trim_rear)
  kmers = dp.cdr3s_to_kmers(cdr3s, kmer_size)
  kmers = dp.normalize_sample(kmers)
  subject = path.split('/')[-1].split('.')[0]
  if subject in Control_cases:
    controls[subject] = kmers
  else: 
    cases[subject] = kmers

##########################################################################################
# Assemble datasets
##########################################################################################

# Load embeddings
#
aminoacids_dict = ds.load_aminoacid_embedding_dict('../../aminoacid-representation/atchley_factors_normalized.csv')

# Convert to numeric representations
#
samples = ds.assemble_samples(cases, controls, aminoacids_dict)

# Split into a training and validation cohort
#
samples_train, samples_val = ds.split_samples(samples, args.holdouts)

# Weight samples
#
samples_train = ds.weight_samples(samples_train)
samples_val = ds.weight_samples(samples_val)

# Normalize features
#
samples_train, samples_val = ds.normalize_samples(samples_train, samples_val)

##########################################################################################
# Assemble tensors
##########################################################################################

# Settings
#
device = torch.device(args.device)

# Convert numpy arrays to pytorch tensors
#
for sample in samples_train:
  sample['features'] = torch.from_numpy(sample['features']).to(device)
  sample['label'] = torch.tensor(sample['label']).to(device)
  sample['weight'] = torch.tensor(sample['weight']).to(device)

# Convert numpy arrays to pytorch tensors
#
for sample in samples_val:
  sample['features'] = torch.from_numpy(sample['features']).to(device)
  sample['label'] = torch.tensor(sample['label']).to(device)
  sample['weight'] = torch.tensor(sample['weight']).to(device)

##########################################################################################
# Model
##########################################################################################

# Settings
#
num_features = samples_train[0]['features'].shape[1]
num_fits = args.num_fits

torch.manual_seed(args.seed)

# Function for initializing the weights of the model
#
def init_weights():
  return torch.cat(
    [
      0.5**0.5*torch.rand([ num_features-1, num_fits ])/(num_features-1.0)**0.5,  # Weights for the Atchley factors
      0.5**0.5*torch.rand([ 1, num_fits ])/(1.0)**0.5,  # Weight for the abundance term
    ],
    0
  )

# Class defining the model
#
class MaxSnippetModel(torch.nn.Module):
  def __init__(self):
    super(MaxSnippetModel, self).__init__()
    self.linear = torch.nn.Linear(num_features, num_fits)
    with torch.no_grad():
      self.linear.weights = init_weights()  # Initialize the weights
  def forward(self, x):
    ls = self.linear(x)
    ms, _ = torch.max(ls, axis=0)
    return ms

# Instantiation of the model
#
msm = MaxSnippetModel()

# Turn on GPU acceleration
#
msm.to(device)

##########################################################################################
# Metrics and optimization
##########################################################################################

# Settings
#
learning_rate = 0.01

# Optimizer
#
optimizer = torch.optim.Adam(msm.parameters(), lr=learning_rate)  # Adam is based on gradient descent but better

# LR Scheduler
#
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = learning_rate, total_steps = 1000)

# Metrics
#
loss = torch.nn.BCEWithLogitsLoss(reduction='none')  # The loss function is calculated seperately for each fit by setting reduction to none

def accuracy(ls_block, ys_block):  # The binary accuracy is calculated seperate for each fit
  a = torch.nn.Sigmoid()
  ps_block = a(ls_block)
  cs_block = (torch.round(ps_block) == torch.round(ys_block)).to(ys_block.dtype)
  return cs_block

# Setting up best_val_loss for model selection
# and early stopping
#
best_val_loss = float('inf')
best_model_state = None
patience = 150
patience_counter = 0

##########################################################################################
# Fit and evaluate model
##########################################################################################

# Settings
#
num_epochs = 2048

# Restore saved models
#
if args.restart is not None:
  msm = torch.load(args.output+'_model.p')

# Each iteration represents one batch
#
for epoch in range(0, num_epochs):

  # Reset the gradients
  #
  optimizer.zero_grad()

  # Training block
  #
  es_train = 0.0  # Cross-entropy error
  as_train = 0.0  # Accuracy

  for sample in samples_train:

    xs_block = sample['features']
    ys_block = torch.tile(sample['label'], [ num_fits ])
    w_block = sample['weight']

    ls_block = msm(xs_block)
    sample['predictions'] = torch.sigmoid(ls_block)

    es_block = w_block*loss(ls_block, ys_block)  # The loss function is calculated seperately for each fit
    as_block = w_block*accuracy(ls_block, ys_block)  # The binary accuracy is calculated seperate for each fit

    es_train += es_block.detach()
    as_train += as_block.detach()

    e_block = torch.sum(es_block)
    e_block.backward()
  
  optimizer.step()

  # Validation block
  #
  es_val = 0.0
  as_val = 0.0

  with torch.no_grad():

    for sample in samples_val:

      xs_block = sample['features']
      ys_block = torch.tile(sample['label'], [ num_fits ])
      w_block = sample['weight']

      ls_block = msm(xs_block)
      sample['predictions'] = torch.sigmoid(ls_block)

      es_block = w_block*loss(ls_block, ys_block)  # The loss function is calculated seperately for each fit
      as_block = w_block*accuracy(ls_block, ys_block)  # The binary accuracy is calculated seperate for each fit

      es_val += es_block.detach()
      as_val += as_block.detach()

  # Compute average validation loss & get index of best fit to the validation data
  #
  val_loss = es_val.mean().item()
  i_bestfit = torch.argmin(es_val)  # Very important index selects the best fit to the validation data


  # Print report
  #
  print(
    'Epoch:', epoch,
    'Accuracy (train):', round(100.0*float(as_train[i_bestfit]), 2), '%',
    'Accuracy (val):', round(100.0*float(as_val[i_bestfit]), 2), '%',
    flush=True
  )

  # Save parameters and results from the best fit to the training data
  #
  if epoch%32 == 0:
    ws = msm.linear.weights.detach().numpy()
    bs = msm.linear.bias.cpu().detach().numpy()
    np.savetxt(args.output+'_'+str(epoch)+'_ws.csv', ws[:,i_bestfit])
    np.savetxt(args.output+'_'+str(epoch)+'_b.csv', bs[[i_bestfit.cpu()]])
    with open(args.output+'_'+str(epoch)+'_ms_train.csv', 'w') as stream:
      print('Cross Entropy (bits)', 'Accuracy (%)', sep=',', file=stream)
      print(float(es_train[i_bestfit])/np.log(2.0), 100.0*float(as_train[i_bestfit]), sep=',', file=stream)
    with open(args.output+'_'+str(epoch)+'_ms_val.csv', 'w') as stream:
      print('Cross Entropy (bits)', 'Accuracy (%)', sep=',', file=stream)
      print(float(es_val[i_bestfit])/np.log(2.0), 100.0*float(as_val[i_bestfit]), sep=',', file=stream)
    with open(args.output+'_'+str(epoch)+'_ps_train.csv', 'w') as stream:
      print('Subject', 'Label', 'Weight', 'Prediction', sep=',', file=stream)
      for sample in samples_train:
        print(sample['subject'], float(sample['label'].detach()), float(sample['weight'].detach()), float(sample['predictions'][i_bestfit].detach()), sep=',', file=stream)
    with open(args.output+'_'+str(epoch)+'_ps_val.csv', 'w') as stream:
      print('Subject', 'Label', 'Weight', 'Prediction', sep=',', file=stream)
      for sample in samples_val:
        print(sample['subject'], float(sample['label']), float(sample['weight']), float(sample['predictions'][i_bestfit]), sep=',', file=stream)
  
  # Check if val_loss improved for model selection and early stopping
  #
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = copy.deepcopy(msm.state_dict())
    patience_counter =0
    print(f'Epoch {epoch}: new best val loss = {val_loss:.4f}')
  else:
    patience_counter += 1

  scheduler.step()

  # Early stopping check
  #
  if patience_counter >= patience:
    print(f'Early stopping triggered at epoch {epoch}')
    break

# Save the best model
#
msm.load_state_dict(best_model_state)
torch.save(msm, args.output + '_model.p')