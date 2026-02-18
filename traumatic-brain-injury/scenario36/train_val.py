#!/usr/bin/env python3
##########################################################################################
# Base Model Author: Jared L. Ostmeyer
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
parser.add_argument('--motif_size', help='Motif size', type=int, default=5)
parser.add_argument('--window_size', help='Window size', type=int, default=5)
parser.add_argument('--min_count', help='Minimum count for cdr3s', type=int, default=2)
parser.add_argument('--control', help='Control group to use for training', type=str, choices=['sedation', 'day1'], default='sedation')
args = parser.parse_args()

##########################################################################################
# Assemble sequences
##########################################################################################

# Settings
#
trim_front = 0
trim_rear = 0

# Settings for using motif representation 
#
motif_size = args.motif_size
window_size = args.window_size

min_count = args.min_count

# Defining control and case subjects
#
if args.control == "sedation":
  Control_cases = ['TBI07PX01', 'TBI08PX01', 'TBI10PX01', 'TBI11PX01', 'TBI12PX01', 'TBI17PX01', 'TBI19PX01', 'TBI24PX01', 'TBI25PX01', 'TBI27PX01']
elif args.control == "day1":
  Control_cases = ['TBI01PX01', 'TBI02PX01', 'TBI03PX01', 'TBI04PX01', 'TBI05PX01', 'TBI06PX01', 'TBI09PX01', 'TBI13PX01', 'TBI14PX01', 'TBI15PX01', 'TBI16PX01', 'TBI18PX01', 'TBI20PX01', 'TBI21PX01', 'TBI22PX01', 'TBI23PX01', 'TBI26PX01', 'TBI28PX01']
else:
  raise ValueError('Invalid control group specified. Choose either "sedation" or "day1".')

# To hold sequences from each subject
#
cases = {}
controls = {}

# Load immune repertoires
#
for path in glob.glob('../dataset_d9/IGH/*.tsv'):
  cdr3s = dp.load_cdr3s(path, min_length=window_size, max_length=42)
  cdr3s = dp.trim_cdr3s(cdr3s, trim_front=trim_front, trim_rear=trim_rear)
  motifs = dp.cdr3s_to_motifs(cdr3s, window_size, motif_size)
  motifs = dp.dropLowCountKmers(motifs, min_count)
  motifs = dp.normalize_sample(motifs)
  subject = path.split('/')[-1].split('.')[0]
  if subject in Control_cases:
    controls[subject] = motifs
  else: 
    cases[subject] = motifs

##########################################################################################
# Assemble datasets
##########################################################################################

# Remove kmers in the controls from the cases
#
cases = ds.removeOverlappingKmers(cases, controls, args.holdouts)

# Load embeddings
#
aminoacids_dict = ds.load_aminoacid_embedding_dict('../../aminoacid-representation/atchley_factors_normalized.csv')

# Convert to numeric representations
#
samples = ds.assemble_samples(cases, controls, aminoacids_dict)
print(cases)

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
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = learning_rate, epochs = 2048, steps_per_epoch = len(samples_train))

# Metrics
#
loss = torch.nn.BCEWithLogitsLoss(reduction='none')  # The loss function is calculated seperately for each fit by setting reduction to none

def accuracy(ls_block, ys_block):  # The binary accuracy is calculated seperate for each fit
  a = torch.nn.Sigmoid()
  ps_block = a(ls_block)
  cs_block = (torch.round(ps_block) == torch.round(ys_block)).to(ys_block.dtype)
  return cs_block

# Setting up best_train_loss for model selection
# and early stopping
#
best_train_loss = float('inf')
best_model_state = None
best_model_epoch = 0
patience = 250
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

  # Compute average training loss & get index of best fit to the training data
  #
  train_loss = es_train.mean().item()
  i_bestfit = torch.argmin(es_train)  # Very important index: selects the best fit to the training data

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
  if train_loss < best_train_loss:
    best_train_loss = train_loss
    best_model_state = copy.deepcopy(msm.state_dict())
    best_model_epoch = epoch
    print(f'Epoch {epoch}: new best training loss = {train_loss:.4f}')


  scheduler.step()    


'''
Wait until done comparing models to implement early stopping?
    # Only advance the patience counter if beyond 1000 epochs. 
    if epoch > 1000:     
        patience_counter = 0
    else:
        patience_counter += 1


  # Early stopping check
  #
  if patience_counter >= patience:
    print(f'Early stopping triggered at epoch {epoch}')
    break
    '''


# Save the best model
#
msm.load_state_dict(best_model_state)
torch.save(msm, args.output + '_model.p')
print(f'best model saved at epoch # {best_model_epoch}')