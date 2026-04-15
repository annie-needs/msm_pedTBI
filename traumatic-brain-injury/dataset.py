#########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2021-11-16
# Environment: Python3
# Purpose: Utilities for converting immune receptor sequences into numeric features
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import csv
import numpy as np

##########################################################################################
# Utilities
##########################################################################################

def load_aminoacid_embedding_dict(path_embedding):

  # Amino acid factors
  #
  names = []
  factors = []
  with open(path_embedding, 'r') as stream:
    for line in stream:
      rows = line.split(',')
      names.append(rows[0])
      factors.append(np.array(rows[1:], dtype=np.float32))
  names = np.array(names)
  factors = np.array(factors)

  # Convert into a dictionary
  #
  aminoacids_dict = { name: factors[i,:] for i, name in enumerate(names) }

  return aminoacids_dict
  
  
def removeOverlappingKmers(cases, controls, holdout = None):

  # Remove any sequences from the cases that appear in the controls
  #
  control_sequences = set()

  for subject, seqs in controls.items():
    if subject != holdout:
      control_sequences.update(seqs.keys())
    else:
      continue
  
  filtered_cases = {}

  for subject, seqs in cases.items():
    if subject == holdout:
      filtered_cases[subject] = seqs
    else: 
      filtered_seqs = {
        seq: qty
        for seq, qty in seqs.items()
        if seq not in control_sequences
      }
      filtered_cases[subject] = filtered_seqs  
      
  return filtered_cases
  

def assemble_samples(cases, controls, aminoacids_dict):

  # Determine tensor dimensions 
  #
  max_steps = -1
  for sequences in cases.values():
    for sequence in sequences.keys():
      if len(sequence) > max_steps:
        max_steps = len(sequence)
  for sequences in controls.values():
    for sequence in sequences.keys():
      if len(sequence) > max_steps:
        max_steps = len(sequence)
  num_factors = len(list(aminoacids_dict.values())[0])

  # Assemble dataset
  #
  samples = []

  for subject in sorted(cases.keys()):

    sequences = cases[subject]

    # Initialize tensors
    #
    xs = np.zeros([ len(sequences), max_steps*num_factors+1 ], dtype=np.float32)

    # Fill tensors
    #
    for i, ( sequence, quantity ) in enumerate(sorted(sequences.items())):
      if '*' in sequence:
        continue
      if '_' in sequence:
        continue
      for j, aa in enumerate(sequence):
        xs[i,num_factors*j:num_factors*(j+1)] = aminoacids_dict[aa]
      xs[i,-1] = np.log(quantity)

    u = np.mean(xs[:,-1])
    v = np.var(xs[:,-1])
    if v > 1e-10:
      xs[:,-1] = (xs[:,-1]-u)/np.sqrt(v)
    else:
      xs[:,-1] = 0.0  # Setting the feature to 0 since vairance ~ 0 

    samples.append(
      {
        'subject': subject,
        'features': xs,
        'label': 1.0
      }
    )

  for subject in sorted(controls.keys()):

    sequences = controls[subject]

    # Initialize tensors
    #
    xs = np.zeros([ len(sequences), max_steps*num_factors+1 ], dtype=np.float32)

    # Fill tensors
    #
    for i, ( sequence, quantity ) in enumerate(sorted(sequences.items())):
      for j, aa in enumerate(sequence):
        xs[i,num_factors*j:num_factors*(j+1)] = aminoacids_dict[aa]
      xs[i,-1] = np.log(quantity)

    u = np.mean(xs[:,-1])
    v = np.var(xs[:,-1])
    if v > 1e-10:
      xs[:,-1] = (xs[:,-1]-u)/np.sqrt(v)
    else:
      xs[:,-1] = 0.0  # Setting the feature to 0 since vairance ~ 0 


    samples.append(
      {
        'subject': subject,
        'features': xs,
        'label': 0.0
      }
    )

  return samples

def split_samples(samples, holdouts):
  samples_train = []
  samples_val = []
  for sample in samples:
    if sample['subject'] not in holdouts:
      samples_train.append(sample)
    else:
      samples_val.append(sample)
  return samples_train, samples_val

def split_samples_TrainAllPX(samples, holdouts):
  samples_train = []
  samples_val = []
  for sample in samples:
    # put all samples in the sample_train set. 
    samples_train.append(sample)
    # Put a holdout sample in the sample_val set to maintain model compatibility with the original code. Model selection is based on best training performance so the sample_val set is not used for model selection.
    if sample['subject'] in holdouts:
      samples_val.append(sample)
  return samples_train, samples_val

def weight_samples(samples):
  num_case = 0
  num_control = 0
  for sample in samples:
    if sample['label'] > 0.5:
      num_case += 1
    else:
      num_control += 1
  for sample in samples:
    if sample['label'] > 0.5:
      sample['weight'] = 0.5/num_case if num_control > 0 else 1.0/num_case
    else:
      sample['weight'] = 0.5/num_control if num_case > 0 else 1.0/num_control
  return samples

def normalize_samples(samples_train, samples_holdout):

  # Calculate normalization statistics from the training samples
  # vs = variance
  #
  us = 0.0
  us2 = 0.0
  for sample in samples_train:
    xs_sample = sample['features']
    us_sample = np.mean(xs_sample, axis=0)
    us2_sample = np.mean(xs_sample**2, axis=0)
    us += sample['weight']*us_sample
    us2 += sample['weight']*us2_sample
  vs = us2-us**2

  # Normalize the training samples
  #
  for sample in samples_train:
    xs_sample = sample['features']
    xs_sample = (xs_sample-us)/np.sqrt(vs)
    sample['features'] = xs_sample

  # Normalize the holdout samples
  #
  for sample in samples_holdout:
    xs_sample = sample['features']
    xs_sample = (xs_sample-us)/np.sqrt(vs)
    sample['features'] = xs_sample

  return samples_train, samples_holdout
    
def debug_permute_labels(samples):
  labels = []
  for sample in samples:
    labels.append(sample['label'])
  np.random.shuffle(labels)
  for sample, label in zip(samples, labels):
    sample['label'] = label
  return samples
