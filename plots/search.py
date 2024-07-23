#!/usr/bin/env python3
# Usage: plots/search.py logs/search
"""
Visualize the marginal distributions p(L|t) of the validation loss L seen in the search under every hyperparameter t.
The data is taken from all training*.txt files in the argument directory.
"""

# import numpy as np
import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import fontconfig

def extract_data_from_file(file_path):
  pattern = re.compile(
    r'it(?P<iteration>\d+)\s+seq\d+\s+\|\s+lr=(?P<lr>[0-9.e+-]+)\s+decay=(?P<decay>[0-9.e+-]+)\s+N=(?P<N>\d+)\s+l=(?P<l>\d+)\s+d=(?P<d>\d+)\s+dq=(?P<dq>\d+)\s+m=(?P<m>\d+)'
  )
  validation_loss_pattern = re.compile(r'validation\s+([0-9.]+)')

  with open(file_path, 'r') as file:
    lines = file.readlines()

  match = pattern.search(lines[0])
  if not match:
    return None

  data = match.groupdict()
  data = {k: float(v) if k in ['lr', 'decay'] else int(v) for k, v in data.items()}
  
  validation_losses = []
  for line in lines[1:]:
    match = validation_loss_pattern.search(line)
    if match:
      validation_losses.append(float(match.group(1)))

  if validation_losses:
    data['min_validation_loss'] = min(validation_losses)
  else:
    data['min_validation_loss'] = None
  
  return data

def gather_data(directory):
  data_list = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.startswith('training') and file.endswith('.txt'):
        file_path = os.path.join(root, file)
        data = extract_data_from_file(file_path)
        if data:
          data_list.append(data)
  
  return pd.DataFrame(data_list)

def plot(df):
  hyperparameters = ['lr', 'decay', 'N', 'l', 'd', 'dq', 'm']

  fig, axs = plt.subplots(2, 4, figsize=(20, 10))
  axs = axs.ravel()

  colors = plt.cm.get_cmap('tab10', len(df)).colors
  shapes = {0: 'v', 1: 'v', 2: 'X', 3: 'X'}

  for i, param in enumerate(hyperparameters):
    ax = axs[i]
    for idx, row in df.iterrows():
      ax.scatter(row[param], row['min_validation_loss'], color=colors[idx], marker=shapes[row['iteration']])

    ax.set_title(f'Validation Loss vs {param}')
    ax.set_xlabel(param)
    ax.set_ylabel('Min Validation Loss')

    if param == 'lr':
      ax.set_xscale('log')
      # ax.set_xlim(0, 0.01)
    if param == 'l':
      ax.set_xticks([1, 2, 3, 4, 5])

  # Hide the last subplot as there are only 7 plots needed
  axs[-1].axis('off')

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  df = gather_data(path)
  # import ace_tools as tools; tools.display_dataframe_to_user(name="hparams", dataframe=df)
  print('Preparing plots...')
  plot(df)
