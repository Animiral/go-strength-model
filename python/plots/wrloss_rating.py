"""
Given an input CSV file describing players' ratings and associated single move point loss values,
plot them in a chart that shows the distribution of mistakes by different rank categories.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import fontconfig

config = {
  'low_rating': 1245,  # 10-kyu
  'high_rating': 1920, # 1-dan
  'gains_level': -.03,
  'blunders_level': .04
}

def read_csv(path):
  df = pd.read_csv(path, header=0, dtype=float)
  df.sort_values('wrloss', inplace=True)

  low_data = df[df['rating'] < config['low_rating']]['wrloss']
  medium_data = df[(df['rating'] >= config['low_rating']) & (df['rating'] <= config['high_rating'])]['wrloss']
  high_data = df[df['rating'] > config['high_rating']]['wrloss']

  # left extra bar
  gains_rows = df['wrloss'] < config['gains_level']
  gains = { 'low': len(low_data[gains_rows]) / len(low_data),
            'medium': len(medium_data[gains_rows]) / len(medium_data),
            'high': len(high_data[gains_rows]) / len(high_data) }

  # right extra bar
  blunders_rows = df['wrloss'] > config['blunders_level']
  blunders = { 'low': len(low_data[blunders_rows]) / len(low_data),
               'medium': len(medium_data[blunders_rows]) / len(medium_data),
               'high': len(high_data[blunders_rows]) / len(high_data) }

  density = gaussian_kde(df['wrloss'])
  low_density = gaussian_kde(low_data)
  medium_density = gaussian_kde(medium_data)
  high_density = gaussian_kde(high_data)
  x = np.linspace(config['gains_level'], config['blunders_level'], 100)
  y = density(x)
  low_y = low_density(x)
  medium_y = medium_density(x)
  high_y = high_density(x)
  return x, y, low_y, medium_y, high_y, gains, blunders

def plot(x, y, low_y, medium_y, high_y, gains, blunders):
  low_color = '#FFB6C1'  # pink
  medium_color = '#77DD77'  # green
  high_color = '#AEC6CF'  # blue
  overall_color = '#707070'  # grey

  scale = max(y)  # highest point in the chart
  plt.figure(figsize=[6.4, 4.8])
  plt.plot(x, y, color=overall_color, linewidth=2, linestyle='--', label='Everyone')
  plt.plot(x, high_y, color=high_color, label=f'High Rating ($>{config["high_rating"]}$)')
  plt.plot(x, medium_y, color=medium_color, label='Medium Rating')
  plt.plot(x, low_y, color=low_color, label=f'Low Rating ($<{config["low_rating"]}$)')
  plt.fill_between(x, 0, high_y, color=high_color, alpha=0.3)
  plt.fill_between(x, 0, medium_y, color=medium_color, alpha=0.3)
  plt.fill_between(x, 0, low_y, color=low_color, alpha=0.3)

  # Adding bars for surprising gains and large blunders
  plt.bar(-.042, gains['low']*scale, width=0.003, color=low_color, alpha=1, align='center')
  plt.bar(-.038, gains['medium']*scale, width=0.003, color=medium_color, alpha=1, align='center')
  plt.bar(-.034, gains['high']*scale, width=0.003, color=high_color, alpha=1, align='center')
  plt.bar(.044, blunders['low']*scale, width=0.003, color=low_color, alpha=1, align='center')
  plt.bar(.048, blunders['medium']*scale, width=0.003, color=medium_color, alpha=1, align='center')
  plt.bar(.052, blunders['high']*scale, width=0.003, color=high_color, alpha=1, align='center')

  plt.xlabel('Winrate Lost')
  plt.ylabel('Density')
  plt.title('Distribution of Winrate Loss')
  plt.legend()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  x, y, low_y, medium_y, high_y, gains, blunders = read_csv(path)
  print('Preparing plot...')
  plot(x, y, low_y, medium_y, high_y, gains, blunders)
