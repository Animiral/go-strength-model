"""
Given an input CSV file describing players' ratings and associated single move point loss values,
plot them in a chart that shows the distribution of mistakes by different rank categories.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

config = {
  'low_rating': 1200,
  'high_rating': 1800,
  'gains_level': -2,
  'blunders_level': 15
}

def read_csv(path):
  df = pd.read_csv(path, names=['rating', 'loss'])
  df.sort_values('loss', inplace=True)

  low_data = df[df['rating'] < config['low_rating']]['loss']
  medium_data = df[(df['rating'] >= config['low_rating']) & (df['rating'] <= config['high_rating'])]['loss']
  high_data = df[df['rating'] > config['high_rating']]['loss']

  # left extra bar
  gains_rows = df['loss'] < config['gains_level']
  gains = { 'low': len(low_data[gains_rows]) / len(low_data),
            'medium': len(medium_data[gains_rows]) / len(medium_data),
            'high': len(high_data[gains_rows]) / len(high_data) }

  # right extra bar
  blunders_rows = df['loss'] > config['blunders_level']
  blunders = { 'low': len(low_data[blunders_rows]) / len(low_data),
               'medium': len(medium_data[blunders_rows]) / len(medium_data),
               'high': len(high_data[blunders_rows]) / len(high_data) }

  # blunders_low = len(low_data[low_data['loss'] > blunder_level])
  # blunders_medium = len(medium_data[medium_data['loss'] > blunder_level])
  # blunders_high = len(high_data[high_data['loss'] > blunder_level])
  # blunders_sum = blunders_low + blunders_medium + blunders_high
  # blunders_low, blunders_medium, blunders_high = blunders_low / blunders_sum, blunders_medium / blunders_sum, blunders_high / blunders_sum

  density = gaussian_kde(df['loss'])
  low_density = gaussian_kde(low_data)
  medium_density = gaussian_kde(medium_data)
  high_density = gaussian_kde(high_data)
  x = np.linspace(-2, 15, 100)
  # x = np.linspace(df['loss'].min(), df['loss'].max(), 1000)
  y = density(x)
  low_y = low_density(x)
  medium_y = medium_density(x)
  high_y = high_density(x)
  # gains = dict(low: gains_low, medium: gains_medium, high: gains_high)
  # blunders = dict(low: blunders_low, medium: blunders_medium, high: blunders_high)
  return x, y, low_y, medium_y, high_y, gains, blunders

def plot(x, y, low_y, medium_y, high_y, gains, blunders):
  low_color = '#FFB6C1'  # pink
  medium_color = '#77DD77'  # green
  high_color = '#AEC6CF'  # blue
  # low_color = '#FFE4E1'  # rose
  # medium_color = '#FFDAB9'  # peach
  # high_color = '#F08080'  # coral
  # low_color = '#008080'  # teal
  # medium_color = '#4169E1'  # royal blue
  # high_color = '#1E90FF'  # dodger blue
  overall_color = '#707070'  # grey

  scale = max(high_y)  # highest point in the chart
  plt.plot(x, y, color=overall_color, linewidth=2, linestyle='--', label='Everyone')
  plt.plot(x, high_y, color=high_color, label=f'High Rating (>{config["high_rating"]})')
  plt.plot(x, medium_y, color=medium_color, label='Medium Rating')
  plt.plot(x, low_y, color=low_color, label=f'Low Rating (<{config["low_rating"]})')
  plt.fill_between(x, 0, high_y, color=high_color, alpha=0.3)
  plt.fill_between(x, 0, medium_y, color=medium_color, alpha=0.3)
  plt.fill_between(x, 0, low_y, color=low_color, alpha=0.3)

  # Adding bars for surprising gains and large blunders
  # gains_heights = [v * scale for v in gains.values()]
  # blunders_heights = [v * scale for v in blunders.values()]
  # gains_bottom = [0, gains['low'] * scale, (gains['low']+gains['medium']) * scale]
  # blunders_bottom = [0, blunders['low'] * scale, (blunders['low']+blunders['medium']) * scale]
  # plt.bar(-3, gains_heights, bottom=gains_bottom, width=1, color=[low_color, medium_color, high_color], alpha=0.7, align='center')
  # plt.bar(18, blunders_heights, bottom=blunders_bottom, width=1, color=[low_color, medium_color, high_color], alpha=0.7, align='center')
  plt.bar(-4.5, gains['low'], width=0.4, color=low_color, alpha=1, align='center')
  plt.bar(-4, gains['medium'], width=0.4, color=medium_color, alpha=1, align='center')
  plt.bar(-3.5, gains['high'], width=0.4, color=high_color, alpha=1, align='center')
  plt.bar(16.5, blunders['low'], width=0.4, color=low_color, alpha=1, align='center')
  plt.bar(17, blunders['medium'], width=0.4, color=medium_color, alpha=1, align='center')
  plt.bar(17.5, blunders['high'], width=0.4, color=high_color, alpha=1, align='center')

  plt.xlabel('Points Lost')
  plt.ylabel('Density')
  plt.title('Distribution of Points Loss')
  plt.legend()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  x, y, low_y, medium_y, high_y, gains, blunders = read_csv(path)
  print('Preparing plot...')
  plot(x, y, low_y, medium_y, high_y, gains, blunders)
