"""
Given an input CSV file from a rating system evaluation,
determine the last rating predictions for every player ("final rating") and plot their distribution.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def to_rank(rating):
  # 654  = 25k, rank(654) = 5
  # 962  = 16k, rank(962) = 14
  # 1005 = 15k, rank(1005) = 15
  # 1246 = 10k, rank(1246) = 20
  # 1919 = 1d,  rank(1919) = 30
  return np.log(rating / 525) * 23.15  # from https://forums.online-go.com/t/2021-rating-and-rank-adjustments/33389

def to_rating(rank):
  return np.exp(rank / 23.15) * 525

def rankstr(rank):
  # 654==25.0k, 30==1.0d
  if rank < 30:
    kyu = min(30 - rank, 30);
    return f'{kyu}-kyu'
  else:
    dan = min(rank - 29, 9);
    return f'{dan}-dan'

def read_csv(path):
  df = pd.read_csv(path, usecols=['Player White', 'Player Black', 'WhiteRating', 'BlackRating'])
  df_white = df[['Player White', 'WhiteRating']].rename(columns={'Player White': 'Player', 'WhiteRating': 'Rating'})
  df_black = df[['Player Black', 'BlackRating']].rename(columns={'Player Black': 'Player', 'BlackRating': 'Rating'})
  df = pd.concat([df_white, df_black])
  df = df.drop_duplicates(subset=['Player'], keep='last')
  df.sort_values('Rating', inplace=True)

  # estimate distribution
  density = gaussian_kde(df['Rating'])
  x = np.linspace(min(df['Rating']), max(df['Rating']), 100)
  y = density(x)

  return x, y

def plot(x, y):
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

  # scale = max(high_y)  # highest point in the chart
  plt.plot(x, y, color=overall_color, linewidth=2, linestyle='--')
  plt.xlabel('Rating')
  plt.ylabel('Density')

  # plt.plot(x, high_y, color=high_color, label=f'High Rating (>{config["high_rating"]})')
  # plt.plot(x, medium_y, color=medium_color, label='Medium Rating')
  # plt.plot(x, low_y, color=low_color, label=f'Low Rating (<{config["low_rating"]})')
  ddk = to_rank(x) <= 20
  sdk = (to_rank(x) > 20) & (to_rank(x) < 30)
  dan = to_rank(x) >= 30
  plt.fill_between(x[ddk], 0, y[ddk], color=low_color, alpha=0.3, label='10–kyu-25–kyu')
  plt.fill_between(x[sdk], 0, y[sdk], color=medium_color, alpha=0.3, label='9–kyu-1–kyu')
  plt.fill_between(x[dan], 0, y[dan], color=high_color, alpha=0.3, label='1–dan-9–dan')

  # plt.bar(-4.5, gains['low'], width=0.4, color=low_color, alpha=1, align='center')
  # plt.bar(-4, gains['medium'], width=0.4, color=medium_color, alpha=1, align='center')
  # plt.bar(-3.5, gains['high'], width=0.4, color=high_color, alpha=1, align='center')
  # plt.bar(16.5, blunders['low'], width=0.4, color=low_color, alpha=1, align='center')
  # plt.bar(17, blunders['medium'], width=0.4, color=medium_color, alpha=1, align='center')
  # plt.bar(17.5, blunders['high'], width=0.4, color=high_color, alpha=1, align='center')
  if x[ddk].size > 0:
    plt.text((x[ddk][-1]+x[ddk][0]) / 2, 0.00005, "DDK", horizontalalignment='center', size=20, color=low_color)
  if x[sdk].size > 0:
    plt.text((x[sdk][-1]+x[sdk][0]) / 2, 0.00005, "SDK", horizontalalignment='center', size=20, color=medium_color)
  if x[dan].size > 0:
    plt.text((x[dan][-1]+x[dan][0]) / 2, 0.00005, "DAN", horizontalalignment='center', size=20, color=high_color)

  # 2nd x-axis tick set for ranks
  axes = plt.gca()
  rank_axis = axes.twiny()
  # print(f"Set xlim to {axes.get_xlim()} -> {(rank(axes.get_xlim()[0]), rank(axes.get_xlim()[1]))}")
  # rank_ticks = rank(np.linspace(start=min(x), stop=max(x), num=10))
  rank_ticks = np.array([0, 10, 15, 20, 25, 30, 34, 39])
  rank_axis.set_xticks(to_rating(rank_ticks))
  rank_axis.set_xlim(axes.get_xlim()[0], axes.get_xlim()[1])
  rank_axis.set_xticklabels(rankstr(r) for r in rank_ticks)
  rank_axis.set_label('Rank')

  plt.title('Distribution of Final Glicko-2 Rating')
  # plt.legend()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  x, y = read_csv(path)
  plot(x, y)
