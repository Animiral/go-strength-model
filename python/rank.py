# conversion between ratings and ranks
import numpy as np

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
    return f'{kyu:.1f}-kyu'
  else:
    dan = min(rank - 29, 9);
    return f'{dan:.1f}-dan'
