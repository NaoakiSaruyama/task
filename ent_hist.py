from cProfile import label
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import argrelmax
from mz import x
from matplotlib import cm

root_path = "C:/Users/Saruyama/Downloads/roop2/hist2/"

#グラフ作成
def plt_fig(x,y,xlabel,ylabel,sidelabe,img_name):
    plt.subplots(1,1 , figsize=(8,5))
    t = np.linspace(0,1,401)
    plt.scatter(x,y,c = t , cmap=cm.jet , marker=".",lw=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.colorbar()
    ax.set_label(sidelabe)
    plt.savefig(root_path + img_name)

#全磁化
mz_all_img = []

#全エントロピー
entropy_all_img = []

for i in range(0,401):
  #画像読み込み
  img =  cv2.imread("C:/Users/Saruyama/Downloads/roop2/background/img" + str(i) + ".jpg"  ,cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
  w,h = img.shape

  #正規化
  sigma3 = np.round(3*np.std(img))
  var_mean = np.mean(img)
  img_rectmax = np.where(abs(img) >= var_mean + sigma3 , var_mean,img)
  img_round = np.where(abs(img_rectmax) <= var_mean - sigma3 , var_mean,img_rectmax)

  img_array = np.array(img_round)
  img_processed = img_array - np.mean(img_array)

  mini = np.max(img_processed)
  maxi = np.min(img_processed)
  img_mz = (img_processed-mini)/(maxi-mini)*(1-(-1)) + (-1)

  #一次元配列化
  array = np.ravel(img_mz)

  #ヒストグラムから値抽出
  histo , bin  = np.histogram(array , range=(-1,1) , bins=300)

  bin_x = bin[1:]

  fig  , ax = plt.subplots(dpi = 130)
  ax.plot(bin_x , histo , "o")
  ax.plot(bin_x , histo , "-")
  ax.set(xlabel = "bin" , ylabel = "Frequency")

  #258~280 , 308~310
  if 257 < i < 281 or 307 < i < 311:
    for n in range(1 , 20):
      arg_r_max = argrelmax(histo, order= n)
      max = arg_r_max[0]
      if len(max) < 3:
        break
  else:
    arg_r_max = argrelmax(histo, order= 20)
    max = arg_r_max[0]

  ax.plot(bin_x[max] , histo[max] ,"mo" , label="argrelmax")
  plt.savefig(root_path + str(i) + "max.png")

  #磁化の算出
  mz = sum(bin_x[max])
  mz_all_img.append(mz)

  max_abs  = abs(bin_x[max])

  entropy = 0

  #エントロピーの算出
  for i in range(0,len(bin_x[max])):
    entropy += -max_abs[i] * np.log(max_abs[i])
  entropy_all_img.append(entropy)

#磁化
plt_fig(x,mz_all_img , "Tempruture[℃]" , "Magnetization[$Wb/m^2$]" , "time[sec]" , "mz.png")

#エントロピー
plt_fig(x,entropy_all_img , "Tempruture[℃]" , "Entropy[J/K]" , "time[sec]" , "entropy.png")
