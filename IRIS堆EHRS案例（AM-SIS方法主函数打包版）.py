import os
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import truncexpon
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示中文标签
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
from joblib import Parallel, delayed
from shutil import copy

def alter(filepath, search_str, old_str, new_str):
  """
  替换文件中的字符串
  filepath:文件名
  search_str:搜索位置
  old_str:旧字符串
  new_str:新字符串
  """
  file_data = ''
  with open(filepath, 'r', encoding="gbk") as f:
      for line in f:
          if search_str in line:
              line = line.replace(old_str, new_str)
          file_data += line
  with open(filepath, 'w', encoding="gbk") as f:
      f.write(file_data)

def rewrite_line(filepath, search_str, new_line):
  """
  重写文件中的某一行
  filepath:文件名
  search_str:搜索位置
  new_line:新的内容
  """
  file_data = ''
  with open(filepath, 'r', encoding="gbk") as f:
      for line in f:
          if search_str in line:
              line = new_line
          file_data += line
  with open(filepath, 'w', encoding="gbk") as f:
      f.write(file_data)

def run_relap(filepath, i_name):
  """
  运行RELAP5
  filepath:relap5所在位置
  i_name:i文件名称
  """
  o_name = i_name.replace('.i','.o')
  r_name = i_name.replace('.i','.r')
  os.chdir(filepath)
  parameter = "%s\\relap5.exe -i %s -o %s -r %s"%(filepath, i_name, o_name, r_name)

  if os.path.exists(o_name):
      # print("%s is exist"%(o_name))
      os.remove(o_name)
      os.remove(r_name)
                  
  # os.popen(parameter)
  os.system(parameter)

def restart_relap(filepath, i_name, r_name):
  """
  运行RELAP5
  filepath:relap5所在位置
  i_name:i文件名称
  """
  o_name = i_name.replace('.i','.o')
  os.chdir(filepath)
  parameter = "%s\\relap5.exe -i %s -o %s -r %s"%(filepath, i_name, o_name, r_name)

  if os.path.exists(o_name):
      # print("%s is exist"%(o_name))
      os.remove(o_name)
                  
  os.system(parameter)

def mkstripf(filepath, r_name):
  """
  生成stripf文件
  filepath:relap5所在位置
  r_name:r文件名称
  """
  i_name = "strip-1.i"
  o_name = "strip-1.o"
  os.chdir(filepath)
  parameter = "%s\\relap5.exe -i %s -o %s -r %s"%(filepath,i_name,o_name,r_name)

  if os.path.exists(o_name):
      # print("strip-1.o is exist")
      os.remove(o_name)
      os.remove('stripf')
  
  # os.popen(parameter)
  os.system(parameter)
  # time.sleep(0.5)

def mkdir(filepath):
  """ 创建给定路径的文件夹 """
  try:
      os.mkdir(filepath)
      # print(filepath+" has been successfully created!")
  except:
      if os.path.exists(filepath):
          print(filepath+" already exist!")
      else:
          print(filepath+" error!")

def copy_irelap5(from_path, to_path, i_name, r_name):
  """
  备份.i relap5相关文件
  from_path:文件原位置
  to_path:文件要备份到的位置
  i_name:i文件名称
  """
  i_f_path = from_path + '\\' + i_name
#   r_f_path = from_path + '\\' + r_name
  f_path_1 = from_path + '\\relap5.exe'
  f_path_2 = from_path + '\\glut32.dll'
  f_path_3 = from_path + '\\strip-1.i'
  f_path_4 = from_path + '\\tpfh2o'
  # i_t_path = to_path + '\\' + i_name
  # 备份位置删除已存在的文件
  # if os.path.exists(i_t_path):
      # print("i o stripf_file is exist")
      # os.remove(i_t_path)
  #     os.remove(o_t_path)
  #     os.remove(stripf_t_path)
  copy(i_f_path, to_path)
#   copy(r_f_path, to_path)
  copy(f_path_1, to_path)
  copy(f_path_2, to_path)
  copy(f_path_3, to_path)
  copy(f_path_4, to_path)


def stripf2csv(FromFile,TargetFile):
  """ 从FromFile中读取stripf文件,将读取的文件以csv格式输出到TargetFile中 """
  # 本脚本用于从relap5mod3.5中stripf文件中提取数据

  stripf_file = FromFile
  with open(stripf_file,"r") as f:
      temp = f.readlines()
  #print(type(temp[1]))
  #print(temp[3])

  #读取变量列表
  strip_names = temp[3].replace(" plotalf-plotnum","")
  #print(strip_names.split())
  strip_names = strip_names.split()

  #读取变量数据
  a = []
  for data in temp[4:]:
      data = data.replace("plotrec","")
      a.append(data.split())
  strip_datas = np.array(a,dtype=float)

  # print(strip_datas.shape)
  # print(len(temp[4:]),len(strip_names))
  # strip_datas = np.transpose(strip_datas)

  #判断数据是否为空，如果是则添加无用数据0
  if len(temp[4:]) == 0:
      print('Input data missing!')
      strip_datas = np.zeros((1, len(strip_names)))

  #保存数组到csv
  pd.DataFrame(strip_datas).to_csv(TargetFile,header=strip_names,index=None)
  return strip_names,strip_datas

def csv_plot(FromFile,TargetFile):
  """ 本函数用于从csv中画图 """
  df = pd.read_csv(FromFile,sep=",")
  names = df.columns.values
  time = df.iloc[:,0].values
  for i in range(1,len(names)):
      temp = df.iloc[:,i].values
      plt.figure(figsize=(16,10),dpi=200)
      plt.plot(time,temp,linewidth=1.2,color="black")
      plt.xlabel("time")
      plt.ylabel(names[i])
      plt.savefig(TargetFile+"\\"+names[i]+".jpg")
      plt.close()
  return df,time

def relap_calculate(copy_dir, i_name, i):
  # 运行relap5
  # OutDir = copy_dir + '\\No.%d'%(i)
  OutDir = copy_dir + '\\EHRS0718No.%d'%(i)
  # o_path = OutDir + '\\' + i_name.replace('i','o')
  # r_path = OutDir + '\\' + i_name.replace('i','r')
  # print('No.%d calculating'%(i))
  run_relap(OutDir, i_name)
  # print('No.%d calculation completed'%(i))
  # return(i)

def mkstripf_calculate(copy_dir, r_name, i):
  # 运行relap5
  # OutDir = copy_dir + '\\No.%d'%(i)
  copy_to_path = copy_dir + '\\EHRS0718No.%d'%(i)
  mkstripf(copy_to_path, r_name)
  # print('No.%d calculation completed'%(i))
  # return(i)

def calculateMSE(X,Y):
    return sum([(y-x)**2 for x,y in zip(X,Y)])/len(X)

def RELAP5(u, i_name, r_name, OutDir, copy_dir):
    # 路径（不用修改）
    i_path = OutDir + '\\' + i_name  # i文件路径
    o_path = OutDir + '\\' + i_name.replace('.i','.o')  # o文件路径
    r_path = OutDir + '\\' + r_name  # r文件路径
    search_str = 'Transient terminated by failure'  # 计算失败关键词
    csv_path = OutDir + '\\' + '图表'  # 保存csv文件的路径（需要自己创建）

    N = len(u)
    
    # 二回路流量修改
    secondary_flow = u[:, 0] * 126.1 / 1E9

    # 衰变热修改
    # u[:, 1] = truncnorm.ppf(random_seed[:, 1], a = -2, b = 2, loc = 8207426, scale = 270845)
    # u[:, 1] = np.ones((N, 1))*(truncnorm.ppf(1, a = -2, b = 2, loc = 8207426, scale = 270845))
    decayheat_19 = u[:, 1] * 13233700 / 8207426
    decayheat_18 = u[:, 1] * 18038850 / 8207426
    decayheat_17 = u[:, 1] * 22259480 / 8207426
    decayheat_16 = u[:, 1] * 26736960 / 8207426
    decayheat_15 = u[:, 1] * 30016270 / 8207426
    decayheat_14 = u[:, 1] * 50353200 / 8207426
    decayheat_13 = u[:, 1] * 59536130 / 8207426
    decayheat_12 = u[:, 1] * 82041500 / 8207426
    decayheat_11 = u[:, 1] * 115277300 / 8207426
    decayheat_10 = u[:, 1] * 121821900 / 8207426
    decayheat_9 = u[:, 1] * 129661400 / 8207426
    decayheat_8 = u[:, 1] * 139232200 / 8207426
    decayheat_7 = u[:, 1] * 151252400 / 8207426
    decayheat_6 = u[:, 1] * 167134000 / 8207426
    decayheat_5 = u[:, 1] * 217818900 / 8207426
    decayheat_4 = u[:, 1] * 848313700 / 8207426

    # 燃料导热系数修改
    # u[:, 2] = random_seed[:, 2] * 1.832432652 + 5.072125674
    # u[:, 2] = np.ones((N, 1))*(1 * 1.832432652 + 5.072125674)
    numbda_17 = u[:, 2] * 5.323740 / 5.988342
    numbda_16 = u[:, 2] * 4.866826 / 5.988342
    numbda_15 = u[:, 2] * 4.614138 / 5.988342
    numbda_14 = u[:, 2] * 4.579524 / 5.988342
    numbda_13 = u[:, 2] * 4.783750 / 5.988342
    numbda_12 = u[:, 2] * 4.897980 / 5.988342
    numbda_11 = u[:, 2] * 5.043360 / 5.988342
    numbda_10 = u[:, 2] * 5.673348 / 5.988342
    numbda_9 = u[:, 2] * 5.673348 / 5.988342
    numbda_8 = u[:, 2] * 5.967574 / 5.988342
    numbda_7 = u[:, 2] * 6.310258 / 5.988342
    numbda_6 = u[:, 2] * 6.715250 / 5.988342
    numbda_5 = u[:, 2] * 7.760614 / 5.988342
    numbda_4 = u[:, 2] * 9.266354 / 5.988342
    numbda_3 = u[:, 2] * 11.56477 / 5.988342
    numbda_2 = u[:, 2] * 12.92000 / 5.988342
    numbda_1 = u[:, 2] * 16.88000 / 5.988342
    
    # 复制所需的文件到指定文件夹
    for i in range(N):
        from_path = OutDir
        copy_to_path = copy_dir + '\\EHRS0718No.%d'%(i)
        mkdir(copy_to_path)
        copy_irelap5(from_path, copy_to_path, i_name, r_name)

    
    for i in range(N):
    # 修改输入卡
        i_path = copy_dir + '\\EHRS0718No.%d'%(i) + '\\' + i_name   #!!! (改)
        # i_path = copy_dir + '\\testNo.%d'%(i) + '\\' + i_name

        # rewrite_line(i_path, '9080202', ' 9080202   10.0  %.2f 0.0  0.0'%(mflowj_908[i]) + '\n')

        rewrite_line(i_path, '20290501', ' 20290501    0.0        %.6e'%(u[i, 0]) + '\n')
        alter(i_path, ' 126.1 ', ' 126.1 ', ' %.3f '%(secondary_flow[i]))

        rewrite_line(i_path, '20290504', ' 20290504    1.5        %.6e'%(decayheat_4[i]) + '\n')
        rewrite_line(i_path, '20290505', ' 20290505    2.5        %.6e'%(decayheat_5[i]) + '\n')
        rewrite_line(i_path, '20290506', ' 20290506    3.5        %.6e'%(decayheat_6[i]) + '\n')
        rewrite_line(i_path, '20290507', ' 20290507    4.5        %.6e'%(decayheat_7[i]) + '\n')
        rewrite_line(i_path, '20290508', ' 20290508    5.5        %.6e'%(decayheat_8[i]) + '\n')
        rewrite_line(i_path, '20290509', ' 20290509    6.5        %.6e'%(decayheat_9[i]) + '\n')
        rewrite_line(i_path, '20290510', ' 20290510    7.5        %.6e'%(decayheat_10[i]) + '\n')
        rewrite_line(i_path, '20290511', ' 20290511    8.5        %.6e'%(decayheat_11[i]) + '\n')
        rewrite_line(i_path, '20290512', ' 20290512    18.5       %.6e'%(decayheat_12[i]) + '\n')
        rewrite_line(i_path, '20290513', ' 20290513    38.5       %.6e'%(decayheat_13[i]) + '\n')
        rewrite_line(i_path, '20290514', ' 20290514    55.5       %.6e'%(decayheat_14[i]) + '\n')
        rewrite_line(i_path, '20290515', ' 20290515    226.5      %.6e'%(decayheat_15[i]) + '\n')
        rewrite_line(i_path, '20290516', ' 20290516    385.5      %.6e'%(decayheat_16[i]) + '\n')
        rewrite_line(i_path, '20290517', ' 20290517    885.5      %.6e'%(decayheat_17[i]) + '\n')
        rewrite_line(i_path, '20290518', ' 20290518    1885.5     %.6e'%(decayheat_18[i]) + '\n')
        rewrite_line(i_path, '20290519', ' 20290519    4885.5     %.6e'%(decayheat_19[i]) + '\n')
        rewrite_line(i_path, '20290520', ' 20290520    24885.5    %.6e'%(u[i, 1]) + '\n')

        rewrite_line(i_path, '20100301', '20100301       273.15      %.6f'%(numbda_1[i]) + '\n')
        rewrite_line(i_path, '20100302', '20100302       416.67      %.6f'%(numbda_2[i]) + '\n')
        rewrite_line(i_path, '20100303', '20100303       533.15      %.6f'%(numbda_3[i]) + '\n')
        rewrite_line(i_path, '20100304', '20100304     699.8167      %.6f'%(numbda_4[i]) + '\n')
        rewrite_line(i_path, '20100305', '20100305     866.4833      %.6f'%(numbda_5[i]) + '\n')
        rewrite_line(i_path, '20100306', '20100306     1033.150      %.6f'%(numbda_6[i]) + '\n')
        rewrite_line(i_path, '20100307', '20100307     1088.706      %.6f'%(numbda_7[i]) + '\n')
        rewrite_line(i_path, '20100308', '20100308     1199.817      %.6f'%(numbda_8[i]) + '\n')
        rewrite_line(i_path, '20100309', '20100309     1283.150      %.6f'%(numbda_9[i]) + '\n')
        rewrite_line(i_path, '20100310', '20100310     1366.483      %.6f'%(numbda_10[i]) + '\n')
        rewrite_line(i_path, '20100311', '20100311     1533.150      %.6f'%(numbda_11[i]) + '\n')
        rewrite_line(i_path, '20100312', '20100312     1616.483      %.6f'%(numbda_12[i]) + '\n')
        rewrite_line(i_path, '20100313', '20100313     1699.817      %.6f'%(numbda_13[i]) + '\n')
        rewrite_line(i_path, '20100314', '20100314     1977.594      %.6f'%(numbda_14[i]) + '\n')
        rewrite_line(i_path, '20100315', '20100315     2255.372      %.6f'%(numbda_15[i]) + '\n')
        rewrite_line(i_path, '20100316', '20100316     2533.150      %.6f'%(numbda_16[i]) + '\n')
        rewrite_line(i_path, '20100317', '20100317     2810.928      %.6f'%(numbda_17[i]) + '\n')
        rewrite_line(i_path, '20100318', '20100318     3088.706      %.6f'%(u[i, 2]) + '\n')

        alter(i_path, ' 322.01 ', ' 322.01 ', ' %.2f '%(u[i, 3]))

        rewrite_line(i_path, '17000801', ' 17000801  0.0 10.0 10.0 0.0  0.0  0.0  0.0 1.0 0.0 1.1 %.3f 10'%(u[i, 4]) + '\n')
        rewrite_line(i_path, '17000901', ' 17000901  0.0 10.0 10.0 0.0  0.0  0.0  0.0 1.0 0.0 1.1 %.3f 10'%(u[i, 4]) + '\n')
        rewrite_line(i_path, '17100801', ' 17100801  0.0 10.0 10.0 0.0  0.0  0.0  0.0 1.0 0.0 1.1 %.3f 10'%(u[i, 4]) + '\n')
        rewrite_line(i_path, '17100901', ' 17100901  0.0 10.0 10.0 0.0  0.0  0.0  0.0 1.0 0.0 1.1 %.3f 10'%(u[i, 4]) + '\n')
        rewrite_line(i_path, '17200801', ' 17200801  0.0 10.0 10.0 0.0  0.0  0.0  0.0 1.0 0.0 1.1 %.3f 10'%(u[i, 4]) + '\n')
        rewrite_line(i_path, '17200901', ' 17200901  0.0 10.0 10.0 0.0  0.0  0.0  0.0 1.0 0.0 1.1 %.3f 10'%(u[i, 4]) + '\n')

        alter(i_path, ' 15.512e6 ', ' 15.512e6 ', ' %.6e '%(u[i, 5]))

        rewrite_line(i_path, '1430101', ' 1430101  142020002 145010001 0.0081 1.0 1.0 0100  %.3f  %.3f  %.3f'%(u[i, 6], u[i, 6], u[i, 6]) + '\n')
        rewrite_line(i_path, '1530101', ' 1530101  152020002 155010001 0.0081 1.0 1.0 0100  %.3f  %.3f  %.3f'%(u[i, 6], u[i, 6], u[i, 6]) + '\n')
        rewrite_line(i_path, '1630101', ' 1630101  162020002 165010001 0.0081 1.0 1.0 0100  %.3f  %.3f  %.3f'%(u[i, 6], u[i, 6], u[i, 6]) + '\n')

        rewrite_line(i_path, '1420801', ' 1420801   %.3e     0.0     3'%(u[i, 7]) + '\n')
        rewrite_line(i_path, '1520801', ' 1520801   %.3e     0.0     3'%(u[i, 7]) + '\n')
        rewrite_line(i_path, '1620801', ' 1620801   %.3e     0.0     3'%(u[i, 7]) + '\n')

        print(i_path + ' modification completed')

    # 并行计算
    numbers = list(range(N))
    Parallel(n_jobs = 8)(delayed(relap_calculate)(copy_dir, i_name, i) for i in numbers)
    
    # stripf提取数据
    numbers = list(range(N))
    Parallel(n_jobs = 8)(delayed(mkstripf_calculate)(copy_dir, r_name, i) for i in numbers)
    
    failure_case = []
    for i in range(N):
        o_path = copy_dir + '\\EHRS0718No.%d'%(i) + '\\' + i_name.replace('.i','.o')  
    with open(o_path, 'r', encoding="gbk") as f:
        for line in f:
            if search_str in line:
                failure_case.append(i)
                print('No.%d '%(i) + search_str)

    T_max = np.zeros([N, 1])
    coreLEVEL_min = np.zeros([N, 1])
    totalLEVEL_min = np.zeros([N, 1])
    count = np.zeros([N, 1])
    for i in range(N):
        # 提取stripf数据到csv文件中
        copy_to_path = copy_dir + '\\EHRS0718No.%d'%(i)
        from_path = copy_to_path + '\\stripf'
        target_path = csv_path + '\\EHRS0718No.%d.csv'%(i)
        stripf2csv(from_path, target_path)
        # print(i)
        # 提取堆芯最低液位
        df = pd.read_csv(target_path)
        data_100 = df.iloc[10:, :].values
        T_data = data_100[:, 1]
        T_max[i] = max(T_data)
        coreLEVEL_min[i] = min(data_100[:, 2])
        totalLEVEL_min[i] = min(data_100[:, 3])
        count[i] = i
    
    # 处理数据
    # 失效案例的值替换为111
    for i in range(len(failure_case)):
        totalLEVEL_min[failure_case[i]] = 111
    
    return(totalLEVEL_min)

def latin_hypercube(n, d):
    """
    生成拉丁超立方抽样点集合

    Args:
        n: 抽样点数量
        d: 维度

    Returns:
        返回一个形状为 (n, d) 的 NumPy 数组，其中包含 n 个 d 维均匀分布的抽样点
    """
    # 生成均匀分布的样本点
    x = np.zeros((n, d))
    # for i in range(d):
    #     x[:, i] = np.random.uniform(size=n)
    # 对每一列进行排列
    for i in range(d):
        order = np.random.permutation(range(n))
        x[:, i] = (order + np.random.uniform(size=n)) / n
    return x

# 神经网络模型（包含归一化）
def ANN(u, model, x, y):
    """
    使用神经网络模型预测输出。
    
    参数：
    u : numpy.ndarray
        输入参数的数组。
    model : keras.Model
        预训练的神经网络模型。
    x : numpy.ndarray
        训练数据的输入特征。
    y : numpy.ndarray
        训练数据的输出特征。

    返回：
    numpy.ndarray
        经过模型预测并反归一化后的输出结果。
    """
    # 对输入参数进行归一化处理
    x1_ANN = (2 * u[:, 0] - max(x[:, 0]) - min(x[:, 0])) / (max(x[:, 0]) - min(x[:, 0]))
    x2_ANN = (2 * u[:, 1] - max(x[:, 1]) - min(x[:, 1])) / (max(x[:, 1]) - min(x[:, 1]))
    x3_ANN = (2 * u[:, 2] - max(x[:, 2]) - min(x[:, 2])) / (max(x[:, 2]) - min(x[:, 2]))
    x4_ANN = (2 * u[:, 3] - max(x[:, 3]) - min(x[:, 3])) / (max(x[:, 3]) - min(x[:, 3]))
    x5_ANN = (2 * u[:, 4] - max(x[:, 4]) - min(x[:, 4])) / (max(x[:, 4]) - min(x[:, 4]))
    x6_ANN = (2 * u[:, 5] - max(x[:, 5]) - min(x[:, 5])) / (max(x[:, 5]) - min(x[:, 5]))
    x7_ANN = (2 * u[:, 6] - max(x[:, 6]) - min(x[:, 6])) / (max(x[:, 6]) - min(x[:, 6]))
    x8_ANN = (2 * u[:, 7] - max(x[:, 7]) - min(x[:, 7])) / (max(x[:, 7]) - min(x[:, 7]))

    # 将归一化后的特征组合在一起
    x_ANN = np.hstack((x1_ANN.astype('float32').reshape(-1,1), x2_ANN.astype('float32').reshape(-1,1), x3_ANN.astype('float32').reshape(-1,1), x4_ANN.astype('float32').reshape(-1,1), x5_ANN.astype('float32').reshape(-1,1), x6_ANN.astype('float32').reshape(-1,1), x7_ANN.astype('float32').reshape(-1,1), x8_ANN.astype('float32').reshape(-1,1)))
    
    # 使用神经网络模型预测输出
    y_ANN = model.predict(x_ANN)

    # 对预测结果进行反归一化处理
    y_output = (y_ANN * (max(y) - min(y)) + max(y) + min(y)) / 2

    # 返回反归一化后的预测结果
    return(y_output)

# 二分法找零点，计算条件失效概率
def binary_search(f, a, b, epsilon):
    while abs(b - a) > epsilon:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def ANN_train(path_DOE, path_ANN, path_extrmedata, hidden_layer_node, d, failure_criteria):
    """
    ---------------------------------------------------------------------------
    训练神经网络模型
    参数：
    path_DOE: str - 包含设计实验(DOE)数据的CSV文件路径。
    path_ANN: str - 用于保存训练后模型的路径。
    path_extrmedata: str - 包含极值配置的CSV文件路径。
    hidden_layer_node: int - 隐藏层节点数。
    d: int - 输入维度（特征数）。
    failure_criteria: float - 失效准则，用于转换输出值，使得y<0为失效区域
    
    返回：
    model2: keras.Model - 训练后的最佳神经网络模型。
    x: numpy.ndarray - 输入特征数据（未归一化）。
    y: numpy.ndarray - 输出特征数据（未归一化）。
    ---------------------------------------------------------------------------
    """
    # 输入参数及对应的输出值
    filepath = path_DOE
    df = pd.read_csv(filepath)
    data_total = df.values

    # 读取极值配置并将其与DOE数据合并
    filepath = path_extrmedata
    df_extrme = pd.read_csv(filepath)
    data_extrme = df_extrme.values
    data = np.vstack((data_extrme, data_total))

    # 数据集大小
    total_size = len(data_total)
    train_size = round(total_size * 0.8) + len(data_extrme)
    test_size = round(total_size * 0.2)
    N = np.arange(total_size)
    # np.random.shuffle(N) # 打乱顺序

    x1 = data[N,0].astype('float32').reshape(-1,1)
    x2 = data[N,1].astype('float32').reshape(-1,1)
    x3 = data[N,2].astype('float32').reshape(-1,1)
    x4 = data[N,3].astype('float32').reshape(-1,1)
    x5 = data[N,4].astype('float32').reshape(-1,1)
    x6 = data[N,5].astype('float32').reshape(-1,1)
    x7 = data[N,6].astype('float32').reshape(-1,1)
    x8 = data[N,7].astype('float32').reshape(-1,1)
    # 根据输出值的特点改变y的表达式（如包壳最高温度是失效准则在左边，最低液位是失效准则在右边）
    y = data[N,8].astype('float32').reshape(-1,1) - failure_criteria # 使得0作为失效区域与成功区域的分界线
    x = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8))

    # 归一化到(-1,1)
    x1_total = (2 * x1 - max(x1) - min(x1)) / (max(x1) - min(x1))
    x2_total = (2 * x2 - max(x2) - min(x2)) / (max(x2) - min(x2))
    x3_total = (2 * x3 - max(x3) - min(x3)) / (max(x3) - min(x3))
    x4_total = (2 * x4 - max(x4) - min(x4)) / (max(x4) - min(x4))
    x5_total = (2 * x5 - max(x5) - min(x5)) / (max(x5) - min(x5))
    x6_total = (2 * x6 - max(x6) - min(x6)) / (max(x6) - min(x6))
    x7_total = (2 * x7 - max(x7) - min(x7)) / (max(x7) - min(x7))
    x8_total = (2 * x8 - max(x8) - min(x8)) / (max(x8) - min(x8))
    y_total = (2 * y - max(y) - min(y)) / (max(y) - min(y))
    x_total = np.hstack((x1_total, x2_total, x3_total, x4_total, x5_total, x6_total, x7_total, x8_total))

    # 划分为训练集，验证集
    x_train = x_total[0:train_size,:]
    x_test = x_total[train_size:train_size+test_size,:]
 
    y_train = y_total[0:train_size,:]
    y_test = y_total[train_size:train_size+test_size,:]

    # 搭建神经网络模型
    model1 = Sequential()
    model1.add(Dense(units=hidden_layer_node, input_dim=d, activation="tanh"))
    model1.add(Dense(units=1, activation="linear"))

    class stopAtLossValue(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            THR = 0.0000001
            if logs.get('loss') < THR:
                print("\n !!! enough small loss, no further training !!!")
                self.model.stop_training = True
    end_train = stopAtLossValue()  

    path = path_ANN
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_loss', save_best_only=True)
    opm = tf.keras.optimizers.Adam(learning_rate=0.01)
    model1.compile(optimizer=opm,loss='mse')  
    # model1.summary()

    # 训练模型（batch_size默认为32）
    history = model1.fit(x_train, y_train, batch_size = 16, validation_data = (x_test, y_test),
                        epochs = 1000, callbacks = [end_train, checkpoint])
    
    # 提取最佳模型
    model2 = keras.models.load_model(path)
    return model2, x, y

def SS(ANN, model, x, y, d):
    """
    ---------------------------------------------------------------------------
    使用神经网络运行子集模拟: 简单自然循环回路案例: y = 460 - RELAP5(u)
    参数：
    ANN: 归一化的神经网络模型
    model: str - 用于保存训练后模型的路径。
    x: numpy.ndarray - 输入特征数据（未归一化）。
    y: numpy.ndarray - 输出特征数据（未归一化）。
    d: int - 输入维度（特征数）。
    
    返回：
    res: 包含输入参数矩阵和对应预测输出值的排序结果数组（来源于子集模拟最后一层）
    ---------------------------------------------------------------------------
    """

    pi_pdf = list()
    # 8个输入参数的分布
    pi_pdf.append(ERADist('truncatednormal','PAR',[1E9, 1E7, 0.98E9, 1.02E9])) # 截断正态分布
    pi_pdf.append(ERADist('truncatednormal','PAR',[8207426, 270845, 7665736, 8749116])) # 截断正态分布
    pi_pdf.append(ERADist('uniform','PAR',[5.072125674, 6.904558326])) # 均匀分布
    pi_pdf.append(ERADist('truncatednormal','PAR',[322, 10.304, 301.392, 342.608])) # 截断正态分布
    pi_pdf.append(ERADist('truncatednormal','PAR',[1, 0.1965, 0.607, 1.393])) # 截断正态分布
    pi_pdf.append(ERADist('uniform','PAR',[1.520176E7, 1.582224E7])) # 均匀分布
    pi_pdf.append(ERADist('uniform','PAR',[0.84, 1.16])) # 均匀分布
    pi_pdf.append(ERADist('uniform','PAR',[5.04E-6, 7.56E-6])) # 均匀分布

    # correlation matrix
    R = np.eye(d)   # independent case

    # object with distribution information
    pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

    N  = 1000      # Total number of samples for each level
    p0 = 0.1         # Probability of each subset, chosen adaptively

    # Implementation of sensitivity analysis: 1 - perform, 0 - not perform
    sensitivity_analysis = 0

    # Samples return: 0 - none, 1 - final sample, 2 - all samples
    samples_return = 1

    print('\n\nSUBSET SIMULATION: ')
    [Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, S_F1] = SuS(N, p0, ANN, pi_pdf, sensitivity_analysis, samples_return, model, x, y)

    # show p_f results
    # print('\n***Exact Pf: ', pf_ex, ' ***')
    print('***SuS Pf: ', Pf_SuS, ' ***\n')

    X = np.array(samplesX).ravel().reshape(-1,d)   # 提取最后一层的样本点

    X_ANN = pd.DataFrame(X).drop_duplicates().values   #去除重复行
    y_ANN = ANN(X_ANN, model, x, y)

    output = np.hstack((X_ANN, y_ANN))
    res = output[np.argsort(-output[:, d])]        #对输出值降序排列（带负号就是降序）
    # header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y_ANN']
    # TargetFile = r'C:\Users\JH\Desktop\IRIS数据集0718\IRIS_extra(ANN).csv'
    # pd.DataFrame(res).to_csv(TargetFile, header = header, index = None)
    return res


def total(sample_num, d, i_name, r_name, OutDir, copy_dir, path_relap5, path_initialDOE, failure_criteria, path_ANN, path_dataforiteration, path_extradata, path_totaldata, path_extrmedata):  
    """
    ---------------------------------------------------------------------------
    AM-SIS方法主体部分
    参数：
    sample_num: 最终用于计算条件失效概率的样本数量。
    d: 输入参数的维度。
    i_name: RELAP5输入文件名。
    r_name: RELAP5输出文件名。
    OutDir: 输出目录。
    copy_dir: 复制目录，用于存储中间文件。
    path_relap5: RELAP5的路径。
    path_initialDOE: 初始DOE数据集的路径。
    failure_criteria: 失效准则。
    path_ANN: 神经网络模型的路径。
    path_dataforiteration: 使用神经网络计算失效概率的同一数据集路径。
    path_extradata: 额外数据集的路径。
    path_totaldata: 总体数据集的路径。
    path_extrmedata: 极值配置数据的路径。
    
    返回：
    P_f: 最终失效概率
    ---------------------------------------------------------------------------
    """
    # 初始化参数
    res_i = np.zeros(10) # 记录针对同一数据集的失效概率结果，初始为0
    res_err = np.ones(10) # 记录失效概率的相对变化，初始为100%

    # 训练神经网络模型
    hidden_layer_node = 10  # 隐藏层初始节点数
    path_ANN0 = path_ANN + '\\' + 'keras_8_0.h5' # 保存初始最佳（验证集误差最小）的神经网络模型位置
    model, x, y = ANN_train(path_initialDOE, path_ANN0, path_extrmedata, hidden_layer_node, d, failure_criteria)

    # 计算第一次的失效概率
    filepath = path_dataforiteration
    df = pd.read_csv(filepath)
    x_tt = df.values

    y_ANN = ANN(x_tt, model, x, y)

    for j in range(len(x_tt)):
        if y_ANN[j] < 0:
            res_i[0] = res_i[0] + 1/len(x_tt)
    print('res_%d = %f'%(0, res_i[0]))    # 第一次计算的失效概率

    data_extra_ANN = SS(ANN, model, x, y, d)
    data = data_extra_ANN[:8,:]      # 额外数据集的大小（8个,可以改）

    X00 = data[:, 0:d]
    y00 = RELAP5(X00, i_name, r_name, path_relap5, copy_dir)        # 使用RELAP5计算真实输出值
    output = np.hstack((X00, y00.reshape(-1,1)))
    header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
    TargetFile = path_extradata + '1.csv'  # 额外的输入参数
    pd.DataFrame(output).to_csv(TargetFile, header = header, index = None)

    # ========================迭代开始=================================
    for k in range(10):
        filepath = path_extradata + '%d.csv'%(k+1) # 新增加的样本集地址
        df_extra = pd.read_csv(filepath)
        data_extra = df_extra.values
        total_size = len(data_extra)
        train_size = round(total_size * 0.8)
        test_size = round(total_size * 0.2)
        N = np.arange(total_size)
        np.random.shuffle(N)
        data_N = data_extra[N,:]
        data_train = data_N[0:train_size,:]
        data_test = data_N[train_size:train_size+test_size,:]

        filepath = path_totaldata + '%d.csv'%(k+1) # 之前总体样本集地址
        df = pd.read_csv(filepath)
        data = df.values
        data_total = np.vstack((data_train, data, data_test))

        header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y_true']
        path_DOE = path_totaldata + '%d.csv'%(k+2) # 合并成一个数据集
        pd.DataFrame(data_total).to_csv(path_DOE, header = header, index = None)

        path_ANN_k = path_ANN + '\\' + 'keras_8_%d.h5'%(k+1)
        hidden_layer_node = int((len(data_total)-1)/(d+2))  # 根据总体样本的大小确定隐藏层节点数
        model, x, y = ANN_train(path_DOE, path_ANN_k, path_extrmedata, hidden_layer_node, d, failure_criteria)

        filepath = path_dataforiteration
        df = pd.read_csv(filepath)
        x_tt = df.values

        y_ANN = ANN(x_tt, model, x, y)

        for j in range(len(x_tt)):
            if y_ANN[j] < 0:
                res_i[k+1] = res_i[k+1] + 1/len(x_tt)
        res_err[k] = (res_i[k+1] - res_i[k])/ res_i[k]
        print('res_i = %f'%(res_i[k+1]))          # 第k次迭代的失效概率
        print('res_err = %f'%(res_err[k]))        # 第k次迭代的失效概率相对变化

        if np.abs(res_err[k]) < 0.01:
            print('k = %d\n'%k)
            print('迭代结束')
            break
        else:
            """
            ---------------------------------------------------------------------------
            使用子集模拟生成失效边界附近的样本点
            ---------------------------------------------------------------------------
            """
            data_extra_ANN = SS(ANN, model, x, y, d)
            data = data_extra_ANN[:8,:]      # 额外数据集的大小（8个,可以改）

            X_extra = data[:, 0:d]
            y_extra = RELAP5(X_extra, i_name, r_name, path_relap5, copy_dir)        # 使用RELAP5计算真实输出值
            output = np.hstack((X_extra, y_extra.reshape(-1,1)))
            header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
            TargetFile = path_extradata + '%d.csv'%(k+2)  # 额外的输入参数
            pd.DataFrame(output).to_csv(TargetFile, header = header, index = None)
            print('k = %d'%k)

    # ================================迭代结束======================================
    # 使用最后一次迭代的样本集和神经网络模型进行最终的重要抽样
    X_m = x
    y_m = y
    y_ANN = ANN(X_m, model, x, y)

    output = np.hstack((X_m, y_m.reshape(-1,1), y_ANN.reshape(-1,1)))
    res = output[np.argsort(output[:, d])]        #对输出值升序排列
    header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y_m', 'y_ANN']
    TargetFile = OutDir + '\\' + 'z_m.csv'
    pd.DataFrame(res).to_csv(TargetFile, header = header, index = None)

    for i in range(len(res)):
        if res[i, d] > 0:               # 提取输出值小于0的所有样本点（即位于失效区域）
            res_FC = res[0:i, :]
            break
    header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y_m', 'y_ANN']
    TargetFile = OutDir + '\\' + 'z_FC.csv'
    pd.DataFrame(res_FC).to_csv(TargetFile, header = header, index = None)

    x_FC = res_FC[:sample_num, 0:d]
    # y_FC = res_FC[:sample_num, d]

    m = len(x_FC)

    # 二分法查找各个条件分布函数的零点，根据单调性计算条件失效概率
    P_conditional_1 = np.zeros(m)
    for i in range(m):
        ff = lambda xx: ANN(np.hstack((xx, x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], x_FC[i, 5], x_FC[i, 6], x_FC[i, 7])).reshape(1,-1), model, x, y)
        zero_point = binary_search(ff, 0.98E9, 1.02E9, 1E4)
        P_conditional_1[i] = 1 - truncnorm.cdf(zero_point, a = -2, b = 2, loc = 1E9, scale = 1E7)
        print(i)

    P_conditional_8 = np.zeros(m)
    for i in range(m):
        ff = lambda xx: ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], x_FC[i, 5], x_FC[i, 6], xx)).reshape(1,-1), model, x, y)
        zero_point = binary_search(ff, 5.04E-6, 7.56E-6, 1E-10)
        P_conditional_8[i] = (zero_point - 5.04E-6) / (7.56E-6 - 5.04E-6)
        print(i)

    P_f = 0
    m = len(x_FC)
    for i in range(m):
        # f_x = truncnorm.pdf(x_FC[i, 0], a = -5, b = 5, loc = 0, scale = 1) * truncnorm.pdf(x_FC[i, 1], a = -5, b = 5, loc = 0, scale = 1)
        f_x = 1
        g_x = 0
        for j in range(m):
            ANN_1 = ANN(np.hstack((x_FC[i, 0], x_FC[j, 1], x_FC[j, 2], x_FC[j, 3], x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            ANN_2 = ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[j, 2], x_FC[j, 3], x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            ANN_3 = ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[j, 3], x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            ANN_4 = ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            ANN_5 = ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            ANN_6 = ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], x_FC[i, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            ANN_7 = ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], x_FC[i, 5], x_FC[i, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
            # 计算条件失效概率
            if ANN_1 < 0 and ANN_2 < 0 and ANN_3 < 0 and ANN_4 < 0 and ANN_5 < 0 and ANN_6 < 0 and ANN_7 < 0:
                
                ff = lambda xx: ANN(np.hstack((x_FC[i, 0], xx, x_FC[j, 2], x_FC[j, 3], x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
                zero_point = binary_search(ff, 7665736, 8749116, 1E1)
                P_conditional_2 = 1 - truncnorm.cdf(zero_point, a = -2, b = 2, loc = 8207426, scale = 270845)
                ff = lambda xx: ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], xx, x_FC[j, 3], x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
                zero_point = binary_search(ff, 5.072125674, 6.904558326, 1E-5)
                P_conditional_3 = (zero_point - 5.072125674) / (6.904558326 - 5.072125674)
                ff = lambda xx: ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], xx, x_FC[j, 4], x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
                zero_point = binary_search(ff, 301.392, 342.608, 1E-2)
                P_conditional_4 = truncnorm.cdf(zero_point, a = -2, b = 2, loc = 322, scale = 10.304)
                ff = lambda xx: ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], xx, x_FC[j, 5], x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
                zero_point = binary_search(ff, 0.607, 1.393, 1E-5)
                P_conditional_5 = truncnorm.cdf(zero_point, a = -2, b = 2, loc = 1, scale = 0.1965)
                ff = lambda xx: ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], xx, x_FC[j, 6], x_FC[j, 7])).reshape(1,-1), model, x, y)
                zero_point = binary_search(ff, 1.520176E7, 1.582224E7, 1E2)
                P_conditional_6 = (zero_point - 1.520176E7) / (1.582224E7 - 1.520176E7)
                ff = lambda xx: ANN(np.hstack((x_FC[i, 0], x_FC[i, 1], x_FC[i, 2], x_FC[i, 3], x_FC[i, 4], x_FC[i, 5], xx, x_FC[j, 7])).reshape(1,-1), model, x, y)
                zero_point = binary_search(ff, 0.84, 1.16, 1E-7)
                P_conditional_7 = 1 - (zero_point - 0.84) / (1.16 - 0.84)
            
                g_x = g_x + f_x /(P_conditional_1[j] * P_conditional_2 * P_conditional_3 * P_conditional_4 * P_conditional_5 * P_conditional_6 * P_conditional_7 * P_conditional_8[i])
        g_x = g_x / m
        if g_x != 0:
            P_f = P_f + f_x / g_x
        print(i)
    P_f = P_f / m
    print('最终失效概率：%f'%(P_f))
    return P_f


# 路径
i_name = 'IRIS_3EHRS_8000_4.25_0718.i'  # i文件名称
r_name = 'IRIS_3EHRS_8000_4.25_0718.r'  # r文件名称
OutDir = r'C:\Users\JH\Desktop\IRIS数据集0718'  # 保存迭代数据集的位置
copy_dir = r'D:\copydir'  # 备份.i .o stripf的路径
# filepath_inputs = r'C:\Users\JH\Desktop\inputs.xlsx'  # 输入参数路径
path_relap5 = r'D:\IRIS'  # relap5路径
path_initialDOE = r'C:\Users\JH\Desktop\IRIS数据集0718\IRIS_4.25_totalLEVEL_min.csv'  # 初始数据集路径
path_dataforiteration = r'C:\Users\JH\Desktop\IRIS数据集0718\inputdatas_for_iteration.csv'  # 使用神经网络计算失效概率的同一数据集路径
path_ANN = r'D:\IRIS\model'  # 保存神经网络模型的位置
path_extradata = OutDir + '\\IRIS_extra'  # 额外的输入参数位置及前缀
path_totaldata = OutDir + '\\IRIS_total'  # 样本总体的位置及前缀
path_extrmedata = OutDir + '\\IRIS_extrme.csv'  # 极值配置数据的路径
failure_criteria = 16.08  # 失效准则
sample_num = 30  # 抽样次数
d = 8  # 输入参数维度

res = total(sample_num, d, i_name, r_name, OutDir, copy_dir, path_relap5, path_initialDOE, failure_criteria, path_ANN, path_dataforiteration, path_extradata, path_totaldata, path_extrmedata)
print(res)
