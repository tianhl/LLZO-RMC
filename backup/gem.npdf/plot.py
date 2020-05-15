import numpy as np


def loadData(filename=None, suffix=None):
      if((filename is None)and(suffix is not None)):
          filename = self.default+suffix
      print('Read  file: ' + filename)
      
      f=open(filename)
      l=f.readlines()
      x_array = []
      y_array = []
      for line in l:
          print(line)
          if(line[0].startswith("#")):
              continue
          i = line.split()
          print(i)
          if(len(i)<2):
              continue
          x_array.append(float(i[0]))
          y_array.append(float(i[1]))
      return np.array(x_array), np.array(y_array)
