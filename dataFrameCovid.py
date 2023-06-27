import pandas as pd
import os
class DataFrameCovid:
    def getDf():
      # Monta dataframe com nome dos arquivos
      fn1 = os.listdir("D:/DropB/Faculeste/TCC/Imagens/COVID-19_Radiography_Dataset/Normal2/images")
      fn2 = os.listdir("D:/DropB/Faculeste/TCC/Imagens/COVID-19_Radiography_Dataset/COVID/images")

      ct1 = []
      for fn in fn1:
        ct1.append([0])

      df1 = pd.DataFrame({
        'arquivo': fn1,    
        'categoria': ct1   # Normal
      })
      #df1

      ct2 = []
      for fn in fn2:
        ct2.append([1])

      df2 = pd.DataFrame({
        'arquivo': fn2,
        'categoria': ct2   # COVID
      })
      df = df1.append(df2)
      return df