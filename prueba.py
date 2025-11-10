import pandas as pd
df = pd.read_csv('data/train.csv')
print(df.isna().sum()) # Muestra cuántas filas tienen NaN por columna
df_clean = df.dropna() # Elimina filas con cualquier NaN
