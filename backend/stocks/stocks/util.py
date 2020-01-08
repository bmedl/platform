import numpy as np

def get_timeseries(df, time_steps, output_col_num, limit, predict_interval):
  dim_0 = df.shape[0] - time_steps
  dim_1 = 9
  x = np.zeros((dim_0, time_steps, dim_1))
  y = np.zeros((dim_0,))

  for i in range(dim_0):
    x[i] = df[i:time_steps+i]
    if (df[time_steps+i, output_col_num] - df[time_steps+i-predict_interval, output_col_num] > df[time_steps+i, output_col_num]*limit):
      y[i] = 1
    elif (abs(df[time_steps+i, output_col_num] - df[time_steps+i-predict_interval, output_col_num]) < df[time_steps+i, output_col_num]*limit):
      y[i] = 0
    else:
      y[i] = -1
  return x, y
  
def trim_dataset(df, batch_size):
    no_of_rows_drop = df.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return df[:-no_of_rows_drop]
    else:
        return df