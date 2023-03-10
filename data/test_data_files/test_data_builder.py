

import pandas as pd
import pickle
import pyarrow.feather as feather


# make example dataframe to test
df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'd': [
                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'd': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'f': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

df_1 = df.loc[:df.shape[0]/2]
df_2 = df.loc[df.shape[0]/2:]


# In[17]:


df_1, df_2 = train_test_split(df, test_size=0.5, random_state=42)
df_1.to_csv(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/csv_test/test_1.csv')
df_2.to_csv(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/csv_test/test_2.csv')


df_3, df_4 = df_1*1, df_2*2
df_5, df_6 = df_1*3, df_2*3
df_3.to_pickle(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/pkl_test/test_1.pkl')
df_4.to_pickle(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/pkl_test/test_2.pkl')

df_5.reset_index().to_feather(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/feather_test/test_1.feather')
df_6.reset_index().to_feather(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/feather_test/test_2.feather')

df_7, df_8, df_9 = df_1/1000, df_2/1000, df_1/1000

df_7.to_csv(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/csv_test/dont_detect.csv')
df_8.to_pickle(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/pkl_test/dont_detect.pkl')
df_9.reset_index().to_feather(
    '/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/feather_test/dont_detect.feather')
