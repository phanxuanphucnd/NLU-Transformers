import pandas as pd

from sklearn.model_selection import train_test_split

data_df = pd.read_csv('nlu-kcloset.csv', encoding='utf-8')

train_df, test_df = train_test_split(
    data_df, test_size=0.15, stratify=data_df['intent'], random_state=123)

train_df.to_csv('./train.csv', encoding='utf-8', index=False)
test_df.to_csv('./test.csv', encoding='utf-8', index=False)