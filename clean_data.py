import json
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


def one_hot_column(df, column_name):
    column = df[column_name]
    column.fillna('[]', inplace=True)
            
    column = column[column.apply(lambda x: isinstance(x, str) and isinstance(eval(x), list))]
    column = column.apply(lambda x: {item['name'] for item in eval(x)})

    mlb = MultiLabelBinarizer()
    onehot_columns = pd.DataFrame(mlb.fit_transform(column), columns=mlb.classes_)
    # print(f'{column_name} unique values: ', list(onehot_columns.columns))

    onehot_columns.rename(
        columns={col: f'{column_name}=' + col for col in onehot_columns}, inplace=True)
    # print(f'{column_name} columns: ', list(df_encoded.columns))

    return onehot_columns


def one_hot_multiple_columns(df, columns):
    onehots = pd.DataFrame()
    for col in columns:
        hot = one_hot_column(df, col)
        onehots = pd.concat([onehots, hot], axis=1)
        print(onehots.shape)
        print(col, 'ended')
    
    df = pd.concat([df, onehots], axis=1)
    df.drop(columns, axis=1, inplace=True)
    return df

def change_dtype_str(df, columns):
    for col in columns:
        df[col] = df[col].astype(str)
    return df
    

def clean_data():
    keywords_df = pd.read_csv('movies-dataset/keywords.csv')
    keywords_df = one_hot_multiple_columns(keywords_df, columns=['keywords'])
    # print('keywords_df.columns', list(keywords_df.columns))
    keywords_df.to_csv('cleaned-dataset/keywords.csv', index=False)

    # did not do this because of low memory:
    # credits_df = pd.read_csv('movies-dataset/credits.csv')
    # credits_df = one_hot_multiple_columns(credits_df, columns=['cast', 'crew'])
    # # print('credits_df.columns', list(credits_df.columns))
    # credits_df.to_csv('cleaned-dataset/credits.csv', index=False)

    meta_df = pd.read_csv('movies-dataset/movies_metadata.csv', low_memory=False)
    # print('meta_df.columns', list(meta_df.columns))
    meta_df.drop(columns=['adult', 'belongs_to_collection', 'budget', 'homepage', 'original_title', 'overview', 'production_companies',
                 'poster_path', 'release_date', 'revenue', 'runtime', 'original_language', 'status', 'tagline', 'title', 'video'], inplace=True)
    meta_df = one_hot_multiple_columns(meta_df, columns=['genres', 'spoken_languages', 'production_countries'])
    meta_df.to_csv('cleaned-dataset/meta.csv', index=False)


def merge_cleaned():
    print('merge started')
    keywords_df = pd.read_csv('cleaned-dataset/keywords.csv')
    # credits_df = pd.read_csv('cleaned-dataset/credits.csv')
    meta_df = pd.read_csv('cleaned-dataset/meta.csv', low_memory=False)
    
    for col in meta_df.columns:
        if col.startswith('keyword'):
            meta_df[col] = meta_df[col].astype(bool)
    
    for col in keywords_df.columns:
        if col.startswith('genres') or col.startswith('spoken_languages') or col.startswith('production_countries'):
            keywords_df[col] = keywords_df[col].astype(bool)
            
    
    meta_df['id'] = meta_df['id'].astype(str)
    keywords_df['id'] = keywords_df['id'].astype(str)
    
    # print(keywords_df.describe())
    # print(meta_df.describe())
    print('here')

    merged_df = pd.merge(meta_df, keywords_df, on='id')
    # merged_df = pd.merge(merged_df, credits_df, on='id')
    merged_df.to_csv('cleaned-dataset/merged.csv', index=False)
    

merge_cleaned()
