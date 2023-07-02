import pandas as pd


def one_hot_column(df, column_name):
    column = df[column_name]
    unique_values = set()
    
    for value_list in column:
        value_list = eval(value_list)
        for item in value_list:
            unique_values.add(item['name'])
    
    onehot_columns = pd.DataFrame(0, index=df, columns=list(unique_values))
    for i, value_list in enumerate(column):
        value_list = eval(value_list)
        for item in value_list:
            onehot_columns.at[i, item['name']] = 1
            
    df_encoded = pd.concat([df, onehot_columns], axis=1)
    return df_encoded

def one_hot_multiple_columns(df, columns):
    for col in columns:
        df = one_hot_column(df, col)
        
    df.drop(columns, inplace=True)
    return df
        

def clean_data():
    meta_df = pd.read_csv('movies-dataset/movies_metadata.csv', low_memory=False)
    keywords_df = pd.read_csv('movies-dataset/keywords.csv')
    credits_df = pd.read_csv('movies-dataset/credits.csv')
    
    keywords_df = one_hot_multiple_columns(keywords_df, columns=['keywords'])
    print('keywords_df.columns', list(keywords_df.columns))
    credits_df = one_hot_multiple_columns(credits_df, columns=['cast', 'crew'])
    print('credits_df.columns', list(credits_df.columns))
    meta_df = one_hot_multiple_columns(meta_df, columns=['genres'])
    print('meta_df.columns', list(meta_df.columns))

    merged_df = pd.merge(meta_df, keywords_df, on='id')
    merged_df = pd.merge(merged_df, credits_df, on='id')

    merged_df.to_csv('movies-dataset/cleaned.csv')
    
    

clean_data()