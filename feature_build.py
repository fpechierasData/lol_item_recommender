import pandas as pd
import json
import sqlite3

def load_data():
    """Load data from database and return as pandas dataframe"""
    conn = sqlite3.connect('match_entries.db')
    df = pd.read_sql("SELECT * FROM player_items_champions", conn)
    conn.close()
    return df

def create_champ_df(data_filepath='data/13.1.1/data/en_US/champion.json', feature_filepath='champ_matrix_filled.csv', save=True):
    """Create dataframe of champions and their attributes"""

    # open json file, get data
    f = open('data/13.1.1/data/en_US/champion.json', encoding="utf8")
    champ_data = json.load(f)['data']

    # define features we want to keep
    features = ['version','id','key','name', 'info', 'tags']

    #create a list of dictionaries, each dictionary is a champion
    champ_list = []
    for key, value in champ_data.items():
        champ_dict = {}
        temp = value
        for feature in features:
            champ_dict[feature] = temp[feature]
            if feature == 'info':
                for key, value in value[feature].items():
                    champ_dict[key] = value
        champ_list.append(champ_dict)

    # create dataframe from list of dictionaries
    champ_df = pd.DataFrame().from_dict(champ_list)
    champ_df = champ_df.drop(labels=['info'], axis=1)

    #load in manually-defined feature csv and join with current dataframe
    champ_features = ['mobility','poke','sustained','burst','engage','disengage','healing']
    temp_df = pd.read_csv('champ_matrix_filled.csv')
    temp_df = temp_df[champ_features]
    champ_df = pd.concat([champ_df, temp_df], axis=1)

    #one hot encode the tags column, and sum to get back to original row shape
    temp_df = pd.get_dummies(champ_df['tags'].explode(), columns=['tags'])
    temp_df = temp_df.groupby(temp_df.index).sum()
    
    #merge temp_df with champ_df
    champ_df = pd.concat([champ_df, temp_df],axis=1)

    #transform key column to int
    champ_df['key'] = champ_df['key'].astype(int)

    #drop tags and difficulty columns
    champ_df = champ_df.drop(labels=['tags','difficulty'], axis=1)

    #save to csv
    if save:
        champ_df.to_csv('champ_df.csv', index=False)
        
    return champ_df

def get_summed_features(df, champ_df):
    """Get summed features for ally and enemy teams"""

    #champ_df indices we want
    cols = champ_df.columns[4:].to_list()
    
    #create unique for ally and enemy sums
    ally_cols = ["ally_" + x for x in cols]
    enemy_cols = ["enemy_" + x for x in cols]
    
    #new dataframe to store vals in
    summed_features = pd.DataFrame(columns=ally_cols+enemy_cols)
    
    for index, row in df.iterrows():
        #enemies list
        enemy_ids = row[21:26].to_list()
        #ally list
        ally_ids = row[26:31].to_list()
        
        #list of vals to fill
        ally_stats = champ_df[champ_df['key'].isin(ally_ids)].sum()[4:].to_list()
        enemy_stats = champ_df[champ_df['key'].isin(enemy_ids)].sum()[4:].to_list()
        
        stats = ally_stats + enemy_stats
        summed_features.loc[len(summed_features)] = stats
        
    #merge with match_ids
    df = pd.concat([df, summed_features], axis=1)
    
    #create KDA column
    df['kda'] = (df['kills'] + df['assists']) / df['deaths']
    df.loc[df['deaths'] == 0, 'kda'] = df['kills'] + df['assists'] #where deaths = 0, set kd_ratio to kills + assists 

    #move the kda column to the front
    column_to_move = df.pop("kda") #remove column
    #insert column at position 10
    df.insert(10, "kda", column_to_move)

    #return dataframe
    return df

if __name__ == "__main__":
    df = load_data()
    champ_df = create_champ_df()
    print('summing features')
    df = get_summed_features(df, champ_df)
    df.to_csv('match_entries.csv', index=False) #save to csv