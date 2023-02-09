import numpy as np
import pandas as pd
import json
import argparse
from scipy.spatial import KDTree

def load_df(filepath='match_entries.csv'):
    """Load data from csv and return as pandas dataframe"""
    df = pd.read_csv(filepath)
    return df

def normalize_df(df):
    """Normalize the dataframe, keep min and max stored for normalizing new entries"""
    #normalize our columns
    df_scaled = df.copy()
    #store max, min in dict
    norm_dict = {}
    for column in df.columns[33:]:
        norm_dict[column] = [df_scaled[column].max(), df_scaled[column].min()]
        df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (df_scaled[column].max() - df_scaled[column].min()) 

    return df_scaled, norm_dict

def create_trees(df):
    """Create KDTree for each champion"""
    # create a dictionary to store the KDTrees for each champion
    kdt_dict = {}

    # create a KDTree for each champion
    for championId in df['championId'].unique():
        champion_data = df[df['championId'] == championId].iloc[:,33:]
        kdt_dict[championId] = KDTree(champion_data)

    return kdt_dict

def load_champ_df(champ_df_filepath='champ_df.csv'):
    #load in champ_df
    champ_df = pd.read_csv(champ_df_filepath)

    return champ_df

def champ_name_to_id(champ_df, champ_names):
    '''returns champ ids list from champ names list'''
    champ_list = []
    for champ in champ_names:
        champId = champ_df[champ_df['name'] == champ]['key'].array[0]
        champ_list.append(champId)
    return champ_list

def convert_query(champ_list, champ_df, norm_dict):
    #champ_df indices we want
    cols = champ_df.columns[4:].to_list()
    #create unique for ally and enemy sums
    ally_cols = ["ally_" + x for x in cols]
    enemy_cols = ["enemy_" + x for x in cols]

    ally_ids = champ_list[0:5]
    enemy_ids = champ_list[5:10]
    ally_stats = champ_df[champ_df['key'].isin(ally_ids)].sum()[4:].to_list()
    enemy_stats = champ_df[champ_df['key'].isin(enemy_ids)].sum()[4:].to_list()

    stats = ally_stats + enemy_stats

    #new series to store vals in
    summed_features = pd.Series(data=stats, index=ally_cols+enemy_cols)
    
    #normalize features
    for key, value in norm_dict.items():
        summed_features[key] = (summed_features[key] - value[1]) / (value[0] - value[1])
    return summed_features

def query(df, champ_list, summed_features, kdt_dict):
    '''
    takes in a champ list where 0 = self, 1-4 = allies, 5-9 = enemies
    '''
    query = summed_features
    indices = kdt_dict[champ_list[0]].query(query, k=15, distance_upper_bound=2.0)
    #retrieve datapoints for desired champion
    result = df[df['championId'] == champ_list[0]].iloc[indices[1]]
    
    #filter out bad performances if pool is deep enough
    mask = (result['kda'] > 3) & (result['win'] == 1)
    if result[mask].shape[0] > 5:
        result = result[mask]

    return result

def load_item_data(item_data_filepath='data/13.1.1/data/en_US/item.json'):
    """Load in item data"""

    f = open(item_data_filepath, encoding="utf8")
    item_data = json.load(f)['data']

    return item_data

def filter_items(item_recs, item_data):
    """Filter out items that are not valid for the build"""
    #filter item recs to only include boots and completed items
    item_recs_filtered = [] #nondestructive, new list
    boot_rec = False #bool to store if a boot has been recommended
    for item in item_recs:
        item_valid = False #bool to store if meets conditions
        item_desc = item_data[str(item)] #get item info
        if 'Boots' in item_desc['tags'] and item_desc['gold']['total'] > 350 and item_desc['depth'] == 2: #if it is a boot and tier 2
            if not boot_rec:
                item_valid = True
                boot_rec = True
        if item_desc['gold']['total'] >= 2200: #if it costs more or equal to cheapest legendary item
            item_valid = True
        if item_desc['gold']['total'] > 1300: #filtering out starter items and components
            if not item_desc['gold']['purchasable'] == False: #if it is not purchasable, dont perform these checks
                if item_desc['depth'] == 3: #if it is depth of 3, valid item
                    item_valid = True
                if item_desc['depth'] == 4: #checking for ornn item, false if so
                    item_valid = False
        if item_valid:    
            item_recs_filtered.append(item)

    return item_recs_filtered

def item_recommendations(result, item_data):

    item_matrix = result.iloc[:,3:9].values.ravel() #1d np array of items
    item_matrix = item_matrix[item_matrix != 0] #remove where no item

    items, count = np.unique(item_matrix, return_counts=True) #get unique items and their counts
    count_sort = np.argsort(-count) #sort by count descending
    item_recs = items[count_sort] #sort items by count descending

    item_recs_filtered = filter_items(item_recs, item_data)

    return item_recs_filtered

def main():
    parser = argparse.ArgumentParser(description='Get item recommendations for a given champion')
    parser.add_argument('champ_names', type=str, nargs='+', help='Champion names, 0 = self, 1-4 = allies, 5-9 = enemies')
    args = parser.parse_args()
    champ_names = args.champ_names
    #load in data
    print('Loading dataframe...')
    df = load_df()
    #normalize df
    print('Normalizing dataframe...')
    df_scaled, norm_dict = normalize_df(df)
    #create trees
    print('Creating trees...')
    kdt_dict = create_trees(df_scaled)
    #load in champion data
    print('Loading champion data...')
    champ_df = load_champ_df()
    #convert query
    print('Converting query...')
    champ_list = champ_name_to_id(champ_df, champ_names)
    summed_features = convert_query(champ_list, champ_df, norm_dict)
    #query
    print('Querying...')
    result = query(df, champ_list, summed_features, kdt_dict)
    #load in item data
    print('Loading item data...')
    item_data = load_item_data()
    #get item recommendations
    print('Getting item recommendations...')
    item_recs = item_recommendations(result, item_data)

    print([item_data[str(x)]['name'] for x in item_recs])

if __name__ == '__main__':
    main()