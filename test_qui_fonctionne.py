import numpy as np
import pandas as pd
import json
import os
import platform
import argparse
import requests
import sys
from scipy.spatial import KDTree
from feature_build import load_data

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()

def normalize_df(df):
    """Normalize the dataframe, keep min and max stored for normalizing new entries"""
    #normalize our columns
    df_scaled = df.copy()
    #store max, min in dict
    norm_dict = {}
    for column in df.columns[31:]:
        norm_dict[column] = [df_scaled[column].max(), df_scaled[column].min()]
        df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (df_scaled[column].max() - df_scaled[column].min())

    return df_scaled, norm_dict

# def create_trees(df):
#     """Create KDTree for each champion"""
#     # create a dictionary to store the KDTrees for each champion
#     kdt_dict = {}

#     # create a KDTree for each champion
#     for championId in df['championId'].unique():
#         champion_data = df[df['championId'] == championId].iloc[:,31:]
#         kdt_dict[championId] = KDTree(champion_data)

#     return kdt_dict

def load_champ_df(champ_df_filepath='data/champ_df.csv'):
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

def load_item_data(version_filepath='data/version.json'):
    """Load in item data"""
    # open json file, get version
    f = open(version_filepath)
    version = json.load(f)[0]
    item_data_filepath=f'data/{version}/{version}/data/en_US/item.json'
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

def get_client_champ_data(install_path=None):
    '''
    Get champ select data from local client
    Auto uses default install directory for each OS, can specify install_path to force
    '''
    system = platform.system()

    if not install_path:
        if system == "Linux":
            user_id = os.getuid()
            user_info = pwd.getpwuid(user_id)
            username = user_info.pw_name
            install_path = f"/home/{username}/Games/league-of-legends/drive_c/Riot Games/League of Legends/"

        elif system == "Windows":
            install_path = "C:/Riot Games/League of Legends/"

        elif system == "Darwin":
            install_path = "/Applications/League of Legends.app/Contents/LoL/"

    f = open(install_path+"lockfile", "r")
    client_info = f.read().split(sep=":")
    f.close()

    port = client_info[2]
    pw = client_info[3]
    auth=('riot', pw) #auth tuple for requests. riot static username, password changes on client restart

    url = "https://127.0.0.1:"+port
    champ_sel = "/lol-champ-select/v1/session"
    current_champ = "/lol-champ-select/v1/current-champion"
    data = requests.get((url + champ_sel), auth=auth, verify=False)
    self_champ = requests.get((url + current_champ), auth=auth, verify=False).json()

    # construct champ list
    allies = [x['championId'] for x in data.json()['myTeam'] if x['championId'] != self_champ]
    enemies = [x['championId'] for x in data.json()['theirTeam']]
    champ_list = [self_champ] + allies + enemies

    return champ_list

print('Loading champion data...')
champ_df = load_champ_df()

#load in data
print('Loading dataframe...')
df = load_data(table="match_features_fix_simple")
sys.argv = ['script_name', '-m', 'Champion1', 'Champion2', '-c']
parser = argparse.ArgumentParser(description='Get item recommendations for a given champion')
parser.add_argument('-m', '--manual', dest='champ_names', type=str, nargs='+', required=False, help='Champion names as strings, 0 = self, 1-4 = allies, 5-9 = enemies')
parser.add_argument('-c', '--client', dest='client', action='store_true', required=False, help='Get champion data from client')
args = parser.parse_args()

if args.champ_names is None and args.client is False:
    # raise Exception('No arguments given, please use -h for help')
    print('No arguments given, please use -h for help')

if args.client:
    try:
        champ_list = get_client_champ_data()
    except:
        print('Client not running or no game in progress')
        raise
else:
    champ_names = args.champ_names
    champ_list = champ_name_to_id(champ_df, champ_names)


def find_nearest_neighbors_for_user_champion(champion_id, champion_features, kdtree, df_match, num_neighbors=min(15,len(df.query(f"championId == {champ_list[0]}")))):
    """
    Trouver les num_neighbors voisins les plus proches pour le champion de l'utilisateur 
    parmi les matchs où ce champion est joué.
    
    Arguments:
    champion_id -- ID du champion pour lequel on cherche les voisins (ex. Yasuo = 157)
    champion_features -- Caractéristiques du champion à partir de summed_features
    kdtree -- KDTree construit uniquement pour les matchs où le champion de l'utilisateur est joué
    df_match -- DataFrame contenant les données des matchs
    num_neighbors -- Nombre de voisins les plus proches à retourner (par défaut 15)
    
    Retourne:
    Les lignes correspondantes aux matchs les plus proches où le championId est celui de l'utilisateur.
    """
    
    print(f"len df : {len(df.query(f'championId == {champ_list[0]}'))}")
    # Trouver les num_neighbors voisins les plus proches
    distances, indices = kdtree.query(champion_features.reshape(1, -1), k=num_neighbors)
    # distances = distances[0][1:]
    # indices = indices[0][1:]
    
    # Récupérer les lignes correspondantes à partir des indices
    closest_matches = df_match[df_match['championId'] == champion_id].iloc[indices[0]]
    # closest_matches[1:]
    
    return distances, closest_matches

def create_trees_for_user_champion(df, champion_id):
    """
    Créer un KDTree uniquement pour les matchs où le championId correspond à celui de l'utilisateur.
    
    Arguments:
    df -- DataFrame contenant les données des matchs
    champion_id -- ID du champion de l'utilisateur (ex. Yasuo = 157)
    
    Retourne:
    KDTree basé sur les matchs du champion de l'utilisateur.
    """
    # Filtrer les matchs où le championId est celui de l'utilisateur
    champion_data = df[df['championId'] == champion_id].iloc[:, 31:].values
    if champion_data.shape[0] > 0:
        return KDTree(champion_data)
    else:
        raise ValueError(f"Aucun match trouvé pour le champion avec l'ID {champion_id}")

def calculate_confidence_from_distances(distances, min_distance, max_distance):
    """
    Calcule l'indicateur de confiance en fonction des distances aux voisins les plus proches, avec normalisation.
    
    Arguments:
    distances -- Tableau des distances par rapport aux voisins les plus proches
    min_distance -- La distance minimale observée dans l'ensemble des données
    max_distance -- La distance maximale observée dans l'ensemble des données
    
    Retourne:
    Un score de confiance basé sur la distance normalisée.
    """
    # Calculer la distance moyenne
    mean_distance = np.mean(distances)
    
    # Normalisation de la distance par rapport aux distances min et max
    normalized_distance = (mean_distance - min_distance) / (max_distance - min_distance)
    
    # Calculer l'indicateur de confiance (plus la distance est faible, plus la confiance est élevée)
    confidence_score = 1 - normalized_distance  # Inverse pour que la plus petite distance donne une confiance élevée
    
    return confidence_score

# def find_global_min_max_distances(kdtree, df_match, champion_id):
#     """
#     Trouver les distances minimale et maximale globales pour normaliser l'indicateur de confiance.
    
#     Arguments:
#     kdtree -- KDTree construit pour le champion
#     df_match -- DataFrame contenant les données des matchs
#     champion_id -- ID du champion pour lequel on calcule les distances
    
#     Retourne:
#     La distance minimale et maximale entre les matchs du champion dans l'ensemble de données.
#     """
#     # Filtrer les caractéristiques du champion
#     champion_features = df_match[df_match['championId'] == champion_id].iloc[:, 31:].values
    
#     # Trouver les 2 voisins les plus proches (y compris lui-même)
#     distances, _ = kdtree.query(champion_features, k=2)
    
#     # Ignorer la première colonne qui est la distance de chaque point à lui-même (donc 0)
#     valid_distances = distances[:, 1]  # Prendre la 2ème colonne qui contient la vraie distance
    
#     # Retourner les distances minimales et maximales
#     return valid_distances.min(), valid_distances.max()

#normalize df
print('Normalizing dataframe...')
df_scaled, norm_dict = normalize_df(df)
#create trees
print('Creating trees...')
# kdt_dict = create_trees(df_scaled)
#load in champion data

#convert query
print('Converting query...')


print("create tree")
kdtree = create_trees_for_user_champion(df, champ_list[0])
print(f"kdtree : {kdtree}")

# Trouver les distances minimale et maximale pour normaliser
# min_distance, max_distance = find_global_min_max_distances(kdtree, df, champ_list[0])
# print(f"Min distance : {min_distance}, Max distance : {max_distance}")

user_champ_features = df[df['championId'] == champ_list[0]].iloc[0, 31:].values
distances_neighbors_matches, nearest_neighbors_matches = find_nearest_neighbors_for_user_champion(champ_list[0], user_champ_features, kdtree, df, len(df.query(f'championId == {champ_list[0]}'))) #, num_neighbors=min(15,len(df_match.query(f"championId == {champ_list[0]}"))))
print(f"distance proches voisins : {distances_neighbors_matches}")

# Calculer l'indicateur de confiance normalisé
# confidence_score = calculate_confidence_from_distances(distances_neighbors_matches, min_distance, max_distance)
# print(f"Indicateur de confiance basé sur la distance normalisée : {confidence_score}")

# if isinstance(nearest_neighbors_matches, pd.Series):
#     nearest_neighbors_matches = nearest_neighbors_matches.to_frame()
nearest_neighbors_matches = nearest_neighbors_matches.astype({col: 'int64' for col in nearest_neighbors_matches.select_dtypes(include=['float64']).columns})

summed_features = convert_query(champ_list, champ_df, norm_dict)

#query
# print('Querying...')
# result = query(df, champ_list, summed_features, kdt_dict)
#load in item data
print('Loading item data...')
item_data = load_item_data()
#get item recommendations
print('Getting item recommendations...')
# print(f"\n\nRecommended items for your champ, in order of build frequency in similar games:")
# item_recs = item_recommendations(result, item_data)
print(f"\n\nRecommended items for your champ ({champ_df.query(f'key == {champ_list[0]}')['id'].values[0]}), in order of build frequency in similar games:")
item_recs = item_recommendations(nearest_neighbors_matches, item_data)

print([item_data[str(x)]['name'] for x in item_recs])

import tkinter as tk

# Supposons que vous ayez une liste d'objets recommandés
objets_recommandes = [item_data[str(x)]['name'] for x in item_recs]

# Création de la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Objets Recommandés")

# Création d'une liste pour afficher les objets
liste = tk.Listbox(fenetre)
for objet in objets_recommandes:
    liste.insert(tk.END, objet)
liste.pack()

# Lancement de la boucle principale
fenetre.mainloop()