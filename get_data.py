from riotwatcher import LolWatcher, ApiError
import requests
import pandas as pd
import sqlite3
import argparse
import logging
import time

class lol_interface:
    def __init__(self, api_key):
        self.api_key = api_key
        self.lol_watcher = LolWatcher(api_key)

    def update_key(self, api_key):
        self.api_key = api_key
        self.lol_watcher = LolWatcher(api_key)

def get_top_players(region, testing=False):
    '''
    uses riotwatcher API to retrieve players in challenger, GM, and masters. 
    returns a list of all summoner IDs
    '''

    #return challengers
    challengers = lol_obj.lol_watcher.league.challenger_by_queue(region, 'RANKED_SOLO_5x5')

    #return grandmasters
    gms = lol_obj.lol_watcher.league.grandmaster_by_queue(region, 'RANKED_SOLO_5x5')

    #return masters
    masters = lol_obj.lol_watcher.league.masters_by_queue(region, 'RANKED_SOLO_5x5')

    #list of the above objects
    if testing:
        all_top_players = [challengers]
    else:
        all_top_players = [challengers, gms, masters]

    #loop through and concat all summoner Ids
    summoner_ids = []
    for division in all_top_players:
        for entry in division['entries']:
            summoner_ids.append(list(entry.values())[0])

    return summoner_ids

def get_puuid(summoner_ids):
    '''
    take in a summoner ID from riot API and fetches the users puuid.
    this is done because other queries require the puuid.
    returns dict object mapping summoner id to puuid.
    '''
    #dict to store vals
    summid_to_puuid = {}
    for summoner in summoner_ids:
        try:
            summid_to_puuid[summoner] =  lol_obj.lol_watcher.summoner.by_id(region, summoner)['puuid']
        except ApiError as e:
            print(f"{e.response.status_code}: Waiting 10s")
            time.sleep(10)

    return summid_to_puuid

def get_champ_mastery(summoner_ids, summid_to_puuid, points=100000):
    '''
    takes in the summoner id and gets the champion id for each champion above points.
    default points value is 100,000.
    returns a dict mapping puuid to list of champion ids.
    '''
    #create dict to hold champion masteries for each summoner by puuid
    mastery_dict = {}

    #fill dict
    for summoner in summoner_ids:
        #make request for champion masteries, store in variable
        masteries = lol_obj.lol_watcher.champion_mastery.by_summoner(region, summoner)
        
        #loop through, adding high mastery champs to list
        puuid = summid_to_puuid[summoner]
        mastery_dict[puuid] = []
        for mastery in masteries:
            if mastery.get('championPoints') > 100000:
                mastery_dict[puuid].append(mastery.get('championId'))

    return mastery_dict

def get_match_data(mastery_dict, num_matches=10):
    '''
    takes in mastery_dict and returns a list of dicts of match data, as well as a set of all match IDs scanned
    num_matches: between 1-100
    '''

    #create list to store dict objects
    data_rows = []
                
    #store set of matches already looked through
    matches_scanned = set()

    #list of features we want to record
    features = ['puuid', 'championId', 'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 
                'kills', 'deaths', 'assists', 'totalDamageDealtToChampions', 'role', 'teamPosition', 'teamId', 'gameEndedInEarlySurrender', 'win']
    

    #expecting API errors
    for key, value in mastery_dict.items():
        
        #store matchlist for each puuid
        try:
            match_list = lol_obj.lol_watcher.match.matchlist_by_puuid(region, key, count = num_matches)
        
        except ApiError as e:
            if e.response.status_code == 403:
                print("bad or expired API key, paste new one here:")
                api_key = input()
                lol_obj.update_key(api_key=api_key)
                match_list = lol_obj.lol_watcher.match.matchlist_by_puuid(region, key, count = num_matches)
            else:
                print(f"{e.response.status_code}: Waiting 10s")
                time.sleep(secs=10)
                match_list = lol_obj.lol_watcher.match.matchlist_by_puuid(region, key, count = num_matches)


        for match in match_list:
            if match not in matches_scanned:
                
                #store match data in variable
                try:
                    match_data = lol_obj.lol_watcher.match.by_id(region, match)

                except ApiError as e:
                    if e.response.status_code == 403:
                        print("bad or expired API key, paste new one here:")
                        api_key = input()
                        lol_obj.update_key(api_key=api_key)
                        match_data = lol_obj.lol_watcher.match.by_id(region, match)
                    else:
                        print("Connection error, waiting 10s then resuming operation")
                        time.sleep(secs=10)
                        match_data = lol_obj.lol_watcher.match.by_id(region, match)

                #store participant information in variable to iterate over (list of dicts) if classic game
                if match_data['info']['gameMode'] == 'CLASSIC':
                    player_info = match_data['info']['participants']
                    #create dict of champs on team1, team2
                    champions_in_game = {}
                    champions_in_game[100] = []
                    champions_in_game[200] = []
                    for player in player_info:
                        #add champ played to dict
                        champions_in_game[player['teamId']].append(player['championId'])
                        #check to see if player in our list of masters+ players
                        if player['puuid'] in mastery_dict.keys(): 
                            #check to see if player on a high mastery champ
                            if player['championId'] in mastery_dict[player['puuid']]:
                                #get player data, store in dictionary
                                player_data = {}
                                for feature in features:
                                    player_data[feature] = player[feature]
                                player_data['patch'] = match_data['info']['gameVersion']
                                player_data['match_id'] = match
                                player_data['champions_in_game'] = champions_in_game
                                #append dictionary to list
                                data_rows.append(player_data)
                                
                    #print out to watch progress
                    #print('champion ID: ',player_data['championId'],', win:',player_data['win'])
                    #print('champs in game: ',player_data['champions_in_game'])
                                
                #append match_id to matches_scanned set
                matches_scanned.add(match)

    return data_rows, matches_scanned

def match_to_df(data_rows):
    '''
    converts data_rows (list of dicts) into dataframe, and manipulates columns to be sql-supported datatypes.
    '''

    df = pd.DataFrame.from_dict(data_rows)
    #drop where teamPosition empty
    df = df[df['teamPosition'] != '']
    #drop where game ended in early surrender
    df = df[df['gameEndedInEarlySurrender'] == False]

    #lets construct columns from the teamId and champions_in_game column

    #new column, list of champions on player's team
    df['teammates_championId'] = df.apply(lambda x: x['champions_in_game'].get(x['teamId']), axis=1)

    #new column, list of enemy champions
    opposite_team_dict = {100:200, 200:100}
    df['opposite_team_id'] = df['teamId'].map(opposite_team_dict)
    df['enemies_championId'] = df.apply(lambda x: x['champions_in_game'].get(x['opposite_team_id']), axis=1)

    #split list into individual columns
    player_cols = ["enemies_championId", "teammates_championId"]
    for col in player_cols:
        temp_df = df[col].apply(pd.Series)
        temp_df = temp_df.add_prefix(col[:-10])
        df = pd.concat([df, temp_df], axis=1)

    #drop redundant columns
    df = df.drop(labels=["teammates_championId", "enemies_championId"], axis=1)
    df = df.drop(labels=["champions_in_game","opposite_team_id"], axis=1)

    return df

def df_to_sql(df, database='data/matches.db', table_name='player_items_champions'):
    '''
    stores dataframe into a sql database. appends data to table if table already exists.
    '''
    conn = sqlite3.connect(database)
    df.to_sql(name="player_items_champions", con=conn, if_exists='append', index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', required=True, help="Riot Developer API key, found at https://developer.riotgames.com")
    parser.add_argument('-r', '--region', required=True, help="Region to query data for. Supported values can be found at https://developer.riotgames.com/docs/lol")
    args = parser.parse_args()

    #riot games api key
    api_key = args.key
    #set region, init lol watcher obj
    region = args.region
    lol_obj = lol_interface(api_key=api_key)
    #lol_watcher = LolWatcher(api_key)
    logging.info("LolWatcher object created.")
    
    summoner_ids = get_top_players(region=region, testing=False)
    logging.info(f"Top players stored: {len(summoner_ids)} entries.")
    
    summid_to_puuid = get_puuid(summoner_ids=summoner_ids)
    logging.info("puuids retrieved.")
    
    mastery_dict = get_champ_mastery(summoner_ids=summoner_ids, summid_to_puuid=summid_to_puuid)
    logging.info("Champ mastery retrieved.")

    data_rows, matches_scanned = get_match_data(mastery_dict=mastery_dict, num_matches=20)
    logging.info(f"Match data retrieved, {len(matches_scanned)} matches scanned, {len(data_rows)} entries.")

    df = match_to_df(data_rows=data_rows)
    logging.info(f"Converted to dataframe object.")

    df_to_sql(df=df)
    logging.info(f"Stored in sql database")

    