"""
steamp_pipeline.py

this is a pipeline that takes in steam app id's, and outputs a dataframe
for each id that it can grab

author: sean bergen
"""

import steamspypi
# not used at this point so it is commented out
# import steam
import pandas as pd


"""
grab_data
    appID   -> steam appID of game/software you are grabbing data from
    verbose -> flag that will print/not print out data

this function uses steamspypi functions to grab metadata which includes:
    appid -> id of the app on steam, number
    name  -> name of the game
    developer -> developer/studio which made the game
    publisher -> publisher that put the game onto steam
    score_rank -> unsure
    positive -> number of positive reviews of the game on steam
    negative -> number of negative reviews of the game on steam
    userscore -> unsure

    owners -> estimation of range for number of users who own a game
    average_forever -> average amount of playtime in minutes for a game across
all users since its launch
    average_2weeks -> average amount of playtime in minutes for a game across
all users over the past 2 weeks
    median_forever -> median playtime in minutes for a game across all users
since its launch
    median_2weeks -> median playtime in minutes for a game across all users
over the past 2 weeks
    
    price -> current price of the game
    initialprice -> undiscounted price of the game if it is on discount
    discount -> amount of discount on a game currently

    ccu -> concurrent users at the time of query

    languages -> list of languages that are supported by the game

    genre -> genres that the game has been classified as being in
    tags -> set of tags that users have associated a game with along with the
number of times that tag was used by a user

returns:
    data -> queried information in a dictionary structure

THINGS TO MAYBE ADD
    column with a '0' or '1' if 'Multiplayer' is listed in the tags
    column for "number of languages" maybe

    remove userscore or score_rank potentially

    steam API to gather additional metadata
"""
def grab_data(appID, verbose=False):
    # from steamspypi
    data_request = dict()
    data_request['request'] = 'appdetails'
    data_request['appid'] = str(appID)
    data = steamspypi.download(data_request)
    # then from steam
    # if i add calls from steam api they will be here

    # print statement for testing information we get back
    # enabled with "verbose=True"
    if verbose:
        for i in data:
            print(i, data[i])
    return data

"""
steam_pipeline:
    game_list -> list of appIDs which represent a list of games to query
    verbose   -> flag for printing additional information when querying

returns:
    dataframe containing all queried information for all games in the list

THINGS TO ADD
    argument/flag to write dataframe out to CSV maybe, though user could do
this in one or two lines by using pandas
"""
def steam_pipeline(game_list, verbose=False):
    data_list = []
    for appID in game_list:
        data_list.append(grab_data(appID, verbose))
    return pd.DataFrame.from_dict(data_list)

# examples of code below
"""
a = steam_pipeline([570, 730, 440], verbose=True)
print(a)
print(a.columns)
"""
