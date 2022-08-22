#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import random
import numpy as np
import pickle
import os
from lxml import html
import requests
from time import sleep
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

filepath = os.path.abspath('')


# %%


#get historical data and figure out teamID dilemma
old_data = pd.read_csv('MNCAATourneyDetailedResults.csv')
idteams = pd.read_csv('MTeams.csv')
old_data = old_data.merge(idteams, 
                          left_on = 'WTeamID', 
                          right_on = 'TeamID')
old_data = old_data.merge(idteams, 
                          left_on = 'LTeamID', 
                          right_on = 'TeamID')

old_data = old_data[['Season',
          'WTeamID',
          'WScore',
          'LTeamID',
          'LScore',
          'TeamName_x',
         'TeamName_y']].sort_values(by = 'Season', ascending = True)
old_data = old_data.rename(columns={"TeamName_x": "Winning Team",
                   "TeamName_y": "Losing Team"})

#randomly put the teams in the Team 1 or Team 2 column 
#this is because we need to train the model on both wins and losses
old_data['Team 1'] = np.random.randint(2, size=len(old_data))
old_data['Team 2'] = 1 - old_data['Team 1']

#Now we can do some reformatting here...
conds_1 = [(old_data['Team 1'] == 1),(old_data['Team 1'] == 0)]
actions_1 = [old_data['Winning Team'], old_data['Losing Team']]
actions_w = ['W', 'L']

conds_2 = [(old_data['Team 1'] == 0),(old_data['Team 1'] == 1)]
actions_2 = [old_data['Winning Team'], old_data['Losing Team']]

old_data['Win?'] = np.select(conds_1,actions_w)
old_data['Team 2'] = np.select(conds_2,actions_2,default='Other')
old_data['Team 1'] = np.select(conds_1,actions_1,default='Other')
old_data = old_data.drop(columns = ['WTeamID',
                                    'LTeamID',
                                   'Winning Team',
                                   'Losing Team'])

old_data


# %%


pd.DataFrame(old_data['Team 2'].unique()).to_clipboard()
old_data


# %%

#scraping function to get SOS
def scrape(url,
           xpath = "//*[@id='meta']/div[2]/p[6]/text()[1]"):
    sleep(2)
    page = requests.get(url)

    tree = html.fromstring(page.content)  

    
    result = tree.xpath()
    return result[0][1:6]


# %%

links = pd.read_excel('page_links.xlsx')
links['SOS'] = [ scrape(x) for x in links['link-default'] ]
connect = pd.read_excel('name_connection.xlsx')
links = links.merge(connect, 
                    left_on = 'lower',
                    right_on = 'Sportsref')
pickle.dump(links,open(os.path.join(filepath, 'links'),'wb'))
links


# %%


# I already wrote pickle, don't really want to write it again so I'll just scrape it 
links = pd.read_pickle(f'links')


# %%
def get_stats(team, year):
    #find the right row in the links df to get the link
    right_row = links[(links['Year'] == year) &
                      (links['Kaggle'] == team)]
    right_row = right_row.reset_index()
    right_link = right_row['link-default'][0]
    right_link

    #read the data
    sports_ref_data = pd.read_html(right_link)

    #two tables we need from the site
    basic_stats = sports_ref_data[1]


    #get the differences in basic statistics between team and Opp
    basic_stats = basic_stats.loc[:, basic_stats.columns!='Unnamed: 0']
    bad_df = basic_stats.index.isin([1,3])
    basic_stats = basic_stats[~bad_df]
    basic_stats = basic_stats.apply(pd.to_numeric, errors='coerce')
   

    #basic basketball stats
    data_export = pd.DataFrame(basic_stats.iloc[0][['2P','3P', '3P%','FT',
                               'ORB','TRB','AST','STL',
                              'BLK','TOV','PF','PTS']]).transpose()
    
    
    data_export['PA'] = basic_stats.iloc[1,-1]
    
    data_export['SOS'] = right_row['SOS']
    

    
    #pythagoras
    data_export['Pyth'] = (basic_stats.iloc[0, -1]**16.5)/(basic_stats.iloc[0, -1]**16.5 +
                                                          basic_stats.iloc[1, -1]**16.5)
    
    #data_export['Guard Usage'] = guards
    return data_export

get_stats('UNC Asheville', 2003)


# %%





# %%
final_df = pd.DataFrame()
for i in range(len(links)):
    year = links.iloc[i,:]['Year']
    team = links.iloc[i,:]['Kaggle']
    

    
    final_df = final_df.append(get_stats(team, year))
    sleep(3)
final_df.insert(0, "Year", list(links['Year']))
final_df.insert(1, "Team", list(links['Kaggle']))

final_df


# %%
#deal with poll data  
def poll_stats(team,year):
    #find the right row in the links df to get the link
    right_row = links[(links['Year'] == year) &
                      (links['Kaggle'] == team)]
    right_row = right_row.reset_index()
    right_link = right_row['link-default'][0][:-5]+str('-schedule.html')


    #read the data
    poll = pd.read_html(right_link)
    a = poll[0].transpose()
    a.columns = ['Brendan']
    return len(a) - pd.to_numeric(a['Brendan'], errors='coerce').isna().sum()


empty = []
for i in range(len(links)):
    sleep(3)
    year = links.iloc[i,:]['Year']
    team = links.iloc[i,:]['Kaggle']
    empty.append(poll_stats(team, year))

poll_data = pd.DataFrame()
poll_data['Year'] = list(links['Year'])
poll_data['Team'] = list(links['Kaggle'])
poll_data['Weeks Ranked'] = empty
poll_data


# %%
#now merge the final dataframe with the poll data
final_df = final_df.merge(poll_data, left_on = ['Year','Team'], right_on = ['Year','Team'])
final_df


# %%

#create kenpom df for AdjEm, O_rtg, D_rtg variables
kenpom_df = pd.DataFrame()
for i in range(2003,2020):
    
    url = str('https://kenpom.com/index.php?y='+str(i)+'&s=TeamName')

    header = {
      "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"
    }

    r = requests.get(url, headers=header)

    dfs = pd.read_html(r.text)
    a = dfs[0][['Unnamed: 1_level_0','Unnamed: 4_level_0','Unnamed: 5_level_0', 'Unnamed: 7_level_0']]
    a.columns = ['Team','AdjEm', 'O_rtg','D_rtg']
    a.insert(0,"Year",i)
    kenpom_df = kenpom_df.append(a)
kenpom_df['Team'] = kenpom_df['Team'].str.replace(' \d+', '')

kenpom_df


# %%

'''
now, we have to make name associations. I have created 
kenpom_names.xlsx to show a connection between the names
that we have for the final_df, and the names we scraped.

I did something similar earlier to link the Kaggle data that was
scraped, and the names of the teams on sportsreference.com

Now, we merge kenpom_df with the name connector and merge that 
with the giant data scrape

'''
kenpom_df = pd.read_pickle('kenpom_df')
kp_names = pd.read_excel('kenpom_names.xlsx')
kp_names['Kenpom']
kenpom_df

x = kenpom_df.merge(kp_names,
                    left_on = ['Team'],
                    right_on = ['Kenpom'])
final_data = final_df.merge(x,
            left_on = ['Year', 'Team'],
              right_on = ['Year', 'Kaggle'])

# final_data
final_data


# %%


'''
Now we connect final data and old data. 
Old data was downloaded from Kaggle, which is why we
kept that variable with us. If we can match Season/Year and 
Team1, Team2 with Kaggle, we are solid. 

# '''
# final_data
# # a lot of our metrics weren't available before 2011
# old_data = old_data[old_data['Season'] >= 2011]

# old_data
# missing_list = []
# for i in range(len(old_data)):
#     year = old_data.iloc[i,:]['Season']
#     team = old_data.iloc[i,:]['Team 1']
#     missing_list.append(len(final_data[(final_data['Year'] == year)&(final_data['Kaggle_y'] == team)]))
# #old_data.iloc[0,:]

# missing_list

# #len((final_data[(final_data['Year'] == 2011)&(final_data['Kaggle'] == 'Old Dominion')]))
# final_data.columns
final_df.merge(x, 
               left_on = ['Year', 'Team'],
              right_on = ['Year', 'Kaggle'])


# %%


#for now, 5 rows didn't read right. This function should take care of that 
def viable_row(year, team1, team2):
    if len(final_data[(final_data['Year'] == year)&(final_data['Kaggle'] == team1)]) +
    len(final_data[(final_data['Year'] == year)&(final_data['Kaggle'] == team2)]) == 2:
        return True
    else:
        return False

old_data = old_data.reset_index(drop = True)
final_data


# %%


collective_frame = pd.DataFrame()
for i in range(len(old_data)):
    game_row = old_data.iloc[i,:]

    year,team_1, team_2 = old_data.iloc[i,0],old_data.iloc[i,3],old_data.iloc[i,4]

    if viable_row(year, team_1, team_2) == True:
        team_1_df = final_data[(final_data['Year'] == year)&(final_data['Kaggle'] == team_1)].iloc[:,2:-3].drop(columns = ['Team_y'])
        team_2_df = final_data[(final_data['Year'] == year)&(final_data['Kaggle'] == team_2)].iloc[:,2:-3].drop(columns = ['Team_y'])

        num_t1_preds = len(list(team_1_df.columns))
        num_t2_preds = len(list(team_2_df.columns))
        datalist = [team_1_df.iloc[0,j] for j in range(num_t1_preds)] + [team_2_df.iloc[0,j] for j in range(num_t1_preds)]
        datalist.insert(0,year)
        datalist.insert(1,team_1)
        datalist.insert(2,team_2)
        datalist.insert(3+2*len(list(team_1_df.columns)),game_row['Win?'])
        x = pd.DataFrame(datalist).transpose()
        collective_frame = pd.concat([collective_frame,x])
        
namelist = ['year','team_1','team_2'] + [str(i)+' team 1' for i in (list(team_1_df.columns))] + [str(i)+' team 2' for i in (list(team_2_df.columns))] + ['W/L']
new_names_map = {x.columns[k]:namelist[k] for k in range(len(namelist))}
collective_frame.rename(new_names_map, axis=1, inplace=True)


conds = [(collective_frame['W/L'] == 'W'),(collective_frame['W/L'] == 'L')]
actions = [1, 0]
collective_frame['W/L'] = np.select(conds,actions)
collective_frame

collective_frame


# %%





# %%


pickle.dump(collective_frame,open(os.path.join(filepath, '2003-2019_collective'),'wb'))


# %%


subtracting_frame = pd.DataFrame()
for i in range(len(old_data)):
    game_row = old_data.iloc[i,:]
    year,team_1, team_2 = old_data.iloc[i,0],old_data.iloc[i,3],old_data.iloc[i,4]
    if viable_row(year, team_1, team_2) == True:
        team_1_df = final_data[(final_data['Year'] == year)&(final_data['Kaggle'] == team_1)].iloc[:,2:-3].
        drop(columns = ['Team_y']).apply(pd.to_numeric, errors='coerce')
        
        team_2_df = final_data[(final_data['Year'] == year)&(final_data['Kaggle'] == team_2)].iloc[:,2:-3]
        .drop(columns = ['Team_y']).apply(pd.to_numeric, errors='coerce')
        
        subtracted = team_1_df.reset_index(drop = True).subtract(team_2_df.reset_index(drop = True))
        
        subtracted.insert(0,'Year',year)
        subtracted.insert(1,'Team 1',team_1)
        subtracted.insert(2,'Team 2',team_2)
        subtracted.insert(len(subtracted.columns),'Win',old_data.iloc[i,-1])
        subtracting_frame = pd.concat([subtracting_frame,subtracted])
        
conds = [(subtracting_frame['Win'] == 'W'),(subtracting_frame['Win'] == 'L')]
actions = [1, 0]
subtracting_frame['Win'] = np.select(conds,actions)


subtracting_frame
        
subtracting_frame


# %%





# %%


subtracting_frame


# %%





# %%


pickle.dump(subtracting_frame,open(os.path.join(filepath, '2003-2019_subtracting_frame'),'wb'))


# %%


def binary(number):
    if number>=0:
        return 1
    else:
        return 0


binary_frame = subtracting_frame.copy()
data_cols = binary_frame.iloc[:,3:24].columns

binary_frame[data_cols]
for i in range(len(data_cols)):
    binary_frame[data_cols[i]] = binary_frame[data_cols[i]].apply(binary) 
    
conds = [(binary_frame['Win'] == 'W'),(binary_frame['Win'] == 'L')]
actions = [1, 0]
binary_frame['Win'] = np.select(conds,actions)
binary_frame

#pickle.dump(binary_frame,open(os.path.join(filepath, 'binary_frame'),'wb'))

binary_frame


# %%


final_data

