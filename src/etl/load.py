#
# Intro to ML Final Project
# Cole Smith
# load.py
#

import glob
import os
import re

import pandas as pd

RAW_DATA = 'data/raw/'
PREPARED_DATA = 'data/prepared'


def _load(subpath):
    path = os.path.join(RAW_DATA, subpath, '*.csv')
    all_files = glob.glob(path)

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def load_tri():
    """
    Loads the TRI CSV files into one DataFame

    :return: DataFrame
    """
    # Remove NA columns
    combined = _load('tri').dropna(axis=1)

    # Clean column names
    cols = combined.columns
    cols = [c.split(' - ')[-1] for c in cols]
    cols = [c.split('. ')[-1].lower() for c in cols]
    combined.columns = cols

    return combined


def load_aqi():
    """
    Loads the AQI CSV files into one DataFame

    :return: DataFrame
    """
    # Remove NA columns
    combined = _load('aqi').dropna(axis=1)

    # Select relevant columns
    sel = ['State Name', 'county Name', 'Date', 'AQI', 'Defining Parameter']
    combined = combined[sel]

    # Clean column names
    combined.columns = [c.replace(" ", "_").lower() for c in combined.columns]

    # Convert date column to year only
    combined.date = combined.date.str.split('-').str[0]
    combined = combined.rename(columns={'date': 'year'})
    combined = combined.astype({'year': 'int64'})

    # Pivot defining_parameter column
    d = pd.get_dummies(combined['defining_parameter'], prefix='defining')
    df = pd.concat([combined, d], axis=1).drop(['defining_parameter'], axis=1)

    return df


def load_cancer():
    path = os.path.join(RAW_DATA, "health/United States and Puerto Rico Cancer Statistics, 1999-2016 Incidence.txt")
    cancer = pd.read_csv(path, delimiter='\t')

    # Select relevant columns
    sel = ['Year', 'Leading Cancer Sites', 'MSA', 'Count', 'Population', 'Age-Adjusted Rate']
    cancer = cancer[sel]
    cancer.columns = [c.replace(' ', '_').lower() for c in cancer.columns]

    # Pivot criteria
    d = pd.get_dummies(cancer['leading_cancer_sites'], prefix='')
    cancer = pd.concat([cancer, d], axis=1).drop(['leading_cancer_sites'], axis=1)

    # Fix MSA
    tmp = cancer.msa
    ntmp = []

    # Also load manual map
    manual = {}
    with open('data/cancer_manual_map.txt', 'r') as fp:
        for l in fp:
            items = l.split('|')
            manual[items[0].strip()] = items[1].strip()

    for t in tmp:
        t = str(t)
        t = t.replace('-', ' ')
        if len(re.split(',|,,', t)) == 1:
            ntmp.append(t)
            continue

        states = re.split(',|,,', t)[1]
        state = states.split()[0]

        areas = t.split()
        if len(areas) > 1:
            area = t.split()[0] + ' ' + t.split()[1]
        else:
            area = area[0]

        tt = area + ", " + state

        if tt in manual:
            ntmp.append(manual[tt])
        else:
            ntmp.append(tt)

    cancer.msa = ntmp
    return cancer.dropna()


def load_life_exp():
    path = os.path.join(RAW_DATA, 'health', 'U.S._Life_Expectancy_at_Birth_by_State_and_Census_Tract_-_2010-2015.csv')
    life = pd.read_csv(path)

if __name__ == '__main__':
    out = load_life_exp()
    # print('[ INF ] Starting Lat Lon Table for Cancer')
    # loc = out.msa
    # loc = list(set(loc))

    # from src.etl.preprocess import make_lat_lon_map
    # make_lat_lon_map(loc)
