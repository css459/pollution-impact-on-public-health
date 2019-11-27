#
# Intro to ML Final Project
# Cole Smith
# load.py
#

import glob
import json
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
    combined = _load('tri')

    # Clean column names
    cols = combined.columns
    cols = [c.split(' - ')[-1] for c in cols]
    cols = [c.split('. ')[-1].lower() for c in cols]
    combined.columns = cols
    combined = combined.rename(columns={'production wste (8.1-8.7)': 'production waste',
                                        'latitude': 'lat', 'longitude': 'lon'})

    # Select Relevant
    sel = ['year', 'lat', 'lon', 'industry sector',
           'fugitive air', 'stack air', 'water', 'underground',
           'underground cl i', 'underground c ii-v', 'landfills',
           'rcra c landfill', 'other landfills', 'land treatment',
           'surface impndmnt', 'rcra surface im', 'other surface i',
           'on-site release total', 'trns rlse', 'trns trt',
           'total transfers', 'm10', 'm41', 'm62', 'm40 metal', 'm61 metal', 'm71',
           'm81', 'm82', 'm72', 'm63', 'm66', 'm67', 'm64', 'm65', 'm73', 'm79',
           'm90', 'm94', 'm99', 'off-site release total', 'm20', 'm24', 'm26',
           'm28', 'm93', 'off-site recycled total', 'm56', 'm92',
           'm40 non-metal', 'm50', 'm54',
           'm61 non-metal', 'm69', 'm95', 'off-site treated total',
           'total transfer', 'total releases', 'releases', 'on-site contained',
           'off-site contain', 'production waste']

    combined = combined[sel].dropna(axis=1)
    combined.columns = [c.replace(' ', '_') for c in combined.columns]

    # Pivot industry
    d = pd.get_dummies(combined['industry_sector'], prefix='sector')
    df = pd.concat([combined, d], axis=1).drop(['industry_sector'], axis=1)

    return df


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

    # MSA Conversion Policy
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
            area = areas[0]

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

    # Select Relevant
    sel = ['State', 'County', 'Life Expectancy',
           'Life Expectancy Range']
    life = life[sel].dropna()

    # Fix column names
    life.columns = [c.replace(' ', '_').lower() for c in life.columns]

    # Fix Life Exp Range to Min and Max
    rng = list(life.life_expectancy_range)
    rng_max = [float(str(r).split('-')[1].strip()) for r in rng]
    rng_min = [float(str(r).split('-')[0].strip()) for r in rng]

    life['life_expectancy_max'] = rng_max
    life['life_expectancy_min'] = rng_min
    life = life.rename(columns={'life_expectancy': 'life_expectancy_avg'})
    life = life.drop('life_expectancy_range', axis=1).dropna()

    return life.groupby(['state', 'county'], as_index=False).mean()


def load_cancer_tri_aqi():
    lat_lon_json = {}
    with open('data/latlon.json', 'r') as fp:
        lat_lon_json = json.load(fp)

    def get_lat_lon(q):
        if q in lat_lon_json:
            return lat_lon_json[q]
        else:
            return [None, None]

    def change_precision(a, prec=3):
        if None not in a:
            return [round(float(e), prec) for e in a]
        else:
            return None

    print("[ LOAD ] Loading Cancer...")
    cancer = load_cancer()
    loc = list(cancer.msa)
    lat = [get_lat_lon(i)[0] for i in loc]
    lon = [get_lat_lon(i)[1] for i in loc]

    cancer['lat'] = change_precision(lat)
    cancer['lon'] = change_precision(lon)

    print(cancer)

    print("[ LOAD ] Loading TRI...")
    tri = load_tri()
    tri['lat'] = change_precision(list(tri.lat))
    tri['lon'] = change_precision(list(tri.lon))

    print(tri)

    print("[ LOAD ] Loading AQI...")
    aqi = load_aqi()
    loc = list(aqi.state_name + ", " + aqi.county_name)
    lat = [get_lat_lon(i)[0] for i in loc]
    lon = [get_lat_lon(i)[1] for i in loc]

    aqi['lat'] = change_precision(lat)
    aqi['lon'] = change_precision(lon)

    print(aqi)

    print("[ LOAD ] Loading Life Exp...")
    life = load_life_exp()
    loc = list(life.state + ", " + life.county)
    lat = [get_lat_lon(i)[0] for i in loc]
    lon = [get_lat_lon(i)[1] for i in loc]

    life['lat'] = change_precision(lat)
    life['lon'] = change_precision(lon)

    print(life)

    # Group up
    aqi = aqi.dropna().groupby(by=['year', 'lat', 'lon'], as_index=False).sum()
    tri = tri.dropna().groupby(by=['year', 'lat', 'lon'], as_index=False).sum()
    life = life.dropna().groupby(by=['lat', 'lon'], as_index=False).sum()
    cancer = cancer.dropna().groupby(by=['year', 'lat', 'lon'], as_index=False).sum()

    # Split up the join due to memory constraint
    joined = pd.concat([aqi, tri, cancer], join='outer', ignore_index=True)
    merged = joined.groupby(by=['year', 'lat', 'lon'], as_index=False).sum()

    print(merged)

    # merged = pd.merge(aqi, tri, on=['year', 'lat', 'lon'])
    # merged = pd.merge(merged, cancer, on=['year', 'lat', 'lon'])
    # merged = pd.merge(merged, life, on=['year', 'lat', 'lon'])
    # merged3 = pd.join([merged, life]).groupby(by=['lat', 'lon']).sum()

    merged = pd.concat([merged, life], join='outer', ignore_index=True)
    merged = merged.groupby(by=['lat', 'lon'], as_index=False).sum()

    print(merged)

    merged = merged.dropna()
    merged.to_csv('data/merged.csv')

    # TODO: Fix merge
    return merged


if __name__ == '__main__':
    m = load_cancer_tri_aqi()

    # TODO Rerun
    # out = load_life_exp()
    # print('[ INF ] Starting Lat Lon Table for Life Exp')
    # loc = list(out.state + ", " + out.county)
    # # loc = out.msa
    # loc = list(set(loc))

    # from src.etl.preprocess import make_lat_lon_map
    #
    # make_lat_lon_map(loc)
