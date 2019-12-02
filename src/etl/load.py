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

from src.etl.preprocess import make_lat_lon_map

RAW_DATA = 'data/raw/'
PREPARED_DATA = 'data/prepared'

with open('data/latlon.json', 'r') as fp:
    lat_lon_json = json.load(fp)


def _load(subpath):
    path = os.path.join(RAW_DATA, subpath, '*.csv')
    all_files = glob.glob(path)

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def _get_lat_lon(q):
    if q in lat_lon_json:
        return lat_lon_json[q]
    else:
        return [None, None]


def _change_precision(a, prec=0):
    acc = []
    for e in a:
        if e is None:
            acc.append(None)
        else:
            acc.append(round(float(e), prec))
    return acc


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

    # Fix schema
    df.year = df.year.astype(int)
    df.lat = df.lat.astype(float)
    df.lon = df.lon.astype(float)

    # Shift Precision
    df.lat = _change_precision(df.lat)
    df.lon = _change_precision(df.lon)

    return df.dropna()


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
    df = df.dropna().groupby(by=['year', 'state_name', 'county_name'], as_index=False).mean()

    # Convert to Lat/Lon
    loc = list(df.state_name + ", " + df.county_name)
    make_lat_lon_map(list(set(loc)))

    lat = [_get_lat_lon(i)[0] for i in loc]
    lon = [_get_lat_lon(i)[1] for i in loc]

    df['lat'] = _change_precision(lat)
    df['lon'] = _change_precision(lon)

    # Fix Schema
    df.year = df.year.astype(int)
    df.lat = df.lat.astype(float)
    df.lon = df.lon.astype(float)

    # Group up
    df = df.dropna().groupby(by=['year', 'lat', 'lon'], as_index=False).sum()

    return df.dropna()


def load_cancer():
    path = os.path.join(RAW_DATA, "health/United States and Puerto Rico Cancer Statistics, 1999-2016 Incidence.txt")
    cancer = pd.read_csv(path, delimiter='\t')

    # Select relevant columns
    sel = ['Year', 'Leading Cancer Sites', 'MSA', 'Count', 'Population', 'Age-Adjusted Rate']
    cancer = cancer[sel]
    cancer.columns = [c.replace(' ', '_').lower() for c in cancer.columns]

    # Pivot criteria
    d = pd.get_dummies(cancer['leading_cancer_sites'], prefix='cancer')
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
    cancer = cancer.dropna()

    # Convert to Lat/Lon
    loc = list(cancer.msa)
    make_lat_lon_map(list(set(loc)))

    lat = [_get_lat_lon(i)[0] for i in loc]
    lon = [_get_lat_lon(i)[1] for i in loc]

    cancer['lat'] = _change_precision(lat, prec=0)
    cancer['lon'] = _change_precision(lon, prec=0)

    # Fix Schema
    cancer.year = cancer.year.astype(int)
    cancer.lat = cancer.lat.astype(float)
    cancer.lon = cancer.lon.astype(float)

    # Group up
    cancer = cancer.dropna().groupby(by=['year', 'lat', 'lon'], as_index=False).sum()

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

    life = life.groupby(['state', 'county'], as_index=False).mean()

    # Convert to Lat/Lon
    print("Getting lat lon list")
    loc = list(life.state + ", " + life.county)
    make_lat_lon_map(list(set(loc)))

    lat = [_get_lat_lon(i)[0] for i in loc]
    lon = [_get_lat_lon(i)[1] for i in loc]

    life['lat'] = _change_precision(lat, prec=0)
    life['lon'] = _change_precision(lon, prec=0)

    # Fix Schema
    life.lat = life.lat.astype(float)
    life.lon = life.lon.astype(float)

    # Group up
    life = life.dropna().groupby(by=['lat', 'lon'], as_index=False).sum()

    return life.dropna()


def load_all(from_file=True):
    if from_file:
        return pd.read_csv('data/merged.csv')

    print("[ LOAD ] Loading AQI...")
    aqi = load_aqi()
    print(aqi)

    print("[ LOAD ] Loading TRI...")
    tri = load_tri()
    print(tri)

    print("[ LOAD ] Loading Life...")
    life = load_life_exp()
    print(life)

    print("[ LOAD ] Loading Cancer...")
    cancer = load_cancer()
    print(cancer)

    # Join
    j = aqi.merge(tri, on=['year', 'lat', 'lon'], how='inner')
    j = j.merge(cancer, on=['year', 'lat', 'lon'], how='left')
    j = j.merge(life, on=['lat', 'lon'], how='left')
    j = j.dropna()

    j.to_csv('data/merged.csv', index=False)
    return j


if __name__ == '__main__':
    m = load_all(from_file=False)
    m.to_csv("data/merged.csv", index=False)
