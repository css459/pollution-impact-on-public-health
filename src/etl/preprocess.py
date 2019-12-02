#
# Intro to ML Final Project
# Cole Smith
# preprocess.py
#

import json

import geocoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


def split(df, y_cols=None, pct=0.20, shuffle=True, normalize=True):
    """
    Splits DataFrame into training and test sets. If `y_cols`
    is provided, also breaks out those columns as a separate
    Numpy Array. Will return either:
        x, y
        or
        x_train, y_train, x_test, y_test

    :param df:          DataFrame to split
    :param y_cols:      Y columns of DataFrame
    :param pct:         Validation Percent
    :param shuffle:     Shuffle the rows before splitting
    :param normalize    Scale the data using MaxAbsScaler
    :return:            Numpy Arrays
    """
    if y_cols:
        y = df[y_cols]
        cols = [c for c in df.columns if c not in y_cols]
        x = df[cols]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=pct, shuffle=shuffle)

        # Fit the scaler to training only
        if normalize:
            x_scaler = MaxAbsScaler()
            x_scaler.fit(x_train)
            y_scaler = MaxAbsScaler()
            y_scaler.fit(y_train)

            return x_scaler.transform(x_train), y_scaler.transform(y_train), \
                   x_scaler.transform(x_test), y_scaler.transform(y_test)
        else:
            return x_train, y_train, x_test, y_test
    else:
        # No Y columns provided, simply split the data
        x_train, x_test = train_test_split(df, test_size=pct, shuffle=shuffle)

        # Fit the scaler to training only
        if normalize:
            x_scaler = MaxAbsScaler()
            x_scaler.fit(x_train)
            return x_scaler.transform(x_train), x_scaler.transform(x_test)
        else:
            return x_train, x_test


def make_lat_lon_map(inputs, output_json_file='data/latlon.json', load_from='data/latlon.json'):
    """
    Queries OpenStreetMap for Lat/Lon pairs given a list
    of input queries. The results are incrementally written
    to an output file in `data` by default. Errors are logged
    to a separate file for manual lookup.

    :param inputs:              A list of UNIQUE values to query
    :param output_json_file:    Optional separate JSON output file
    :param load_from:           Resumes from another JSON file
                                entries in the file will not be
                                reprocessed
    :return: `None`
    """
    # {"query": [lat, lon]}
    json_acc = {}

    # Remove duplicates
    inputs = list(set(inputs))

    if load_from:
        with open(load_from, 'r') as json_file:
            json_acc = json.load(json_file)
        inputs = [i for i in inputs if i not in json_acc.keys()]

    def dump(j):
        with open(output_json_file, 'w') as fp:
            json.dump(j, fp, indent=4)

    def dump_errors(e):
        with open("data/nominatim_errors.txt", 'w') as fp:
            for err in e:
                fp.write(err + '\n')

    errors = []
    for query in inputs:
        query = str(query)
        if query not in json_acc:
            g = geocoder.osm(query)

            if not g.ok:
                print("[ WRN ]", query, "--", str(g))
                errors.append(query)
                dump_errors(errors)

            else:
                print("[ MAP ] Found", g.latlng, "for", query)
                json_acc[query] = g.latlng
                dump(json_acc)
