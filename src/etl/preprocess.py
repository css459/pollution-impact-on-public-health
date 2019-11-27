#
# Intro to ML Final Project
# Cole Smith
# preprocess.py
#

import json
import geocoder


# TODO
def make_lat_lon_map(inputs, output_json_file='data/latlon.json', load_from=None):
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
        g = geocoder.osm(query)

        if not g.ok:
            print("[ WRN ]", query, "--", str(g))
            errors.append(query)
            dump_errors(errors)

        else:
            print("[ MAP ] Found", g.latlng, "for", query)
            json_acc[query] = g.latlng
            dump(json_acc)
