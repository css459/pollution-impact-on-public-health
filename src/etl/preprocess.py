#
# Intro to ML Final Project
# Cole Smith
# preprocess.py
#

import geocoder


# TODO
def make_lat_lon_map(inputs, output_json_file, load_from=None):
    url = 'http://localhost/nominatim/'

    # {"query": [lat, lon]}
    json_acc = {}

    # Remove duplicates
    inputs = list(set(inputs))

    if load_from:
        # TODO: Load JSON file and remove keys from input
        pass

    for query in inputs:
        query = str(query)
        json_acc[query] = geocoder.osm(query)
