# opioid crisis analysis library.
# David Li, MTH 600.

import numpy as np
from numpy import array
import pandas as pd

# one-to-one correspondence between states and initials.
state_in = {
        "kentucky": "ky",
        "ohio": "oh",
        "pennsylvania": "pa",
        "virginia": "va",
        "west virginia": "wv",
        }
in_state = {v: k for k, v in state_in.items()}

# given state and county, retrieve longitude and latitude.

def locate(state_in, county, df_geo):
    """
    return lattitude and longitude coordinates given state initials and county.
    """
    county = county.lower() + " county"
    indexes = [i for i, cty in enumerate(df_geo["NAME"]) if cty.lower() == county]

    for i in indexes:
        # check if state agrees with input state.
        if list(df_geo["USPS"])[i].lower() == state_in.lower():
            lat = list(df_geo["INTPTLAT"])[i]
            lon = list(df_geo["INTPTLONG                                                                                                               "])[i]
            # I don't know why INTPLTLONG is like this.

            return (lat, lon)

# dataframe analysis.
def feature_extract(df, df_metadata):
    """
    Takes a data set and metadata set.
    Extracts features which we want to work with.
    Returns a sorted list of labels.

    Assumption: given a label HC*X_VC**,
    reject if X is even.
    """

    features = ["GEO.display-label"]

    for key in df_metadata["GEO.id"]:
        # iterates through each key.

        # skip the following labels:
        # if they are geographical (already included)
        # if they are error estimates.

        if key[0] == "G":
            continue
        if int(key[3])%2 == 0:
            continue

        desc = df[key].iloc[0] # get description
        desc = desc.split() # split to words.
        if desc[0] == "Percent;":
            continue # reject percentage estimates.
        
        # also skip if column of feature is string.
        first_dat = df[key].iloc[1] # get first piece of data.
        try:
            x = int(first_dat) # see if it can be converted to integer
        except ValueError:
            continue

        features.append(key)

    return features

def feature_index(ddf, ddf_metadata):
    """
    Takes a list of dataframes and metadataframes.
    Returns dictionary of universally accessible features
    with corresponding labels in their respective years.
    """

    def _get_key(df, desc):
        # get the key from a df corresponding to desc.
        i = list(df.iloc[0]).index(desc)
        return df.keys()[i]

    n = len(ddf) # number of dataframes
    total_features = [None for _ in range(n)]

    for i in range(n):
        _f_extract = feature_extract(ddf[i], ddf_metadata[i])[1:] # get features.
        _f_dict = dict()

        for f in _f_extract:
            
            desc = ddf[i][f].iloc[0] # get description
            _f_dict[desc] = f

        total_features[i] = _f_dict # map from description to label

    # get all map keys.
    total_keys = [f.keys() for f in total_features]
    univ_desc = ["Geography"] + list(set(total_keys[0]).intersection(*total_keys[1:]))

    # define a universal map.
    univ_map = dict()

    for desc in univ_desc:
        univ_map[desc] = [None for _ in range(n)]
        for i in range(n):
            _df = ddf[i]
            _df_metadata = ddf_metadata[i]

            univ_map[desc][i] = _get_key(_df, desc)

    return univ_map

def state_and_county(geography):
    """
    Takes geography data from socio-economic factors,
    returns initials of state and county name (without " county")
    """
    county, state = geography.split(", ") # separate county from state
    state = state_in[state.lower()] # convert state to initials
    county = ' '.join(county.split()[:-1]) # remove the word "county"
    return (state, county)

def drug_matrix(df_nflis, substanceNamesDict):
    """
    Takes the nflis dataframe and names of substances,
    returns an array of substance use vectors.
    """

    n = df_nflis.shape[0] # number of instances.
    m = len(substanceNamesDict) # number of distinct drugs.

    drug_use_matrix = np.zeros((n, m)) # nxm dimensional zero.

    # iterate through each example.
    # for each example, determine drug type and drug report count.
    for i in range(n):
        substanceName = df_nflis["SubstanceName"].iloc[i]
        substanceNameIndex = substanceNamesDict[substanceName]
        drugReports = int(df_nflis["DrugReports"].iloc[i])
        drug_use_matrix[i][substanceNameIndex] = drugReports

    return drug_use_matrix

def drug_vector(yyyy, state_in, county, df_nflis, substanceNamesDict, identify=False):
    """
    Takes year, state, county, and returns overall drug reports as vector.

    First input is instances of drug use,
    second is indices which allow for searching up corresponding drug.
    """

    n = df_nflis.shape[0]
    d_matrix = drug_matrix(df_nflis, substanceNamesDict) # drug matrix.

    # get all vectors corresponding to the instance.
    drug_indices = [
            i for i in range(n) if df_nflis["YYYY"][i] == np.int(yyyy)
            and df_nflis["State"][i].lower() == state_in.lower()
            and df_nflis["COUNTY"][i].lower() == county.lower()
            ]
    
    d_vec = sum(d_matrix[i] for i in drug_indices)

    if identify:
        # identifies the drugs corresponding to indices.
        # compute inverse drug map.

        drug_dict = {
                df_nflis["SubstanceName"].iloc[i]:df_nflis["DrugReports"].iloc[i]
                for i in drug_indices
                }

        return d_vec, drug_dict
    else:
        return d_vec

if __name__ == "__main__":

    # run this code.

    pass
