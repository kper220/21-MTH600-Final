# opioid crisis analysis library.

import pandas as pd

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
    univ_desc = list(set(total_keys[0]).intersection(*total_keys[1:]))

    return univ_desc

if __name__ == "__main__":

    # run this code.

    pass
