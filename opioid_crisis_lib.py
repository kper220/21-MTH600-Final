# opioid crisis analysis library.
# David Li, MTH 600.

import numpy as np
from numpy import array
import pandas as pd

# one-to-one correspondence between states and initials.
state_in_dict = {
        "kentucky": "ky",
        "ohio": "oh",
        "pennsylvania": "pa",
        "virginia": "va",
        "west virginia": "wv",
        }
in_state_dict = {v: k for k, v in state_in_dict.items()}

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

def feature_extract2(df, df_metadata):
    """
    Takes data set and metadata set, extracts features that are compatible with PCA (i.e. scores or percentages).
    """
    features = ["GEO.display-label"]

    for key in df_metadata["GEO.id"]:
        if key[0] == "G":
            continue
        desc = df[key].iloc[0] # get description
        desc = desc.split() # split to words.
        if desc[0] == "Percent;":
            first_dat = df[key].iloc[1]
            try:
                x = np.float64(first_dat) # see if can be converted to integer.
            except ValueError:
                continue
            features.append(key)

    return features

def feature_index2(ddf_yyyy, ddf_metadata_yyyy, include_geography=False, extraction=feature_extract):
    """
    Takes a list of dataframes and metadataframes (indexed by year).
    Returns dictionary of universally accessible features
    with corresponding labels in their respective years.
    """

    def _get_key(df, desc):
        # get key from df corresponding to desc.
        i = list(df.iloc[0]).index(desc)
        return df.keys()[i]

    n = len(ddf_yyyy) # number of dataframes.
    total_features = [None for _ in range(n)]
    
    for i, year in enumerate(ddf_yyyy.keys()):
        _f_extract = extraction(ddf_yyyy[year], ddf_metadata_yyyy[year])[1:] # get features.
        _f_dict = dict() # create dictionary map: set of features to set of labels (which are indexed by year).

        for f in _f_extract:
            desc = ddf_yyyy[year][f].iloc[0] # get description.
            _f_dict[desc] = f
        
        total_features[i] = _f_dict # map from description to label

    # get all map keys.
    total_keys = [f.keys() for f in total_features]
    if include_geography:
        univ_desc = ["Geography"] + list(set(total_keys[0]).intersection(*total_keys[1:]))
    else:
        univ_desc = list(set(total_keys[0]).intersection(*total_keys[1:]))

    # define the universal map: features -> (year -> label)
    univ_map = dict()
    for desc in univ_desc:
        univ_map[desc] = dict() # map: year -> label.
        
        for year in ddf_yyyy.keys():
            _df = ddf_yyyy[year]
            _df_metadata = ddf_metadata_yyyy[year]
            univ_map[desc][year] = _get_key(_df, desc)

    return univ_map

def feature_index(ddf, ddf_metadata, include_geography=False):
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

    if include_geography:
        univ_desc = ["Geography"] + list(set(total_keys[0]).intersection(*total_keys[1:]))
    else:
        univ_desc = list(set(total_keys[0]).intersection(*total_keys[1:]))

    # define a universal map.
    univ_map = dict()

    for desc in univ_desc:
        univ_map[desc] = [None for _ in range(n)]
        for i in range(n):
            _df = ddf[i]
            _df_metadata = ddf_metadata[i]

            univ_map[desc][i] = _get_key(_df, desc)

    return univ_map

def label_from_feature_index(yyyy, f_index):
    """
    Takes year and feature index, returns list of labels.
    """
    return [f_index[key][yyyy] for key in f_index.keys()]

def state_and_county(geography):
    """
    Takes geography data from socio-economic factors,
    returns state and county name (without " county")
    """
    county, state = geography.split(", ") # separate county from state
    state_in = state_in_dict[state.lower()] # convert state to initials
    county = ' '.join(county.split()[:-1]) # remove the word "county"

    return state_in, county, state

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

def generate_sample(ddf_yyyy, ddf_yyyy_meta, f_index, df_nflis, substanceNamesDict, df_geo, debug=False):
    """
    Takes relevant socio-economic dataframes, metadataframes,
    applicable (universal) socio-economic features and drug use data.
    Returns an appropriate sample matrix with explanatory variables followed by response variables.
    """

    # determine dimension.
    m = 0 # sample size.
    
    df_sample_sizes = [ddf_yyyy[yyyy].shape[0] - 1 for yyyy in ddf_yyyy] # -1 so we exclude row with labels.
    m = sum(df_sample_sizes)

    geo_n = 2 # positional features: longitude and latitude.
    socio_n = len(f_index) # non-positional, socio-economic features.
    drug_n = len(substanceNamesDict) # drug type features.
    n = 1 + geo_n + socio_n + drug_n # feature dimension equals sum of above three features, including year.

    sample = np.zeros((m, n)) # instantiate mxn sample matrix.
    
    # fill sample.
    for l, year in enumerate(ddf_yyyy):
        df = ddf_yyyy[year] # dataframe for this year.
        df_msum = sum(df_sample_sizes[:l]) # samples processed by previous dataframes.
        df_m = df_sample_sizes[l] # current dataframe's sample size.
        labels = label_from_feature_index(year, f_index) # get appropriate labels for the year.
        sub_df = df[labels] # sub dataframe comprised of the appropriate labels.

        for i in range(df_m):

            i_sample = i + df_msum # index in the sample matrix
            i_df = i + 1 # index in the dataframe
            
            # append year data.
            sample[i_sample][0] = year

            # append geographic data.
            try:
                state_in, county, state = state_and_county(df["GEO.display-label"].iloc[i_df]) # retrieve state initials, county.
            except:
                print(l)
                print(i)
                print(i_df)
                print(df["GEO.display-label"].iloc[i_df])
                raise
            try:
                lat, lon = locate(state_in, county, df_geo) # get lattitude, longitude
            except TypeError:
                lat = None
                lon = None
            sample[i_sample][1] = lat
            sample[i_sample][2] = lon

            # append socio-economic data.
            try:
                sample[i_sample][3:3+socio_n] = np.array(df[labels].iloc[i_df])
            except ValueError: # dealing with '(X)'.
                #sample[i_sample][3:3+socio_n] = np.nan #just set to nan, we'll remove later.
                _tempdat = df[labels].iloc[i_df]
                _tempdat[list(filter(lambda i:_tempdat[i]  == '(X)', range(len(_tempdat))))] = 0 # replace all '(X)' with zero.
                sample[i_sample][3:3+socio_n] = _tempdat
                #print(year)
                #print(i_df)
                #print(df[labels].iloc[i_df])
                #raise ValueError

            # append drug data.
            drug_vec = drug_vector(year, state_in, county, df_nflis, substanceNamesDict)
            sample[i_sample][3+socio_n:] = drug_vec
            if debug:
                print("processed {}, {}".format(county, state_in))

    return sample

# index to descriptor.
def find_nonzero(mat):
    """
    Takes a matrix with (possibly) zero rows, and returns a (possibly) smaller matrix without those zero rows.
    """

    m, n = mat.shape # get dimensions.
    zero_indices = list()
    nonzero_indices = list()

    for i in range(m):
        if sum(np.abs(mat[i])) != 0:
            nonzero_indices.append(i)
        else:
            zero_indices.append(i)

    return nonzero_indices, zero_indices

def keep_rows(mat, indices):
    """
    Takes a matrix and row indices to keep, and returns a matrix with the kept rows.
    """
    return mat[indices]

def keep_cols(mat, indices):
    """
    Takes a matrix and column indices to keep, and returns a matrix with kept columns.
    """
    return keep_rows(mat.T, indices).T

# find zero rows of a matrix and return matrix without these rows.
def kill_zeros(mat):
    """
    Takes a matrix with (possibly) zero rows, and returns a (possibly) smaller matrix without those zero rows.
    """

    nonzero_indices = find_nonzero(mat)[0]

    # create smaller matrix.
    return np.array([mat[i] for i in nonzero_indices])

def identify_sample_points(indices, ddf_yyyy):
    """
    Find the sample points corresponding to the indices.
    """
    m_indices = len(indices)
    df_sample_sizes = [ddf_yyyy[yyyy].shape[0] - 1 for yyyy in ddf_yyyy] # -1 so we exclude row with labels.
    indexed_sample = np.empty((m_indices, 3), dtype=object) # identified by year, county and state.
    years = sorted(ddf_yyyy.keys())

    curr_df_in = 0
    curr_df = ddf_yyyy[years[0]]
    curr_df_size = df_sample_sizes[0]

    for i in range(m_indices):
        i_ddf = indices[i] # index corresponding to the superdataframe.
        while i_ddf >= curr_df_size:
            curr_df_in += 1
            curr_df = ddf_yyyy[years[curr_df_in]]
            curr_df_size += df_sample_sizes[curr_df_in]

        i_df = i_ddf - sum(df_sample_sizes[:curr_df_in]) + 1 # index within current dataframe (accounted for label row).
        indexed_sample[i][0] = years[curr_df_in] # year of sample point.

        # get state and county
        state_in, county, state = state_and_county(curr_df["GEO.display-label"].iloc[i_df])
        indexed_sample[i][1] = state_in
        indexed_sample[i][2] = county

    return indexed_sample

# standardization.
def standardize(arr):
    """
    Takes a numpy array and standardizes it (with respect to columns).
    """

    arrT = np.copy(arr.T) # transpose input array.
    arrT = kill_zeros(arrT)

    for i, row in enumerate(arrT):
        arrT[i] = (row - np.nanmean(row)) / np.nanstd(row)

    return arrT.T # return un-transposed array.

def threshold_pass(vec, t):
    # takes a vector and returns zero for entries whose absolute value is below threshold t.
    return np.array([v if np.abs(v) >= t else 0 for v in vec])

def pc_explain(i, pc_svd, threshold, features):
    """
    Takes index argument, an argument of principal components, and a threshold argument.
    Returns a description of features of the i-th principal component
    whose coefficient is above the threshold.
    """

    pc_i = pc_svd.T[i]
    pc_i = threshold_pass(pc_i, threshold)
    return list(filter(lambda x: x[1] != 0, [(features[i], pc_i[i]) for i in range(len(pc_i))]))

from matplotlib import pyplot as plt
def compare_plots(lab1a, lab1b, lab2a, lab2b, df1, df2, title1, title2, alpha=.1):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(df1[lab1a], df1[lab2a], alpha=alpha)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.scatter(df2[lab1b], df2[lab2b], alpha=alpha)
    plt.title(title2)
    plt.show()

if __name__ == "__main__":

    # run this code.

    pass
