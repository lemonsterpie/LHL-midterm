import pandas as pd
import numpy as np
import json
import os 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler

current_dir = os.getcwd()
main_dir = os.path.abspath(os.path.join(current_dir, '..'))
data_dir = os.path.join(main_dir, 'data')
csv_dir = os.path.join(main_dir, 'csv_data')


def encode_tags(df, min_occurrences):
    """Encodes tags into binary columns, filtering out low-frequency ones.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'tags' column.
        min_occurrences (int): Minimum times a tag must appear to be included.

    Returns:
        pandas.DataFrame: Modified DataFrame with encoded tags.
    """

    if "tags" not in df.columns:
        print("Warning: 'tags' column not found in DataFrame.")
        return df

    # Flatten tag lists and count occurrences
    tag_counts = df["tags"].explode().value_counts()

    # Filter tags based on frequency threshold
    filtered_tags = tag_counts[tag_counts >= min_occurrences].index.tolist()

    # Efficient One-Hot Encoding using pd.concat()
    tag_df = pd.DataFrame({tag: df["tags"].apply(lambda x: 1 if isinstance(x, list) and tag in x else 0) for tag in filtered_tags})

    # Merge encoded tags back into the original DataFrame
    df = pd.concat([df, tag_df], axis=1)

    return df

class DataLoader(BaseEstimator, TransformerMixin):

    """
    sklearn class that loads json data from a predefined data directory and transforms it to a Pandas DataFrame

    Args:
        BaseEstimator: provides default sklearn implentations
        TransformerMixin: allows fit_transform 

    Returns:
        pandas.DataFrame: list of real estate listings as DataFrame 
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        # Load data from directory (X overrides data_dir if given)
        data_dir = X if isinstance(X, str) else self.data_dir
        if data_dir is None:
            raise ValueError("No data directory provided to load data.")

        data_list = []

        # Loop over all JSON files in the directory
        for file in os.listdir(data_dir):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(data_dir, file)
            with open(file_path, "r") as f:
                raw_data = json.load(f)

                listings = raw_data.get("data", {}).get("results", [])
                if not isinstance(listings, list):
                    print(f"Skipping malformed file: {file}")
                    continue

                for listing in listings:
                    sale_record = {
                    "property_id": listing.get("property_id", "Unknown"),
                    "permalink": listing.get("permalink", "Unknown"),
                    "status": listing.get("status", "Unknown"),
                    "year_built": listing.get("description", {}).get("year_built", None),
                    "garage": listing.get("description", {}).get("garage", None),
                    "stories": listing.get("description", {}).get("stories", None),
                    "beds": listing.get("description", {}).get("beds", None),
                    "baths_1qtr": listing.get("description", {}).get("baths_1qtr", None),
                    "baths_3qtr": listing.get("description", {}).get("baths_3qtr", None),
                    "baths_half": listing.get("description", {}).get("baths_half", None),
                    "baths_full": listing.get("description", {}).get("baths_full", None),
                    "baths": listing.get("description", {}).get("baths", "Unknown"),
                    "type": listing.get("description", {}).get("type", "Unknown"),
                    "sub_type": listing.get("description", {}).get("sub_type", "Unknown"),
                    "lot_sqft": listing.get("description", {}).get("lot_sqft", None),
                    "sqft": listing.get("description", {}).get("sqft", None),
                    "sold_price": listing.get("description", {}).get("sold_price", None),
                    "sold_date": pd.to_datetime(listing.get("description", {}).get("sold_date", None), errors='coerce'),
                    "list_price": listing.get("list_price", None),
                    "last_update_date": pd.to_datetime(listing.get("last_update_date"), errors='coerce'),
                    "city": listing.get("location", {}).get("address", {}).get("city", "Unknown"),
                    "state": listing.get("location", {}).get("address", {}).get("state", "Unknown"),
                    "postal_code": listing.get("location", {}).get("address", {}).get("postal_code", "Unknown"),
                    "street_view_url": listing.get("location", {}).get("street_view_url", "Unknown"),
                    "tags": listing.get("tags", [])
                }
                    data_list.append(sale_record)

        df = pd.DataFrame(data_list)
        return df

def data_cleaning(df):

    """
    Takes in a Pandas DataFrame and processes it according to real estate listings requirements 

    Args:
        pandas.DataFrame: raw list of real estate listings as DataFrame 

    Returns:
        pandas.DataFrame: proccessed list of real estate listings as DataFrame 
    """    

    # Dropping druplicate rows and validating column dtypes
    df = df.drop_duplicates(subset='property_id')
    df.loc[:, 'year_built'] = df['year_built'].astype('Int64') 
    df.loc[:, 'garage'] = df['garage'].astype('Int64')
    df.loc[:, 'stories'] = df['stories'].astype('Int64')
    df.loc[:, 'beds'] = df['beds'].astype('Int64')
    df.loc[:, 'baths'] = df['baths'].astype('Int64')

    # Dropping low relevance columns and rows where target is null  
    df = df.drop(columns=['baths_1qtr','baths_3qtr', 'sub_type', 'baths_half'])
    df = df.drop(columns='baths_full')
    df = df.drop(columns='list_price')
    df = df[~df['sold_price'].isnull()]

    # Filling assuming null values are 0 or global mode, filling accordingly 
    df['garage'] = df['garage'].fillna(value=0) 
    df['lot_sqft'] = df['lot_sqft'].fillna(value=0)
    df['sqft'] = df['sqft'].fillna(value=0)
    df['beds'] = df['beds'].fillna(value=0)
    df['baths'] = df['baths'].fillna(value=0)
    df['tags'] = df['tags'].fillna(value='[]')
    df['stories'] = df['stories'].fillna(value=0)
    df['type'] = df['type'].fillna(df['type'].mode()[0])
    df['city'] = df['city'].fillna(df['type'].mode()[0])

    # Defining conditions for null real estate types 
    condition_sf = (df['type'] == 'single_family') & (df['year_built'].isna())
    condition_land = (df['type'] == 'land') & (df['year_built'].isna())
    condition_condo = (df['type'] == 'condo') & (df['year_built'].isna())
    condition_mobile = (df['type'] == 'mobile') & (df['year_built'].isna())

    # Group by type and city, then get mode of rows of single family type
    modes = (df[df['type'] == 'single_family'].groupby(['type', 'city'])['year_built'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA))

    # Define a function to get the mode from the groupby
    def get_mode(row):
        try:
            return modes.loc[(row['type'], row['city'])]
        except KeyError:
            return row['year_built']  # fallback if no mode available


    df.loc[condition_sf, 'year_built'] = df[condition_sf].apply(get_mode, axis=1)
    df.loc[condition_land, 'year_built'] = 1956
    df.loc[condition_condo, 'year_built'] = 1985
    df = df[df['type'] != 'other']
    df.loc[condition_mobile, 'year_built'] = 1988

    df = df[df['year_built'].isnull() == False]

    df = df.drop(columns=['permalink', 'postal_code', 'street_view_url', 'sold_date', 'last_update_date', 'status', 'property_id'])

    # Feature Engineering
    df['total_sqft'] = df['lot_sqft'] + df['sqft']
    df['building_ratio'] = df['sqft'] / df['lot_sqft'] # higher ratio = more building area(urban), 0 = all lot no building
    df['building_ratio'] = df['building_ratio'].replace([np.inf, -np.inf], np.nan) # fixing cases where ratio is inf (0 sqft and positive lot sqft) 
    df['building_ratio'] = df['building_ratio'].fillna(value=df['building_ratio'].mean())
    
    df.to_csv(os.path.join(csv_dir, "cleaned_real_estate_listings.csv"), index=False)

    return df

def data_encoding(df):

    """
    Takes in a Pandas DataFrame and encodes the tags, city, and state columns 

    Args:
        pandas.DataFrame: proccessed list of real estate listings as DataFrame 

    Returns:
        pandas.DataFrame: encoded list of real estate listings as DataFrame 
    """      
    
    from functions_variables import encode_tags
    
    assert df is not None, "Input df is None before encoding tags"
    df = encode_tags(df, min_occurrences=100)

    from category_encoders import TargetEncoder
    from sklearn.model_selection import KFold

    X = df.drop(columns='sold_price')
    y = df['sold_price']

    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    encoder = TargetEncoder(cols=['city', 'state'], smoothing=3)
    encoded_data = encoder.fit_transform(X_train[['city', 'state']], y_train)

    df.loc[X_train.index, 'encoded_city'] = encoded_data['city'].round(2)
    df.loc[X_train.index, 'encoded_state'] = encoded_data['state'].round(2)

    city_mappings = df[['city', 'encoded_city']].dropna().drop_duplicates().set_index('city')['encoded_city'].to_dict()
    state_mappings = df[['state', 'encoded_state']].dropna().drop_duplicates().set_index('state')['encoded_state'].to_dict()

    df['encoded_city'] = df['encoded_city'].fillna(df['city'].map(city_mappings))
    df['encoded_state'] = df['encoded_state'].fillna(df['state'].map(state_mappings))

    overall_mean_price = df["sold_price"].mean()
    df["encoded_city"] = df["encoded_city"].fillna(overall_mean_price).astype('float')
    df["encoded_state"] = df["encoded_state"].fillna(overall_mean_price).astype('float')

    type_mapping = {
        'mobile': 1,
        'apartment': 2,
        'condo': 3,
        'condos': 4,
        'condo_townhome_rowhome_coop': 5,
        'townhomes': 6,
        'single_family': 7,
        'duplex_triplex': 8,
        'multi_family': 9,
        'land': 10
    }

    df['type'] = df['type'].map(type_mapping).astype('Int64')    

    df = df.drop(columns=['city', 'state', 'tags'])

    df.to_csv(os.path.join(csv_dir, "encoded_real_estate_listings.csv"), index=False)

    return df

class ColumnScaler(BaseEstimator, TransformerMixin):
    
    """
    sklearn class that uses StandardScalar on select, predetermined columns 

    Args:
        BaseEstimator: provides default sklearn implentations
        TransformerMixin: allows fit_transform 
    """
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.columns] = self.scaler.transform(X_copy[self.columns])
        return X_copy


