import pandas as pd
import numpy as np

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

