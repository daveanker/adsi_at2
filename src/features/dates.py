def convert_to_date(df, cols:list):
    import pandas as pd
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df