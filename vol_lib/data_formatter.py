import pandas as pd
from typing import Dict
import numpy as np

def format_options_data_to_dict(df: pd.DataFrame, 
                               strike_col: str = 'strike',
                               iv_col: str = 'ImpliedVolatility', 
                               tte_col: str = 'TimeToExpiry',
                               forward_col: str = 'Forward',
                               type_col: str = 'Type') -> Dict[float, pd.DataFrame]:

    separated_types_dict = {}
    types = ['call', 'put']

    result_dict = {}
    
    unique_ttes = df[tte_col].unique()

    for t in types:

        result_dict[t] = {}
        type_filtered = df[df['Type'] == t].copy()
        
        type_df = pd.DataFrame({
            'Strike': type_filtered[strike_col],
            'ImpliedVolatility': type_filtered[iv_col],
            'Forward': type_filtered[forward_col],
            'TimeToExpiry': type_filtered[tte_col]

        })
        if len(type_df) > 0:
            separated_types_dict[t] = type_df

        for tte in unique_ttes:
            tte_data = separated_types_dict[t][separated_types_dict[t][tte_col] == tte].copy()
            
            formatted_df = pd.DataFrame({
                'Strike': tte_data[strike_col],
                'ImpliedVolatility': tte_data[iv_col],
                'Forward': tte_data[forward_col]
            })
            
            # Remove any rows with missing or invalid data
            formatted_df = formatted_df.dropna()
            formatted_df = formatted_df[formatted_df['ImpliedVolatility'] > 1e-5]  # Remove zero/negative IVs
            
            formatted_df = formatted_df.sort_values('Strike').reset_index(drop=True)
            
            if len(formatted_df) > 0:
                result_dict[t][float(tte)] = formatted_df


    # for tte in unique_ttes:
    #     tte_data = df[df[tte_col] == tte].copy()
        
    #     formatted_df = pd.DataFrame({
    #         'Strike': tte_data[strike_col],
    #         'ImpliedVolatility': tte_data[iv_col],
    #         'Forward': tte_data[forward_col]
    #     })
        
    #     # Remove any rows with missing or invalid data
    #     formatted_df = formatted_df.dropna()
    #     formatted_df = formatted_df[formatted_df['ImpliedVolatility'] > 1e-5]  # Remove zero/negative IVs
        
    #     formatted_df = formatted_df.sort_values('Strike').reset_index(drop=True)
        
    #     if len(formatted_df) > 0:
    #         result_dict[float(tte)] = formatted_df
        
    return result_dict