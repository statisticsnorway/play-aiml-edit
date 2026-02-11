import pandas as pd
import numpy as np

from load_data import get_all_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]

class AccumulationErrors:
    def __init__(self, cfg, years, type_of_errors, total_error_prct):
        self.cfg = cfg
        self.years = years
        self.type_of_errors = type_of_errors
        self.total_error_prct = total_error_prct
        
    def create_accumulation_errors(self, df):
        result_df = df.copy()
        
        self.original_value = f"{self.cfg.omsetning}_original"
        result_df[self.original_value] = result_df[self.cfg.omsetning]
        result_df['error_type'] = None
        
        result_df[self.cfg.dato] = pd.to_datetime(result_df[self.cfg.dato])
        
        result_df = result_df[result_df[self.cfg.dato].dt.year.isin(self.years)]
        
        unique_orgs = result_df[self.cfg.bedrift].unique()
        n_orgs_with_errors = int(len(unique_orgs) * self.total_error_prct)
        
        orgs_with_errors = np.random.choice(
            unique_orgs, 
            size=n_orgs_with_errors, 
            replace=False
        )
        
        for org in orgs_with_errors:
            result_df = self._apply_error_to_org(result_df, org)
        
        result_df[self.cfg.error_col] = (result_df[self.cfg.omsetning] != result_df[self.original_value]).astype(int)
        
        return result_df
    
    def _get_start_idx(self, org_data):
        use_preferred_month = np.random.random() < self.cfg.start_month_prob
        
        if use_preferred_month and self.cfg.start_month > 0:
            months = org_data[self.cfg.dato].dt.month.values
            
            matching_positions = np.where(months == self.cfg.start_month)[0]
            
            valid_positions = matching_positions[matching_positions <= len(org_data) - 3]
            
            if len(valid_positions) > 0:
                selected_pos = np.random.choice(valid_positions)                
                
                return selected_pos
        
        random_pos = np.random.randint(0, len(org_data) - 2)

        return random_pos
    
    def _apply_error_to_org(self, df, org):
        org_mask = df[self.cfg.bedrift] == org
        org_data = df[org_mask].sort_values(self.cfg.dato).copy()
        
        if len(org_data) < 3:
            return df
        
        error_type = np.random.choice(self.type_of_errors)
        
        start_idx = self._get_start_idx(org_data)
        
        # Hvor mange måneder akkumuleringsfeil vil vare
        max_duration = min(12, len(org_data) - start_idx)
        duration = np.random.randint(3, max_duration + 1)
        end_idx = start_idx + duration
        
        error_indices = org_data.index[start_idx:end_idx]
        
        if error_type == 'cumulative_sum':
            modified_values = self._cumulative(
                org_data.iloc[start_idx:end_idx][self.original_value].values
            )
        elif error_type == 'cascading':
            modified_values = self._cascading(
                org_data.iloc[start_idx:end_idx][self.original_value].values
            )
        elif error_type == 'random':
            modified_values = self._random(
                org_data.iloc[start_idx:end_idx][self.original_value].values,
                error_probability=0.3
            )
        
        df.loc[error_indices, self.cfg.omsetning] = modified_values
        df.loc[error_indices, 'error_type'] = error_type
        
        return df

    def _cumulative(self, values):
        """Cumulative sum error"""
        result = np.zeros(len(values))
        for i in range(len(values)):
            result[i] = np.sum(values[:i+1])
        return result
    
    def _cascading(self, values):
        """Cascading error"""
        result = np.zeros(len(values))
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = np.sum(result[:i]) + values[i]
        return result

    def _random(self, values, error_probability=0.3):
        """Random error"""
        result = np.zeros(len(values))
        result[0] = values[0]
        for i in range(1, len(values)):
            if np.random.random() < error_probability:
                result[i] = np.sum(values[:i+1])
            else:
                result[i] = values[i]
        return result


if __name__ == "__main__":
    print("Henter ut data fra VHI")
    hent_data = get_all_data(cfg=Config)
    
    print("Lager akkumuleringsfeil")
    make_errors = AccumulationErrors(
        cfg=Config,
        years=Config.years,
        type_of_errors=Config.acc_errors,
        total_error_prct=0.25 # må se nærmere på denne variabelen, introduserer rundt 1-5% med total_error_prct=0.05-0.30 
    )
    
    df = make_errors.create_accumulation_errors(df=hent_data)