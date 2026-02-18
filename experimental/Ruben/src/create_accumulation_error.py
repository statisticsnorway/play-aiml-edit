import pandas as pd
import numpy as np

from load_data import get_all_data
from config import Config

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
        result_df['error_year'] = None
        
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
            result_df = self._apply_errors_to_org(result_df, org)
        
        result_df[self.cfg.error_col] = (result_df[self.cfg.omsetning] != result_df[self.original_value]).astype(int)
        
        return result_df
    
    def _apply_errors_to_org(self, df, org):
        """Apply one error per year to an organization"""
        org_mask = df[self.cfg.bedrift] == org
        org_data = df[org_mask].sort_values(self.cfg.dato).copy()
        
        if len(org_data) < 3:
            return df
        
        available_years = org_data[self.cfg.dato].dt.year.unique()
        
        if self.cfg.multiple_error_years:
            years_with_errors = []
            for year in available_years: 
                if np.random.random() < self.cfg.error_year_probability:
                    years_with_errors.append(year)
            
            # TODO: muligens endre denne slik at mulighet for ikke feil i år
            if len(years_with_errors) == 0:
                years_with_errors = [np.random.choice(available_years)]
        else:
            years_with_errors = [np.random.choice(available_years)]
        
        for year in years_with_errors:
            df = self._apply_error_to_year(df, org, org_data, year)
        
        return df
    
    def _apply_error_to_year(self, df, org, org_data, year):
        year_mask = org_data[self.cfg.dato].dt.year == year
        year_data = org_data[year_mask].copy()
        
        if len(year_data) < 3:
            return df
        
        start_idx = self._get_start_idx_in_year(year_data, org_data)
        
        months_left_in_year = len(year_data) - (start_idx - year_data.index.get_loc(year_data.index[0]))
        max_duration = min(12, months_left_in_year)
        
        if max_duration < 3:
            return df
        
        duration = np.random.randint(3, max_duration + 1)

        start_pos_in_full = org_data.index.get_loc(year_data.index[start_idx])
        end_pos_in_full = start_pos_in_full + duration
        
        error_indices = org_data.index[start_pos_in_full:end_pos_in_full]
        
        error_type = np.random.choice(self.type_of_errors)
        
        original_values = df.loc[error_indices, self.original_value].values
        
        if error_type == 'cumulative_sum':
            modified_values = self._cumulative(original_values)
        elif error_type == 'cascading':
            modified_values = self._cascading(original_values)
        elif error_type == 'random':
            modified_values = self._random(original_values, error_probability=0.5)
        
        df.loc[error_indices, self.cfg.omsetning] = modified_values
        df.loc[error_indices, 'error_type'] = error_type
        df.loc[error_indices, 'error_year'] = year
        
        return df
    
    def _get_start_idx_in_year(self, year_data, full_org_data):
        """Get start index within a specific year's data"""
        use_preferred_month = np.random.random() < self.cfg.start_month_prob
        
        if use_preferred_month and self.cfg.start_month > 0:
            months = year_data[self.cfg.dato].dt.month.values
            matching_positions = np.where(months == self.cfg.start_month)[0]
            
            valid_positions = matching_positions[matching_positions <= len(year_data) - 3]
            
            if len(valid_positions) > 0:
                return valid_positions[0]
        
        max_start = max(0, len(year_data) - 3)
        return np.random.randint(0, max_start + 1) if max_start > 0 else 0
    
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
        total_error_prct=Config.bedrifter_med_feil
    )
    
    df = make_errors.create_accumulation_errors(df=hent_data)