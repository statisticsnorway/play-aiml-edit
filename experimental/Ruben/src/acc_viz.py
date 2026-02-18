import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class AccumulationErrorVisualizer:
    def __init__(self, df, cfg):
        self.df = df
        self.cfg = cfg
        self.original_col = f"{cfg.omsetning}_original"

        self.month_names = ['Januar', 'Februar', 'Mars', 'April', 'Mai', 'Juni', 
                       'Juli', 'August', 'September', 'Oktober', 'November', 'Desember']
        
    def create_error_report(self, save_path=None, figsize=(16, 10)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Oppsummering av feil', fontsize=16, y=0.995)
        
        self._plot_error_type_distribution(axes[0, 0])
        
        self._plot_error_duration_distribution(axes[0, 1])
        
        self._plot_start_month_distribution(axes[1, 0])
        
        self._plot_all_affected_months_distribution(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_error_type_distribution(self, ax):
        error_data = self.df[self.df['error_type'].notna()]
        error_counts = error_data['error_type'].value_counts()
        
        colors = sns.color_palette("husl", len(error_counts))
        ax.bar(range(len(error_counts)), error_counts.values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(error_counts)))
        ax.set_xticklabels(error_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Records')
        ax.set_title('Error Type Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(error_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
    
    def _plot_error_duration_distribution(self, ax):
        error_orgs = self.df[self.df[self.cfg.error_col] == 1][self.cfg.bedrift].unique()
        
        durations = []
        for org in error_orgs:
            org_data = self.df[self.df[self.cfg.bedrift] == org].sort_values(self.cfg.dato)
            org_errors = org_data[org_data['error_type'].notna()]
            if len(org_errors) > 0:
                durations.append(len(org_errors))
        
        if len(durations) == 0:
            ax.text(0.5, 0.5, 'No errors found', transform=ax.transAxes, 
                    ha='center', va='center')
            ax.axis('off')
            return
        
        bins = range(1, max(durations) + 2)
        ax.hist(durations, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Periode i antall måneder', fontsize=10)
        ax.set_ylabel('Antall organisasjoner', fontsize=10)
        ax.set_title('Periode for feil', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        ax.axvline(mean_duration, color='red', linestyle='--', linewidth=2, 
                   label=f'Gjennomsnitt: {mean_duration:.1f}')
        ax.axvline(median_duration, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_duration:.1f}')
        ax.legend(fontsize=9)
    
    def _plot_start_month_distribution(self, ax):
        error_orgs = self.df[self.df[self.cfg.error_col] == 1][self.cfg.bedrift].unique()
        
        start_months = []
        for org in error_orgs:
            org_data = self.df[self.df[self.cfg.bedrift] == org].sort_values(self.cfg.dato)
            org_errors = org_data[org_data['error_type'].notna()]
            
            if len(org_errors) > 0:
                first_error_month = org_errors.iloc[0][self.cfg.dato].month
                start_months.append(first_error_month)
        
        if len(start_months) == 0:
            ax.text(0.5, 0.5, 'No errors found', transform=ax.transAxes, 
                    ha='center', va='center')
            ax.axis('off')
            return
        
        month_counts = pd.Series(start_months).value_counts().sort_index()
        
        month_counts = month_counts.reindex(range(1, 13), fill_value=0)
        
        colors = "steelblue"
        
        ax.bar(range(12), month_counts.values, color=colors, alpha=0.8)
        ax.set_xticks(range(12))
        ax.set_xticklabels(self.month_names, rotation=45, ha='right')
        ax.set_ylabel('Antall organisasjoner', fontsize=10)
        ax.set_title('Startmåned for feil', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        if month_counts.iloc[0] > 0:
            jan_pct = (month_counts.iloc[0] / len(start_months) * 100)
            ax.text(0, month_counts.iloc[0], f'{jan_pct:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)
        
        ax.text(0.98, 0.98, f'Organisasjoner med feil: {len(start_months)}', 
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    def _plot_all_affected_months_distribution(self, ax):
        error_records = self.df[self.df['error_type'].notna()]
        
        if len(error_records) == 0:
            ax.text(0.5, 0.5, 'No errors found', transform=ax.transAxes, 
                    ha='center', va='center')
            ax.axis('off')
            return
        
        error_months = error_records[self.cfg.dato].dt.month
        month_counts = error_months.value_counts().sort_index()
        
        month_counts = month_counts.reindex(range(1, 13), fill_value=0)
        
        ax.bar(range(12), month_counts.values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(12))
        ax.set_xticklabels(self.month_names, rotation=45, ha='right')
        ax.set_ylabel('Antall feil', fontsize=10)
        ax.set_title('Antall måneder som har feil', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        ax.text(0.98, 0.98, f'Totalt antall feil: {len(error_records):,}', 
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        max_idx = month_counts.idxmax() - 1
        max_val = month_counts.max()
        ax.text(max_idx, max_val, f'{max_val:,}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)

if __name__ == "__main__":
    from create_accumulation_error import AccumulationErrors
    from load_data import get_all_data
    from config import Config
    import os
    
    if os.getcwd() == "/home/onyxia/work/play-aiml-edit":
        os.chdir(path="experimental/Ruben/src/")

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
    
    print("\nLager visualiseringer")
    viz = AccumulationErrorVisualizer(df=df, cfg=Config)
    viz.create_error_report(save_path="figurer/accumulation_error.png")