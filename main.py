import pandas as pd
import quantstats as qs
import os
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm 

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'  
font_properties = fm.FontProperties(fname=font_path)


matplotlib.rcParams['font.family'] = font_properties.get_name()

class MagicFormulaBacktest:
    def __init__(self, empresas_path: str, ibov_path: str):
        """
        Initialize the backtest with the paths to the data files.
        
        :param empresas_path: Path to the CSV file containing company data.
        :param ibov_path: Path to the CSV file containing IBOV data.
        """
        self.empresas_path = empresas_path
        self.ibov_path = ibov_path
        self.dados_empresas = None
        self.rentabilidades_carteiras = None
    
    def load_data(self):
        """
        Load and preprocess company and IBOV data.
        """
        if not os.path.exists(self.empresas_path) or not os.path.exists(self.ibov_path):
            raise FileNotFoundError("One or both data files not found.")
        
        self.dados_empresas = pd.read_csv(self.empresas_path)
        self._process_empresas_data()
        self._process_ibov_data()
    
    def _process_empresas_data(self):
        """
        Process company data to calculate rankings and returns.
        """
        self.dados_empresas["retorno_mensal"] = self.dados_empresas.groupby("ticker")["preco_fechamento_ajustado"].pct_change()
        self.dados_empresas["retorno_mensal"] = self.dados_empresas.groupby("ticker")["retorno_mensal"].shift(-1)
        
        self.dados_empresas = self.dados_empresas[self.dados_empresas["volume_negociado"] > 1_000_000]
        
        self.dados_empresas["ranking_ebit_ev"] = self.dados_empresas.groupby("data")["ebit_ev"].rank(ascending=False)
        self.dados_empresas["ranking_roic"] = self.dados_empresas.groupby("data")["roic"].rank(ascending=False)
        
        self.dados_empresas["ranking_final"] = self.dados_empresas["ranking_ebit_ev"] + self.dados_empresas["ranking_roic"]
        self.dados_empresas["ranking_final"] = self.dados_empresas.groupby("data")["ranking_final"].rank()
        
        self.dados_empresas = self.dados_empresas[self.dados_empresas["ranking_final"] <= 10]
    
    def _process_ibov_data(self):
        """
        Process IBOV data to calculate accumulated returns.
        """
        ibov_data = pd.read_csv(self.ibov_path)
        retornos_ibov = ibov_data["fechamento"].pct_change().dropna()
        retorno_acum_ibov = (1 + retornos_ibov).cumprod() - 1
        
        self.rentabilidades_carteiras = self.dados_empresas.groupby("data")["retorno_mensal"].mean().to_frame()
        self.rentabilidades_carteiras["magic_formula"] = (self.rentabilidades_carteiras["retorno_mensal"] + 1).cumprod() - 1
        self.rentabilidades_carteiras = self.rentabilidades_carteiras.shift(1).dropna()
        self.rentabilidades_carteiras["ibovespa"] = retorno_acum_ibov.values
        self.rentabilidades_carteiras.drop("retorno_mensal", axis=1, inplace=True)
    
    def analyze(self):
        """
        Generate plots and analyze the backtest results.
        """
        qs.extend_pandas()
        
        self.rentabilidades_carteiras.index = pd.to_datetime(self.rentabilidades_carteiras.index)
        
        self._plot_results()


    def _plot_results(self):
        """
        Plot the monthly heatmaps and cumulative return comparison.
        """
        self.rentabilidades_carteiras["magic_formula"].plot_monthly_heatmap()
        self.rentabilidades_carteiras["ibovespa"].plot_monthly_heatmap()
        
        plt.figure()
        self.rentabilidades_carteiras.plot(figsize=(10, 6), title="Comparativo de Rentabilidade - Magic Formula vs IBOV")
        plt.savefig("comparativo_rentabilidade.png")  
        plt.close()

    

def main():
    """
    Main execution function to run the backtest.
    """
    empresas_path = "dados_empresas.csv"
    ibov_path = "ibov.csv"
    
    try:
        backtest = MagicFormulaBacktest(empresas_path, ibov_path)
        backtest.load_data()
        backtest.analyze()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    
    
if __name__ == "__main__":
    main()
