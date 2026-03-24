"""
Streamlit App para Análisis de Dependencia en Datos de Pozo
===========================================================
Con regresión cuantil basada en cópulas
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, kendalltau, spearmanr, pearsonr, gaussian_kde
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Dependencia - Datos de Pozo",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .dependence-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 1rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class CopulaQuantileRegression:
    """
    Clase para implementar regresión cuantil basada en cópulas
    """
    
    def __init__(self):
        self.fitted_copulas = {}
        
    def empirical_cdf(self, x):
        """Calcula la CDF empírica"""
        return stats.rankdata(x) / (len(x) + 1)
    
    def empirical_inverse_cdf(self, y, original_data):
        """Calcula la inversa de la CDF empírica"""
        sorted_data = np.sort(original_data)
        return np.interp(y, np.linspace(0, 1, len(sorted_data)), sorted_data)
    
    def fit_gaussian_copula(self, u, v):
        """Ajusta una cópula Gaussiana"""
        # Transformar a normales
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        
        # Calcular correlación
        rho = np.corrcoef(z1, z2)[0, 1]
        
        return {'type': 'gaussian', 'rho': rho}
    
    def fit_clayton_copula(self, u, v):
        """Ajusta una cópula Clayton"""
        # Estimación de theta para Clayton
        tau = stats.kendalltau(u, v)[0]
        if tau > 0:
            theta = 2 * tau / (1 - tau)
        else:
            theta = 0.1
        return {'type': 'clayton', 'theta': theta}
    
    def fit_gumbel_copula(self, u, v):
        """Ajusta una cópula Gumbel"""
        tau = stats.kendalltau(u, v)[0]
        if tau > 0:
            theta = 1 / (1 - tau)
        else:
            theta = 1.1
        return {'type': 'gumbel', 'theta': theta}
    
    def fit_frank_copula(self, u, v):
        """Ajusta una cópula Frank"""
        tau = stats.kendalltau(u, v)[0]
        # Aproximación para Frank
        if tau > 0:
            theta = 5.0 * tau
        elif tau < 0:
            theta = -5.0 * abs(tau)
        else:
            theta = 0
        return {'type': 'frank', 'theta': theta}
    
    def select_best_copula(self, u, v):
        """Selecciona la mejor cópula basada en AIC"""
        copulas = {
            'gaussian': self.fit_gaussian_copula(u, v),
            'clayton': self.fit_clayton_copula(u, v),
            'gumbel': self.fit_gumbel_copula(u, v),
            'frank': self.fit_frank_copula(u, v)
        }
        
        # Calcular log-likelihood para cada cópula
        best_copula = None
        best_aic = np.inf
        
        for name, params in copulas.items():
            try:
                if name == 'gaussian':
                    # Calcular log-likelihood para Gaussiana
                    rho = params['rho']
                    if abs(rho) < 0.999:
                        log_lik = self._gaussian_log_likelihood(u, v, rho)
                    else:
                        continue
                elif name == 'clayton':
                    log_lik = self._clayton_log_likelihood(u, v, params['theta'])
                elif name == 'gumbel':
                    log_lik = self._gumbel_log_likelihood(u, v, params['theta'])
                elif name == 'frank':
                    log_lik = self._frank_log_likelihood(u, v, params['theta'])
                else:
                    continue
                
                n_params = 1
                aic = -2 * log_lik + 2 * n_params
                
                if aic < best_aic:
                    best_aic = aic
                    best_copula = params
                    
            except:
                continue
        
        if best_copula is None:
            best_copula = {'type': 'gaussian', 'rho': np.corrcoef(u, v)[0, 1]}
        
        return best_copula
    
    def _gaussian_copula_cdf(self, u, v, rho):
        """CDF de la cópula Gaussiana"""
        from scipy.stats import multivariate_normal
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        return multivariate_normal.cdf([z1, z2], mean=mean, cov=cov)
    
    def _gaussian_log_likelihood(self, u, v, rho):
        """Log-likelihood para cópula Gaussiana"""
        n = len(u)
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        log_lik = -n/2 * np.log(1 - rho**2) - np.sum((z1**2 + z2**2 - 2*rho*z1*z2) / (2*(1 - rho**2)))
        return log_lik
    
    def _clayton_copula_cdf(self, u, v, theta):
        """CDF de la cópula Clayton"""
        if theta == 0:
            return u * v
        return (u**(-theta) + v**(-theta) - 1)**(-1/theta)
    
    def _clayton_conditional_cdf(self, v, u, theta):
        """CDF condicional para Clayton: P(V <= v | U = u)"""
        if theta == 0:
            return v
        return u**(-theta-1) * (u**(-theta) + v**(-theta) - 1)**(-1/theta - 1)
    
    def _clayton_log_likelihood(self, u, v, theta):
        """Log-likelihood para cópula Clayton"""
        if theta <= 0:
            return -np.inf
        n = len(u)
        log_lik = np.sum(np.log((1 + theta) * (u * v)**(-theta-1) * 
                                (u**(-theta) + v**(-theta) - 1)**(-1/theta - 2)))
        return log_lik
    
    def _gumbel_copula_cdf(self, u, v, theta):
        """CDF de la cópula Gumbel"""
        if theta == 1:
            return u * v
        return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))
    
    def _gumbel_log_likelihood(self, u, v, theta):
        """Log-likelihood para cópula Gumbel"""
        if theta < 1:
            return -np.inf
        n = len(u)
        log_u = -np.log(u)
        log_v = -np.log(v)
        s = log_u**theta + log_v**theta
        log_lik = np.sum(np.log((s**(2/theta - 2) * (log_u * log_v)**(theta-1) *
                                (1 + (theta-1) * s**(-1/theta))) / (u * v)))
        return log_lik
    
    def _frank_copula_cdf(self, u, v, theta):
        """CDF de la cópula Frank"""
        if theta == 0:
            return u * v
        return -1/theta * np.log(1 + (np.exp(-theta*u) - 1) * (np.exp(-theta*v) - 1) / (np.exp(-theta) - 1))
    
    def _frank_log_likelihood(self, u, v, theta):
        """Log-likelihood para cópula Frank"""
        n = len(u)
        if theta == 0:
            return 0
        log_lik = np.sum(np.log(theta * (1 - np.exp(-theta)) * 
                                np.exp(-theta*(u+v)) / (1 - np.exp(-theta) - 
                                                       (1 - np.exp(-theta*u)) * (1 - np.exp(-theta*v)))**2))
        return log_lik
    
    def quantile_regression(self, x, y, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], 
                           copula_type='best', n_points=100):
        """
        Realiza regresión cuantil basada en cópulas
        
        Parameters:
        -----------
        x, y : array-like
            Datos originales
        quantiles : list
            Cuantiles a estimar (0-1)
        copula_type : str
            Tipo de cópula ('best', 'gaussian', 'clayton', 'gumbel', 'frank')
        n_points : int
            Número de puntos para la curva suavizada
        
        Returns:
        --------
        dict : Diccionario con los resultados de la regresión cuantil
        """
        # Transformar a rangos uniformes
        u = self.empirical_cdf(x)
        v = self.empirical_cdf(y)
        
        # Seleccionar la mejor cópula
        if copula_type == 'best':
            copula = self.select_best_copula(u, v)
        else:
            if copula_type == 'gaussian':
                copula = self.fit_gaussian_copula(u, v)
            elif copula_type == 'clayton':
                copula = self.fit_clayton_copula(u, v)
            elif copula_type == 'gumbel':
                copula = self.fit_gumbel_copula(u, v)
            elif copula_type == 'frank':
                copula = self.fit_frank_copula(u, v)
            else:
                copula = self.fit_gaussian_copula(u, v)
        
        # Crear grid de valores de x
        x_grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_points)
        u_grid = self.empirical_cdf(x_grid)
        
        # Calcular los cuantiles condicionales
        quantile_results = {}
        
        for q in quantiles:
            y_q = []
            for u_val in u_grid:
                # Encontrar v tal que C(u, v) = q
                v_q = self._find_conditional_quantile(u_val, q, copula)
                # Transformar de vuelta a la escala original
                y_val = self.empirical_inverse_cdf(v_q, y)
                y_q.append(y_val)
            
            quantile_results[q] = np.array(y_q)
        
        quantile_results['x_grid'] = x_grid
        quantile_results['copula'] = copula
        
        return quantile_results
    
    def _find_conditional_quantile(self, u, q, copula):
        """
        Encuentra v tal que P(V <= v | U = u) = q
        """
        if copula['type'] == 'gaussian':
            # Para cópula Gaussiana
            rho = copula['rho']
            z_u = norm.ppf(u)
            z_v = norm.ppf(q)
            # La distribución condicional es normal
            z_v_given_u = rho * z_u + np.sqrt(1 - rho**2) * z_v
            return norm.cdf(z_v_given_u)
        
        elif copula['type'] == 'clayton':
            # Para cópula Clayton
            theta = copula['theta']
            if theta == 0:
                return q
            # La función condicional tiene solución analítica
            v = ( (q**(-theta/(theta+1)) - 1) * u**(-theta) + 1 )**(-1/theta)
            return np.clip(v, 0, 1)
        
        elif copula['type'] == 'gumbel':
            # Para cópula Gumbel - solución numérica
            theta = copula['theta']
            if theta == 1:
                return q
            
            def objective(v):
                log_u = -np.log(u)
                log_v = -np.log(v)
                s = log_u**theta + log_v**theta
                cdf_val = np.exp(-s**(1/theta))
                return (cdf_val - q)**2
            
            # Búsqueda de raíz
            from scipy.optimize import brentq
            try:
                v = brentq(lambda v: self._gumbel_copula_cdf(u, v, theta) - q, 0.001, 0.999)
                return v
            except:
                return q
        
        elif copula['type'] == 'frank':
            # Para cópula Frank - solución numérica
            theta = copula['theta']
            if theta == 0:
                return q
            
            def objective(v):
                cdf_val = self._frank_copula_cdf(u, v, theta)
                return (cdf_val - q)**2
            
            from scipy.optimize import brentq
            try:
                v = brentq(lambda v: self._frank_copula_cdf(u, v, theta) - q, 0.001, 0.999)
                return v
            except:
                return q
        
        else:
            return q


class WellLogDependenceAnalyzer:
    """
    Clase para análisis de dependencia de datos de pozo
    """
    
    def __init__(self, data):
        self.data_original = data.copy()
        self.data_clean = self._clean_data(data)
        self.available_cols = self.data_clean.columns.tolist()
        self.correlation_matrices = {}
        self.cqr = CopulaQuantileRegression()
        
    def _clean_data(self, data):
        """Limpia los datos"""
        df = data.copy()
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df
    
    def calculate_dependence_measures(self, col1, col2):
        """Calcula medidas de dependencia"""
        x = self.data_clean[col1].values
        y = self.data_clean[col2].values
        
        # Correlaciones
        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)
        kendall_corr, kendall_p = kendalltau(x, y)
        
        return {
            'Pearson_r': pearson_corr,
            'Pearson_R2': pearson_corr**2,
            'Spearman_rho': spearman_corr,
            'Kendall_tau': kendall_corr,
            'Pearson_p': pearson_p,
            'Spearman_p': spearman_p,
            'Kendall_p': kendall_p
        }
    
    def create_scatter_plot_with_quantiles(self, col1, col2, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                                          copula_type='best', add_regression=True, add_density=True):
        """
        Crea scatter plot con regresión cuantil basada en cópulas
        """
        x = self.data_clean[col1].values
        y = self.data_clean[col2].values
        dep = self.calculate_dependence_measures(col1, col2)
        
        # Calcular regresión cuantil
        with st.spinner('Calculando regresión cuantil basada en cópulas...'):
            quantile_results = self.cqr.quantile_regression(x, y, quantiles, copula_type)
        
        # Crear figura con matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot principal con densidad
        if add_density:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            scatter = ax.scatter(x[idx], y[idx], c=z[idx], s=30, 
                                cmap='viridis', alpha=0.7, edgecolors='white', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Densidad')
        else:
            ax.scatter(x, y, alpha=0.6, s=30, c='steelblue', 
                      edgecolors='white', linewidth=0.5)
        
        # Regresión lineal (opcional)
        if add_regression:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = p(x_line)
            ax.plot(x_line, y_line, 'r--', linewidth=2, 
                   label=f'Lineal (R²={dep["Pearson_R2"]:.3f})')
        
        # Curvas de regresión cuantil
        colors = ['#800000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        quantile_labels = {
            0.05: '5% (Cola inferior)',
            0.10: '10%',
            0.25: '25% (Cuartil 1)',
            0.50: '50% (Mediana)',
            0.75: '75% (Cuartil 3)',
            0.90: '90%',
            0.95: '95% (Cola superior)'
        }
        
        x_grid = quantile_results['x_grid']
        
        for i, q in enumerate(quantiles):
            if q in quantile_results:
                y_q = quantile_results[q]
                color = colors[i % len(colors)]
                linewidth = 2.5 if q == 0.5 else 1.5
                linestyle = '-' if q == 0.5 else '--'
                label = quantile_labels.get(q, f'Cuantil {int(q*100)}%')
                ax.plot(x_grid, y_q, color=color, linestyle=linestyle, 
                       linewidth=linewidth, label=label, alpha=0.8)
        
        # Configuración del gráfico
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f'{col1} vs {col2}\nRegresión Cuantil basada en Cópula {quantile_results["copula"]["type"].capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Añadir caja con medidas de dependencia
        textstr = f'$\\mathbf{{Pearson\\ r = {dep["Pearson_r"]:.4f}}}$\n' \
                  f'$\\mathbf{{Spearman\\ \\rho = {dep["Spearman_rho"]:.4f}}}$\n' \
                  f'$\\mathbf{{Kendall\\ \\tau = {dep["Kendall_tau"]:.4f}}}$\n' \
                  f'$\\mathbf{{R^2 = {dep["Pearson_R2"]:.4f}}}$\n' \
                  f'$\\mathbf{{Cópula: {quantile_results["copula"]["type"].capitalize()}}}$'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
        
        plt.tight_layout()
        return fig, quantile_results
    
    def create_scatter_plot_plotly_with_quantiles(self, col1, col2, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                                                  copula_type='best', add_regression=True, add_density=True):
        """
        Crea scatter plot interactivo con regresión cuantil basada en cópulas usando Plotly
        """
        x = self.data_clean[col1].values
        y = self.data_clean[col2].values
        dep = self.calculate_dependence_measures(col1, col2)
        
        # Calcular regresión cuantil
        with st.spinner('Calculando regresión cuantil basada en cópulas...'):
            quantile_results = self.cqr.quantile_regression(x, y, quantiles, copula_type)
        
        fig = go.Figure()
        
        # Scatter plot principal
        if add_density:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(
                    size=8,
                    color=z,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Densidad"),
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                text=[f'{col1}: {xi:.2f}<br>{col2}: {yi:.2f}' for xi, yi in zip(x, y)],
                hoverinfo='text',
                name='Datos'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(size=8, color='steelblue', opacity=0.6, line=dict(width=0.5, color='white')),
                text=[f'{col1}: {xi:.2f}<br>{col2}: {yi:.2f}' for xi, yi in zip(x, y)],
                hoverinfo='text',
                name='Datos'
            ))
        
        # Regresión lineal
        if add_regression:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = p(x_line)
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f'Lineal (R²={dep["Pearson_R2"]:.3f})'
            ))
        
        # Curvas de regresión cuantil
        colors = ['#800000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        quantile_labels = {
            0.05: '5% (Cola inferior)',
            0.10: '10%',
            0.25: '25% (Cuartil 1)',
            0.50: '50% (Mediana)',
            0.75: '75% (Cuartil 3)',
            0.90: '90%',
            0.95: '95% (Cola superior)'
        }
        
        x_grid = quantile_results['x_grid']
        
        for i, q in enumerate(quantiles):
            if q in quantile_results:
                y_q = quantile_results[q]
                color = colors[i % len(colors)]
                linewidth = 3 if q == 0.5 else 2
                dash = 'solid' if q == 0.5 else 'dash'
                label = quantile_labels.get(q, f'Cuantil {int(q*100)}%')
                
                fig.add_trace(go.Scatter(
                    x=x_grid, y=y_q,
                    mode='lines',
                    line=dict(color=color, width=linewidth, dash=dash),
                    name=label,
                    hoverinfo='text',
                    text=[f'{col1}: {xi:.1f}<br>{col2}: {yi:.1f}<br>Cuantil: {int(q*100)}%' 
                          for xi, yi in zip(x_grid, y_q)]
                ))
        
        # Configuración del layout
        fig.update_layout(
            title=f'<b>{col1} vs {col2}</b><br>Regresión Cuantil basada en Cópula {quantile_results["copula"]["type"].capitalize()}',
            xaxis_title=f'<b>{col1}</b>',
            yaxis_title=f'<b>{col2}</b>',
            hovermode='closest',
            width=1000,
            height=700,
            showlegend=True,
            legend=dict(x=0.95, y=0.05, xanchor='right', yanchor='bottom', 
                       bgcolor='rgba(255,255,255,0.8)')
        )
        
        # Añadir anotación con medidas de dependencia
        annotation_text = (
            f"<b>Pearson r = {dep['Pearson_r']:.4f}</b><br>"
            f"<b>Spearman ρ = {dep['Spearman_rho']:.4f}</b><br>"
            f"<b>Kendall τ = {dep['Kendall_tau']:.4f}</b><br>"
            f"<b>R² = {dep['Pearson_R2']:.4f}</b><br>"
            f"<b>Cópula: {quantile_results['copula']['type'].capitalize()}</b>"
        )
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=11, family="monospace"),
            align="left",
            bgcolor="rgba(245, 245, 220, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=8
        )
        
        return fig, quantile_results
    
    def create_heatmap(self, correlation_type='spearman', figsize=(10, 8)):
        """Crea heatmap de correlación con matplotlib"""
        if not self.correlation_matrices:
            self.compute_correlation_matrices()
        
        corr_matrix = self.correlation_matrices[correlation_type]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Configurar ejes
        ax.set_xticks(np.arange(len(self.available_cols)))
        ax.set_yticks(np.arange(len(self.available_cols)))
        ax.set_xticklabels(self.available_cols, rotation=45, ha='right')
        ax.set_yticklabels(self.available_cols)
        
        # Añadir valores
        for i in range(len(self.available_cols)):
            for j in range(len(self.available_cols)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                             fontsize=10)
        
        # Barra de color
        plt.colorbar(im, ax=ax, label=f'Correlación de {correlation_type.capitalize()}')
        
        title_map = {
            'pearson': 'Pearson (Correlación Lineal)',
            'spearman': 'Spearman (Correlación de Rangos)',
            'kendall': "Kendall τ (Concordancia)"
        }
        ax.set_title(title_map.get(correlation_type, correlation_type), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def compute_correlation_matrices(self):
        """Calcula matrices de correlación"""
        n = len(self.available_cols)
        pearson_matrix = np.zeros((n, n))
        spearman_matrix = np.zeros((n, n))
        kendall_matrix = np.zeros((n, n))
        
        for i, col1 in enumerate(self.available_cols):
            for j, col2 in enumerate(self.available_cols):
                dep = self.calculate_dependence_measures(col1, col2)
                pearson_matrix[i, j] = dep['Pearson_r']
                spearman_matrix[i, j] = dep['Spearman_rho']
                kendall_matrix[i, j] = dep['Kendall_tau']
        
        self.correlation_matrices = {
            'pearson': pearson_matrix,
            'spearman': spearman_matrix,
            'kendall': kendall_matrix
        }
        
        return pearson_matrix, spearman_matrix, kendall_matrix
    
    def create_pairplot_matrix(self, selected_cols, sample_size=500):
        """Crea matriz de dispersión interactiva con Plotly"""
        if len(selected_cols) < 2:
            return None
        
        # Muestrear datos si son muchos
        data_subset = self.data_clean[selected_cols].copy()
        if len(data_subset) > sample_size:
            data_subset = data_subset.sample(n=sample_size, random_state=42)
        
        # Crear matriz de subplots
        n = len(selected_cols)
        fig = make_subplots(
            rows=n, cols=n,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        # Llenar la matriz
        for i, col1 in enumerate(selected_cols):
            for j, col2 in enumerate(selected_cols):
                if i == j:
                    # Diagonal: histogramas
                    hist_data = data_subset[col1]
                    fig.add_trace(
                        go.Histogram(x=hist_data, name=col1, showlegend=False),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: scatter plots
                    x = data_subset[col1]
                    y = data_subset[col2]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=y,
                            mode='markers',
                            marker=dict(size=5, opacity=0.5, color='steelblue'),
                            showlegend=False,
                            hoverinfo='text',
                            text=[f'{col1}: {xi:.2f}<br>{col2}: {yi:.2f}' for xi, yi in zip(x, y)]
                        ),
                        row=i+1, col=j+1
                    )
        
        # Actualizar layout
        fig.update_layout(
            height=800,
            width=800,
            title_text="Matriz de Dispersión",
            showlegend=False
        )
        
        # Actualizar ejes
        for i, col in enumerate(selected_cols):
            fig.update_xaxes(title_text=col if i == n-1 else "", row=n, col=i+1)
            fig.update_yaxes(title_text=col if i == 0 else "", row=i+1, col=1)
        
        return fig
    
    def create_summary_table(self, selected_cols):
        """Crea tabla resumen de estadísticas"""
        summary = []
        for i, col1 in enumerate(selected_cols):
            for j, col2 in enumerate(selected_cols):
                if i < j:
                    dep = self.calculate_dependence_measures(col1, col2)
                    summary.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Pearson r': f"{dep['Pearson_r']:.4f}",
                        'Pearson R²': f"{dep['Pearson_R2']:.4f}",
                        'Spearman ρ': f"{dep['Spearman_rho']:.4f}",
                        'Kendall τ': f"{dep['Kendall_tau']:.4f}",
                        'p-value (Pearson)': f"{dep['Pearson_p']:.4e}",
                        'p-value (Spearman)': f"{dep['Spearman_p']:.4e}",
                        'p-value (Kendall)': f"{dep['Kendall_p']:.4e}"
                    })
        
        return pd.DataFrame(summary)
    
    def get_data_summary(self):
        """Obtiene resumen estadístico de los datos"""
        return self.data_clean.describe()
    
    def get_correlation_summary(self):
        """Obtiene resumen de correlaciones"""
        if not self.correlation_matrices:
            self.compute_correlation_matrices()
        
        summary = []
        for i, col1 in enumerate(self.available_cols):
            for j, col2 in enumerate(self.available_cols):
                if i < j:
                    summary.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Pearson': self.correlation_matrices['pearson'][i, j],
                        'Spearman': self.correlation_matrices['spearman'][i, j],
                        'Kendall': self.correlation_matrices['kendall'][i, j]
                    })
        
        return pd.DataFrame(summary)


def create_synthetic_data():
    """Crea datos sintéticos para demostración"""
    np.random.seed(42)
    n = 500
    
    vclay = np.random.beta(2, 5, n) * 100
    phie = 30 - vclay * 0.2 + np.random.normal(0, 3, n)
    phie = np.clip(phie, 5, 35)
    vp = 4500 - phie * 45 + np.random.normal(0, 150, n)
    vp = np.clip(vp, 2800, 5200)
    vs = vp * 0.55 + np.random.normal(0, 80, n)
    rho = 2.65 - phie * 0.015 + np.random.normal(0, 0.05, n)
    gr = vclay * 1.5 + np.random.normal(0, 8, n)
    gr = np.clip(gr, 20, 180)
    rt = 50 / (phie + 5) + np.random.exponential(2, n)
    sw = 0.2 + 0.7 * np.exp(-phie/12) + np.random.normal(0, 0.05, n)
    sw = np.clip(sw, 0.1, 0.95)
    
    df = pd.DataFrame({
        'Vp': vp,
        'Vs': vs,
        'Phie': phie,
        'rho': rho,
        'GR': gr,
        'RT': rt,
        'SW': sw,
        'Vclay': vclay
    })
    
    return df


# ============================================================================
# INTERFAZ DE STREAMLIT
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🛢️ Análisis de Dependencia en Datos de Pozo</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Carga de datos
    with st.sidebar:
        st.header("📂 Carga de Datos")
        
        # Opción de carga
        data_source = st.radio(
            "Seleccionar fuente de datos:",
            ["📁 Cargar archivo CSV", "🎲 Usar datos sintéticos"]
        )
        
        data = None
        
        if data_source == "📁 Cargar archivo CSV":
            uploaded_file = st.file_uploader("Seleccionar archivo CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Datos cargados: {len(data)} filas, {len(data.columns)} columnas")
                except Exception as e:
                    st.error(f"Error al cargar archivo: {e}")
        else:
            data = create_synthetic_data()
            st.info("🎲 Usando datos sintéticos generados para demostración")
        
        if data is not None:
            # Mostrar preview de datos
            st.subheader("📊 Vista previa de datos")
            st.dataframe(data.head(), use_container_width=True)
            
            # Inicializar analizador
            analyzer = WellLogDependenceAnalyzer(data)
            
            # Selección de variables
            st.subheader("🔧 Configuración")
            available_cols = analyzer.available_cols
            
            if len(available_cols) > 0:
                st.success(f"Variables disponibles: {len(available_cols)}")
            else:
                st.error("No se encontraron variables numéricas en los datos")
        
        st.markdown("---")
        st.caption("Creado para análisis de registros de pozo")
    
    # Main content
    if data is not None and len(analyzer.available_cols) > 0:
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Scatter Plot con Cuantiles", 
            "🔥 Matriz de Correlación", 
            "📊 Matriz de Dispersión",
            "📋 Tabla de Dependencias",
            "ℹ️ Información del Dataset"
        ])
        
        # Tab 1: Scatter Plot con Regresión Cuantil
        with tab1:
            st.markdown('<div class="sub-header">📈 Scatter Plot con Regresión Cuantil basada en Cópulas</div>', unsafe_allow_html=True)
            st.markdown("""
            La regresión cuantil basada en cópulas permite visualizar cómo se comporta la relación
            en diferentes percentiles de la distribución (cola inferior, mediana, cola superior).
            """)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                var_x = st.selectbox("Variable X:", analyzer.available_cols, index=0)
            with col2:
                var_y = st.selectbox("Variable Y:", analyzer.available_cols, index=min(1, len(analyzer.available_cols)-1))
            with col3:
                add_regression = st.checkbox("Mostrar regresión lineal", value=True)
                add_density = st.checkbox("Mostrar densidad", value=True)
                use_plotly = st.checkbox("Modo interactivo", value=True)
            
            # Selección de cuantiles
            st.markdown("### 📊 Selección de Cuantiles")
            col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
            
            with col_q1:
                show_q05 = st.checkbox("5% (Cola inferior)", value=True)
            with col_q2:
                show_q25 = st.checkbox("25% (Cuartil 1)", value=True)
            with col_q3:
                show_q50 = st.checkbox("50% (Mediana)", value=True)
            with col_q4:
                show_q75 = st.checkbox("75% (Cuartil 3)", value=True)
            with col_q5:
                show_q95 = st.checkbox("95% (Cola superior)", value=True)
            
            # Tipo de cópula
            copula_type = st.selectbox(
                "Tipo de cópula para regresión cuantil:",
                ["best", "gaussian", "clayton", "gumbel", "frank"],
                format_func=lambda x: {
                    'best': 'Mejor cópula (automático)',
                    'gaussian': 'Gaussiana',
                    'clayton': 'Clayton (cola inferior)',
                    'gumbel': 'Gumbel (cola superior)',
                    'frank': 'Frank (simétrica)'
                }[x]
            )
            
            # Construir lista de cuantiles
            quantiles = []
            if show_q05:
                quantiles.append(0.05)
            if show_q25:
                quantiles.append(0.25)
            if show_q50:
                quantiles.append(0.50)
            if show_q75:
                quantiles.append(0.75)
            if show_q95:
                quantiles.append(0.95)
            
            if not quantiles:
                quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
            
            if var_x and var_y:
                # Mostrar medidas de dependencia
                dep = analyzer.calculate_dependence_measures(var_x, var_y)
                
                # Métricas destacadas
                st.markdown("### 📊 Medidas de Dependencia")
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Pearson r", f"{dep['Pearson_r']:.4f}", 
                             delta=f"R² = {dep['Pearson_R2']:.4f}")
                with col_metrics[1]:
                    st.metric("Spearman ρ", f"{dep['Spearman_rho']:.4f}")
                with col_metrics[2]:
                    st.metric("Kendall τ", f"{dep['Kendall_tau']:.4f}")
                with col_metrics[3]:
                    diff = abs(dep['Spearman_rho'] - dep['Pearson_r'])
                    st.metric("|Spearman - Pearson|", f"{diff:.4f}",
                             delta="No-linealidad" if diff > 0.2 else "Lineal")
                
                # Generar gráfico con regresión cuantil
                if use_plotly:
                    fig, quantile_results = analyzer.create_scatter_plot_plotly_with_quantiles(
                        var_x, var_y, quantiles, copula_type, add_regression, add_density
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, quantile_results = analyzer.create_scatter_plot_with_quantiles(
                        var_x, var_y, quantiles, copula_type, add_regression, add_density
                    )
                    st.pyplot(fig)
                
                # Mostrar información de la cópula seleccionada
                st.markdown("### 📈 Información de la Cópula")
                cop = quantile_results['copula']
                st.info(f"""
                **Cópula seleccionada:** {cop['type'].capitalize()}
                
                **Parámetros:**
                {self._format_copula_params(cop)}
                
                **Interpretación:**
                {self._interpret_copula(cop)}
                """)
                
                # Interpretación de los cuantiles
                st.markdown("### 💡 Interpretación de la Regresión Cuantil")
                st.markdown("""
                - **Línea sólida (50%)**: Mediana condicional - valor central esperado
                - **Líneas discontinuas**: Otros cuantiles - muestran la variabilidad de la relación
                - **Separación entre cuantiles**: Mayor separación indica mayor heterocedasticidad
                - **Cuantiles extremos (5% y 95%)**: Muestran comportamiento en colas de la distribución
                """)
                
                # Detección de heterocedasticidad
                if len(quantiles) >= 3 and 0.05 in quantile_results and 0.95 in quantile_results:
                    y_05 = quantile_results[0.05]
                    y_95 = quantile_results[0.95]
                    spread = y_95 - y_05
                    spread_range = spread.max() - spread.min()
                    if spread_range > np.std(y) * 0.5:
                        st.warning("⚠️ Se detecta heterocedasticidad (variabilidad no constante) - los cuantiles se separan a lo largo de X")
                    else:
                        st.success("✅ Varianza relativamente constante - los cuantiles se mantienen paralelos")
        
        # Tab 2: Matriz de Correlación
        with tab2:
            st.markdown('<div class="sub-header">🔥 Matriz de Correlación</div>', unsafe_allow_html=True)
            
            corr_type = st.selectbox(
                "Tipo de correlación:",
                ["spearman", "pearson", "kendall"],
                format_func=lambda x: {
                    'pearson': 'Pearson (Correlación Lineal)',
                    'spearman': 'Spearman (Correlación de Rangos)',
                    'kendall': "Kendall τ (Concordancia)"
                }[x]
            )
            
            fig_heatmap = analyzer.create_heatmap(corr_type, figsize=(10, 8))
            st.pyplot(fig_heatmap)
            
            if st.checkbox("Mostrar matriz numérica"):
                if not analyzer.correlation_matrices:
                    analyzer.compute_correlation_matrices()
                corr_df = pd.DataFrame(
                    analyzer.correlation_matrices[corr_type],
                    index=analyzer.available_cols,
                    columns=analyzer.available_cols
                )
                st.dataframe(corr_df.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))
        
        # Tab 3: Matriz de Dispersión
        with tab3:
            st.markdown('<div class="sub-header">📊 Matriz de Dispersión</div>', unsafe_allow_html=True)
            
            selected_vars_matrix = st.multiselect(
                "Variables a incluir:",
                analyzer.available_cols,
                default=analyzer.available_cols[:min(4, len(analyzer.available_cols))]
            )
            
            if len(selected_vars_matrix) >= 2:
                sample_size = st.slider("Tamaño de muestra:", 100, 1000, 500)
                fig_matrix = analyzer.create_pairplot_matrix(selected_vars_matrix, sample_size)
                if fig_matrix:
                    st.plotly_chart(fig_matrix, use_container_width=True)
            else:
                st.warning("Selecciona al menos 2 variables para generar la matriz de dispersión")
        
        # Tab 4: Tabla de Dependencias
        with tab4:
            st.markdown('<div class="sub-header">📋 Tabla de Dependencias</div>', unsafe_allow_html=True)
            
            selected_vars_table = st.multiselect(
                "Variables a analizar (dejar vacío para todas):",
                analyzer.available_cols,
                default=[]
            )
            
            if len(selected_vars_table) == 0:
                selected_vars_table = analyzer.available_cols
            
            if len(selected_vars_table) >= 2:
                summary_table = analyzer.create_summary_table(selected_vars_table)
                st.dataframe(summary_table, use_container_width=True)
                
                csv = summary_table.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar tabla como CSV",
                    data=csv,
                    file_name="dependence_summary.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Selecciona al menos 2 variables para generar la tabla")
        
        # Tab 5: Información del Dataset
        with tab5:
            st.markdown('<div class="sub-header">ℹ️ Información del Dataset</div>', unsafe_allow_html=True)
            
            st.markdown("### 📊 Estadísticas Descriptivas")
            st.dataframe(analyzer.get_data_summary(), use_container_width=True)
            
            st.markdown("### 📈 Resumen de Correlaciones (Top 10)")
            corr_summary = analyzer.get_correlation_summary()
            if not corr_summary.empty:
                corr_summary['abs_Spearman'] = corr_summary['Spearman'].abs()
                corr_summary = corr_summary.sort_values('abs_Spearman', ascending=False).head(10)
                corr_summary = corr_summary.drop('abs_Spearman', axis=1)
                st.dataframe(corr_summary, use_container_width=True)
            
            st.markdown("### 🔍 Calidad de Datos")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total de muestras", len(analyzer.data_clean))
                st.metric("Variables numéricas", len(analyzer.available_cols))
            with col2:
                missing_pct = (1 - len(analyzer.data_clean)/len(analyzer.data_original)) * 100
                st.metric("Datos después de limpieza", f"{len(analyzer.data_clean)} ({100-missing_pct:.1f}%)")
                st.metric("Valores nulos eliminados", f"{len(analyzer.data_original) - len(analyzer.data_clean)}")
    
    else:
        st.info("👈 Por favor, carga un archivo CSV o usa los datos sintéticos para comenzar el análisis")
        
        st.markdown("### 📝 Formato esperado del archivo CSV")
        st.markdown("""
        El archivo CSV debe contener columnas numéricas con registros de pozo como:
        - **Vp**: Velocidad de onda P
        - **Vs**: Velocidad de onda S  
        - **Phie**: Porosidad efectiva
        - **rho**: Densidad
        - **GR**: Rayos Gamma
        - **RT**: Resistividad
        - **SW**: Saturación de agua
        - **Vclay**: Volumen de arcilla
        
        *Nota: Los nombres de las columnas pueden variar, se detectarán automáticamente.*
        """)


def _format_copula_params(cop):
    """Formatea los parámetros de la cópula para mostrar"""
    if cop['type'] == 'gaussian':
        return f"- ρ (correlación) = {cop['rho']:.4f}"
    elif cop['type'] == 'clayton':
        return f"- θ (theta) = {cop['theta']:.4f}\n- Dependencia en cola inferior: Sí"
    elif cop['type'] == 'gumbel':
        return f"- θ (theta) = {cop['theta']:.4f}\n- Dependencia en cola superior: Sí"
    elif cop['type'] == 'frank':
        return f"- θ (theta) = {cop['theta']:.4f}\n- Dependencia simétrica"
    return "-"


def _interpret_copula(cop):
    """Interpreta el significado de la cópula seleccionada"""
    if cop['type'] == 'gaussian':
        return "La cópula Gaussiana sugiere una dependencia simétrica, similar a la correlación lineal tradicional. Es adecuada cuando la relación es aproximadamente lineal y no hay asimetría en las colas."
    elif cop['type'] == 'clayton':
        theta = cop['theta']
        if theta > 1:
            return f"La cópula Clayton (θ={theta:.2f}) indica fuerte dependencia en la cola inferior. Esto significa que valores bajos de X tienden a asociarse con valores bajos de Y (útil para modelar eventos extremos negativos)."
        else:
            return f"La cópula Clayton (θ={theta:.2f}) muestra dependencia moderada en la cola inferior."
    elif cop['type'] == 'gumbel':
        theta = cop['theta']
        if theta > 1.5:
            return f"La cópula Gumbel (θ={theta:.2f}) indica fuerte dependencia en la cola superior. Valores altos de X tienden a asociarse con valores altos de Y (útil para modelar máximos conjuntos)."
        else:
            return f"La cópula Gumbel (θ={theta:.2f}) muestra dependencia moderada en la cola superior."
    elif cop['type'] == 'frank':
        theta = abs(cop['theta'])
        if theta > 5:
            return f"La cópula Frank (θ={cop['theta']:.2f}) muestra dependencia fuerte y simétrica. La relación es similar en todas las partes de la distribución."
        else:
            return f"La cópula Frank (θ={cop['theta']:.2f}) muestra dependencia moderada y simétrica."
    return ""


if __name__ == "__main__":
    main()
