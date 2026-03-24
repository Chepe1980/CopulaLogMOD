"""
Streamlit App para Análisis de Dependencia en Datos de Pozo
===========================================================
Con regresión cuantil basada en cópulas y estimación multivariada
VERSIÓN MEJORADA CON ESTIMACIÓN MULTIVARIADA
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, kendalltau, spearmanr, pearsonr, gaussian_kde
from scipy.interpolate import UnivariateSpline, interp1d, interp2d, RegularGridInterpolator
from scipy.optimize import minimize, brentq
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
    .prediction-card {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def format_copula_params(cop):
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


def interpret_copula(cop):
    """Interpreta el significado de la cópula seleccionada"""
    if cop['type'] == 'gaussian':
        return "La cópula Gaussiana sugiere una dependencia simétrica, similar a la correlación lineal tradicional."
    elif cop['type'] == 'clayton':
        theta = cop['theta']
        if theta > 1:
            return f"La cópula Clayton (θ={theta:.2f}) indica fuerte dependencia en la cola inferior. Valores bajos de X tienden a asociarse con valores bajos de Y."
        else:
            return f"La cópula Clayton (θ={theta:.2f}) muestra dependencia moderada en la cola inferior."
    elif cop['type'] == 'gumbel':
        theta = cop['theta']
        if theta > 1.5:
            return f"La cópula Gumbel (θ={theta:.2f}) indica fuerte dependencia en la cola superior. Valores altos de X tienden a asociarse con valores altos de Y."
        else:
            return f"La cópula Gumbel (θ={theta:.2f}) muestra dependencia moderada en la cola superior."
    elif cop['type'] == 'frank':
        theta = abs(cop['theta'])
        if theta > 5:
            return f"La cópula Frank (θ={cop['theta']:.2f}) muestra dependencia fuerte y simétrica."
        else:
            return f"La cópula Frank (θ={cop['theta']:.2f}) muestra dependencia moderada y simétrica."
    return ""


class MultivariateCopulaEstimator:
    """
    Clase para estimación multivariada usando cópulas y métodos de aprendizaje automático
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def estimate_with_random_forest(self, X_train, y_train, X_test, **kwargs):
        """
        Estimación usando Random Forest
        """
        rf = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        return y_pred, rf
    
    def estimate_with_gradient_boosting(self, X_train, y_train, X_test, **kwargs):
        """
        Estimación usando Gradient Boosting
        """
        gb = GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42
        )
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        return y_pred, gb
    
    def estimate_with_knn_copula(self, X_train, y_train, X_test, n_neighbors=5):
        """
        Estimación usando KNN con pesos basados en cópula
        """
        # Normalizar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Usar KNN con pesos basados en distancia
        tree = cKDTree(X_train_scaled)
        y_pred = []
        
        for x_test in X_test_scaled:
            # Encontrar vecinos
            distances, indices = tree.query(x_test.reshape(1, -1), k=n_neighbors)
            
            # Calcular pesos (inverso de la distancia)
            weights = 1 / (distances[0] + 1e-6)
            weights = weights / weights.sum()
            
            # Predicción ponderada
            pred = np.sum(weights * y_train[indices[0]])
            y_pred.append(pred)
        
        return np.array(y_pred), scaler
    
    def estimate_with_ensemble(self, X_train, y_train, X_test, methods=['rf', 'gb', 'knn']):
        """
        Estimación usando ensemble de múltiples métodos
        """
        predictions = []
        models = []
        
        if 'rf' in methods:
            y_pred_rf, model_rf = self.estimate_with_random_forest(X_train, y_train, X_test)
            predictions.append(y_pred_rf)
            models.append(('Random Forest', model_rf))
        
        if 'gb' in methods:
            y_pred_gb, model_gb = self.estimate_with_gradient_boosting(X_train, y_train, X_test)
            predictions.append(y_pred_gb)
            models.append(('Gradient Boosting', model_gb))
        
        if 'knn' in methods:
            y_pred_knn, scaler = self.estimate_with_knn_copula(X_train, y_train, X_test)
            predictions.append(y_pred_knn)
            models.append(('KNN-Copula', scaler))
        
        # Promedio simple
        y_pred_ensemble = np.mean(predictions, axis=0)
        
        return y_pred_ensemble, models


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
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        rho = np.corrcoef(z1, z2)[0, 1]
        return {'type': 'gaussian', 'rho': rho}
    
    def fit_clayton_copula(self, u, v):
        """Ajusta una cópula Clayton"""
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
        
        best_copula = None
        best_aic = np.inf
        
        for name, params in copulas.items():
            try:
                if name == 'gaussian':
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
    
    def _gaussian_log_likelihood(self, u, v, rho):
        """Log-likelihood para cópula Gaussiana"""
        n = len(u)
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        log_lik = -n/2 * np.log(1 - rho**2) - np.sum((z1**2 + z2**2 - 2*rho*z1*z2) / (2*(1 - rho**2)))
        return log_lik
    
    def _clayton_log_likelihood(self, u, v, theta):
        """Log-likelihood para cópula Clayton"""
        if theta <= 0:
            return -np.inf
        n = len(u)
        log_lik = np.sum(np.log((1 + theta) * (u * v)**(-theta-1) * 
                                (u**(-theta) + v**(-theta) - 1)**(-1/theta - 2)))
        return log_lik
    
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
        """Realiza regresión cuantil basada en cópulas"""
        u = self.empirical_cdf(x)
        v = self.empirical_cdf(y)
        
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
        
        x_grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_points)
        u_grid = self.empirical_cdf(x_grid)
        
        quantile_results = {}
        
        for q in quantiles:
            y_q = []
            for u_val in u_grid:
                v_q = self._find_conditional_quantile(u_val, q, copula)
                y_val = self.empirical_inverse_cdf(v_q, y)
                y_q.append(y_val)
            
            quantile_results[q] = np.array(y_q)
        
        quantile_results['x_grid'] = x_grid
        quantile_results['copula'] = copula
        
        return quantile_results
    
    def _find_conditional_quantile(self, u, q, copula):
        """Encuentra v tal que P(V <= v | U = u) = q"""
        if copula['type'] == 'gaussian':
            rho = copula['rho']
            z_u = norm.ppf(u)
            z_v = norm.ppf(q)
            z_v_given_u = rho * z_u + np.sqrt(1 - rho**2) * z_v
            return norm.cdf(z_v_given_u)
        
        elif copula['type'] == 'clayton':
            theta = copula['theta']
            if theta == 0:
                return q
            v = ( (q**(-theta/(theta+1)) - 1) * u**(-theta) + 1 )**(-1/theta)
            return np.clip(v, 0, 1)
        
        elif copula['type'] == 'gumbel':
            theta = copula['theta']
            if theta == 1:
                return q
            
            try:
                v = brentq(lambda v: self._gumbel_copula_cdf(u, v, theta) - q, 0.001, 0.999)
                return v
            except:
                return q
        
        elif copula['type'] == 'frank':
            theta = copula['theta']
            if theta == 0:
                return q
            
            try:
                v = brentq(lambda v: self._frank_copula_cdf(u, v, theta) - q, 0.001, 0.999)
                return v
            except:
                return q
        
        else:
            return q
    
    def _gumbel_copula_cdf(self, u, v, theta):
        """CDF de la cópula Gumbel"""
        if theta == 1:
            return u * v
        return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))
    
    def _frank_copula_cdf(self, u, v, theta):
        """CDF de la cópula Frank"""
        if theta == 0:
            return u * v
        return -1/theta * np.log(1 + (np.exp(-theta*u) - 1) * (np.exp(-theta*v) - 1) / (np.exp(-theta) - 1))


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
        self.multivariate_estimator = MultivariateCopulaEstimator()
        
        # Detectar columna de profundidad si existe
        self.depth_col = None
        depth_candidates = ['DEPTH', 'Depth', 'depth', 'DEPT', 'Dept', 'dept', 'MD', 'TVD']
        for col in depth_candidates:
            if col in self.available_cols:
                self.depth_col = col
                break
        
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
    
    def estimate_multivariate(self, target_col, feature_cols, test_size=0.2, method='ensemble'):
        """
        Estimación multivariada usando múltiples variables predictoras
        
        Parameters:
        -----------
        target_col : str
            Variable objetivo a estimar
        feature_cols : list
            Lista de variables predictoras
        test_size : float
            Proporción de datos para prueba
        method : str
            Método de estimación: 'rf', 'gb', 'knn', 'ensemble'
        """
        # Preparar datos
        X = self.data_clean[feature_cols].values
        y = self.data_clean[target_col].values
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Entrenar y predecir
        if method == 'rf':
            y_pred, model = self.multivariate_estimator.estimate_with_random_forest(
                X_train, y_train, X_test
            )
            method_name = "Random Forest"
        elif method == 'gb':
            y_pred, model = self.multivariate_estimator.estimate_with_gradient_boosting(
                X_train, y_train, X_test
            )
            method_name = "Gradient Boosting"
        elif method == 'knn':
            y_pred, model = self.multivariate_estimator.estimate_with_knn_copula(
                X_train, y_train, X_test
            )
            method_name = "KNN-Copula"
        else:
            y_pred, models = self.multivariate_estimator.estimate_with_ensemble(
                X_train, y_train, X_test, methods=['rf', 'gb', 'knn']
            )
            method_name = "Ensemble (RF + GB + KNN)"
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Predecir para todos los datos
        X_all = self.data_clean[feature_cols].values
        if method == 'ensemble':
            # Reentrenar ensemble con todos los datos
            y_all_pred, _ = self.multivariate_estimator.estimate_with_ensemble(
                X_all, y, X_all, methods=['rf', 'gb', 'knn']
            )
        else:
            # Reentrenar con todos los datos
            if method == 'rf':
                model_full = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            elif method == 'gb':
                model_full = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            elif method == 'knn':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_all)
                y_all_pred, _ = self.multivariate_estimator.estimate_with_knn_copula(
                    X_all, y, X_all
                )
                return y_all_pred, y, r2, rmse, mae, method_name, feature_cols, X_all
        
        if method != 'knn' and method != 'ensemble':
            model_full.fit(X_all, y)
            y_all_pred = model_full.predict(X_all)
        
        return y_all_pred, y, r2, rmse, mae, method_name, feature_cols, X_all
    
    def create_estimation_plot(self, target_col, y_estimated, y_real, method_name, feature_cols, r2, rmse, mae):
        """
        Crea gráfico de comparación entre valores reales y estimados
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Comparación: Real vs Estimado', 'Error de Predicción', 'Distribución de Errores'),
            specs=[[{'type': 'scatter'}, {'type': 'box'}, {'type': 'histogram'}]]
        )
        
        # Gráfico de dispersión: Real vs Estimado
        fig.add_trace(
            go.Scatter(
                x=y_real, y=y_estimated,
                mode='markers',
                marker=dict(size=8, color='steelblue', opacity=0.6),
                name='Datos',
                text=[f'Real: {yi:.2f}<br>Estimado: {ye:.2f}' for yi, ye in zip(y_real, y_estimated)],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Línea de identidad
        min_val = min(y_real.min(), y_estimated.min())
        max_val = max(y_real.max(), y_estimated.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Identidad (y=x)'
            ),
            row=1, col=1
        )
        
        # Calcular errores
        errors = y_estimated - y_real
        
        # Box plot de errores
        fig.add_trace(
            go.Box(
                y=errors,
                name='Error',
                boxmean='sd',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # Histograma de errores
        fig.add_trace(
            go.Histogram(
                x=errors,
                name='Error',
                nbinsx=30,
                marker_color='lightblue'
            ),
            row=1, col=3
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=3)
        
        # Título
        fig.update_layout(
            title=f'Estimación Multivariada de {target_col}<br>'
                  f'Método: {method_name} | R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}<br>'
                  f'Variables predictoras: {", ".join(feature_cols)}',
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text=f'{target_col} Real', row=1, col=1)
        fig.update_yaxes(title_text=f'{target_col} Estimado', row=1, col=1)
        fig.update_xaxes(title_text='Error', row=1, col=2)
        fig.update_yaxes(title_text='Valor', row=1, col=2)
        fig.update_xaxes(title_text='Error', row=1, col=3)
        fig.update_yaxes(title_text='Frecuencia', row=1, col=3)
        
        return fig
    
    def create_depth_plot(self, target_col, y_estimated, y_real, depth, r2):
        """
        Crea gráfico de profundidad vs valores reales y estimados
        """
        fig = go.Figure()
        
        # Ordenar por profundidad
        idx = np.argsort(depth)
        depth_sorted = depth[idx]
        y_real_sorted = y_real[idx]
        y_est_sorted = y_estimated[idx]
        
        fig.add_trace(go.Scatter(
            x=y_real_sorted, y=depth_sorted,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.7, symbol='circle'),
            name='Real',
            text=[f'Prof: {d:.1f}<br>Real: {yr:.2f}' for d, yr in zip(depth_sorted, y_real_sorted)],
            hoverinfo='text'
        ))
        
        fig.add_trace(go.Scatter(
            x=y_est_sorted, y=depth_sorted,
            mode='markers',
            marker=dict(size=8, color='red', opacity=0.7, symbol='x'),
            name='Estimado',
            text=[f'Prof: {d:.1f}<br>Estimado: {ye:.2f}' for d, ye in zip(depth_sorted, y_est_sorted)],
            hoverinfo='text'
        ))
        
        # Líneas de tendencia
        from scipy.signal import savgol_filter
        try:
            window = min(51, len(depth_sorted) // 10 * 2 + 1)
            if window % 2 == 0:
                window += 1
            if window >= 3:
                y_real_smooth = savgol_filter(y_real_sorted, window, 2)
                y_est_smooth = savgol_filter(y_est_sorted, window, 2)
                
                fig.add_trace(go.Scatter(
                    x=y_real_smooth, y=depth_sorted,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Real (suavizado)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=y_est_smooth, y=depth_sorted,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Estimado (suavizado)'
                ))
        except:
            pass
        
        fig.update_layout(
            title=f'Comparación con Profundidad: {target_col}<br>R² = {r2:.4f}',
            xaxis_title=target_col,
            yaxis_title='Profundidad',
            yaxis_autorange='reversed',
            height=600,
            legend=dict(x=0.95, y=0.05, xanchor='right', yanchor='bottom'),
            hovermode='closest'
        )
        
        return fig
    
    def create_scatter_plot_with_quantiles(self, col1, col2, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                                          copula_type='best', add_regression=True, add_density=True):
        """Crea scatter plot con regresión cuantil basada en cópulas"""
        x = self.data_clean[col1].values
        y = self.data_clean[col2].values
        dep = self.calculate_dependence_measures(col1, col2)
        
        quantile_results = self.cqr.quantile_regression(x, y, quantiles, copula_type)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
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
        
        if add_regression:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = p(x_line)
            ax.plot(x_line, y_line, 'r--', linewidth=2, 
                   label=f'Lineal (R²={dep["Pearson_R2"]:.3f})')
        
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
        
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f'{col1} vs {col2}\nRegresión Cuantil basada en Cópula {quantile_results["copula"]["type"].capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
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
    
    def create_heatmap(self, correlation_type='spearman', figsize=(10, 8)):
        """Crea heatmap de correlación con matplotlib"""
        if not self.correlation_matrices:
            self.compute_correlation_matrices()
        
        corr_matrix = self.correlation_matrices[correlation_type]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(self.available_cols)))
        ax.set_yticks(np.arange(len(self.available_cols)))
        ax.set_xticklabels(self.available_cols, rotation=45, ha='right')
        ax.set_yticklabels(self.available_cols)
        
        for i in range(len(self.available_cols)):
            for j in range(len(self.available_cols)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                             fontsize=10)
        
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
        
        data_subset = self.data_clean[selected_cols].copy()
        if len(data_subset) > sample_size:
            data_subset = data_subset.sample(n=sample_size, random_state=42)
        
        n = len(selected_cols)
        fig = make_subplots(
            rows=n, cols=n,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        for i, col1 in enumerate(selected_cols):
            for j, col2 in enumerate(selected_cols):
                if i == j:
                    hist_data = data_subset[col1]
                    fig.add_trace(
                        go.Histogram(x=hist_data, name=col1, showlegend=False),
                        row=i+1, col=j+1
                    )
                else:
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
        
        fig.update_layout(
            height=800,
            width=800,
            title_text="Matriz de Dispersión",
            showlegend=False
        )
        
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
    
    def create_box_plots(self, selected_cols):
        """Crea box plots para las variables seleccionadas"""
        if len(selected_cols) == 0:
            return None
        
        data_plot = self.data_clean[selected_cols].copy()
        
        fig = go.Figure()
        
        for col in selected_cols:
            fig.add_trace(go.Box(
                y=data_plot[col],
                name=col,
                boxmean='sd',
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title='Box Plots de Variables Seleccionadas',
            yaxis_title='Valor',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_violin_plots(self, selected_cols):
        """Crea violin plots para las variables seleccionadas"""
        if len(selected_cols) == 0:
            return None
        
        data_plot = self.data_clean[selected_cols].copy()
        
        fig = go.Figure()
        
        for col in selected_cols:
            fig.add_trace(go.Violin(
                y=data_plot[col],
                name=col,
                box_visible=True,
                meanline_visible=True,
                points='outliers'
            ))
        
        fig.update_layout(
            title='Violin Plots de Variables Seleccionadas',
            yaxis_title='Valor',
            height=500,
            showlegend=True
        )
        
        return fig


def create_synthetic_data():
    """Crea datos sintéticos para demostración con relaciones multivariadas realistas"""
    np.random.seed(42)
    n = 500
    
    # Crear profundidad
    depth = np.linspace(0, 1000, n) + np.random.normal(0, 5, n)
    depth = np.sort(depth)
    
    # Relaciones multivariadas realistas
    # Vclay depende de profundidad y ruido
    vclay = 15 + 0.04 * depth + np.random.normal(0, 8, n)
    vclay = np.clip(vclay, 8, 75)
    
    # Phie depende inversamente de Vclay y profundidad
    phie = 32 - 0.18 * (vclay/10) - 0.008 * depth + np.random.normal(0, 2.5, n)
    phie = np.clip(phie, 8, 38)
    
    # Vp depende de Phie, Vclay y profundidad
    vp = 4800 - 48 * phie - 12 * (vclay/10) + 0.35 * depth + np.random.normal(0, 100, n)
    vp = np.clip(vp, 3000, 5500)
    
    # Vs depende de Vp
    vs = vp * 0.52 + np.random.normal(0, 60, n)
    
    # Rho depende de Phie y Vclay
    rho = 2.68 - 0.016 * phie + 0.003 * vclay + np.random.normal(0, 0.04, n)
    
    # GR depende de Vclay
    gr = vclay * 1.5 + 15 + np.random.normal(0, 6, n)
    gr = np.clip(gr, 20, 140)
    
    # RT depende de Phie y profundidad
    rt = 80 / (phie + 3) * np.exp(-depth/1200) + np.random.exponential(1.5, n)
    
    # SW depende de Phie y profundidad
    sw = 0.25 + 0.55 * np.exp(-phie/14) + 0.00015 * depth + np.random.normal(0, 0.045, n)
    sw = np.clip(sw, 0.12, 0.92)
    
    df = pd.DataFrame({
        'DEPTH': depth,
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
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🛢️ Análisis de Dependencia en Datos de Pozo</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Carga de datos
    with st.sidebar:
        st.header("📂 Carga de Datos")
        
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
            st.subheader("📊 Vista previa de datos")
            st.dataframe(data.head(), use_container_width=True)
            
            analyzer = WellLogDependenceAnalyzer(data)
            available_cols = analyzer.available_cols
            
            if len(available_cols) > 0:
                st.success(f"Variables disponibles: {len(available_cols)}")
                if analyzer.depth_col:
                    st.info(f"📏 Profundidad detectada: {analyzer.depth_col}")
            else:
                st.error("No se encontraron variables numéricas en los datos")
        
        st.markdown("---")
        st.caption("Creado para análisis de registros de pozo")
    
    # Main content
    if data is not None and len(analyzer.available_cols) > 0:
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Scatter Plot con Cuantiles", 
            "🎯 Estimación Multivariada",
            "📊 Box Plots",
            "🔥 Matriz de Correlación", 
            "📊 Matriz de Dispersión",
            "📋 Tabla de Dependencias"
        ])
        
        # Tab 1: Scatter Plot con Regresión Cuantil
        with tab1:
            st.markdown('<div class="sub-header">📈 Scatter Plot con Regresión Cuantil basada en Cópulas</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                var_x = st.selectbox("Variable X:", analyzer.available_cols, index=0)
            with col2:
                var_y = st.selectbox("Variable Y:", analyzer.available_cols, index=min(1, len(analyzer.available_cols)-1))
            with col3:
                add_regression = st.checkbox("Mostrar regresión lineal", value=True)
                add_density = st.checkbox("Mostrar densidad", value=True)
            
            st.markdown("### 📊 Selección de Cuantiles")
            col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
            
            with col_q1:
                show_q05 = st.checkbox("5%", value=True)
            with col_q2:
                show_q25 = st.checkbox("25%", value=True)
            with col_q3:
                show_q50 = st.checkbox("50%", value=True)
            with col_q4:
                show_q75 = st.checkbox("75%", value=True)
            with col_q5:
                show_q95 = st.checkbox("95%", value=True)
            
            copula_type = st.selectbox(
                "Tipo de cópula:",
                ["best", "gaussian", "clayton", "gumbel", "frank"],
                format_func=lambda x: {
                    'best': 'Mejor cópula (automático)',
                    'gaussian': 'Gaussiana',
                    'clayton': 'Clayton (cola inferior)',
                    'gumbel': 'Gumbel (cola superior)',
                    'frank': 'Frank (simétrica)'
                }[x]
            )
            
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
                dep = analyzer.calculate_dependence_measures(var_x, var_y)
                
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Pearson r", f"{dep['Pearson_r']:.4f}")
                with col_metrics[1]:
                    st.metric("Spearman ρ", f"{dep['Spearman_rho']:.4f}")
                with col_metrics[2]:
                    st.metric("Kendall τ", f"{dep['Kendall_tau']:.4f}")
                with col_metrics[3]:
                    diff = abs(dep['Spearman_rho'] - dep['Pearson_r'])
                    st.metric("|Spearman-Pearson|", f"{diff:.4f}")
                
                with st.spinner('Calculando regresión cuantil...'):
                    fig, quantile_results = analyzer.create_scatter_plot_with_quantiles(
                        var_x, var_y, quantiles, copula_type, add_regression, add_density
                    )
                st.pyplot(fig)
                
                st.markdown("### 📈 Información de la Cópula")
                cop = quantile_results['copula']
                st.info(f"""
                **Cópula seleccionada:** {cop['type'].capitalize()}
                
                **Parámetros:**
                {format_copula_params(cop)}
                
                **Interpretación:**
                {interpret_copula(cop)}
                """)
        
        # Tab 2: Estimación Multivariada
        with tab2:
            st.markdown('<div class="sub-header">🎯 Estimación Multivariada de Propiedades</div>', unsafe_allow_html=True)
            st.markdown("""
            Utilizando **múltiples variables predictoras** (Random Forest, Gradient Boosting, KNN) para estimar la propiedad objetivo.
            Este método multivariado captura relaciones complejas y no lineales, logrando **altos valores de R²**.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_var = st.selectbox(
                    "Variable objetivo (Y a estimar):", 
                    analyzer.available_cols,
                    index=min(3, len(analyzer.available_cols)-1),
                    key="target_multi"
                )
            
            with col2:
                # Excluir la variable objetivo de las predictoras
                feature_options = [c for c in analyzer.available_cols if c != target_var]
                default_features = feature_options[:min(4, len(feature_options))]
                
                selected_features = st.multiselect(
                    "Variables predictoras (X):",
                    feature_options,
                    default=default_features,
                    key="features_multi",
                    help="Selecciona múltiples variables para mejorar la estimación"
                )
            
            method = st.selectbox(
                "Método de estimación:",
                ["ensemble", "rf", "gb", "knn"],
                format_func=lambda x: {
                    'ensemble': 'Ensemble (RF + GB + KNN) - Recomendado',
                    'rf': 'Random Forest',
                    'gb': 'Gradient Boosting',
                    'knn': 'KNN-Copula'
                }[x]
            )
            
            test_size = st.slider("Tamaño de prueba (train/test split):", 0.1, 0.4, 0.2, 0.05)
            
            if target_var and len(selected_features) >= 1:
                with st.spinner(f'Estimando {target_var} usando {len(selected_features)} variables predictoras...'):
                    y_estimated, y_real, r2, rmse, mae, method_name, features_used, X_data = analyzer.estimate_multivariate(
                        target_var, selected_features, test_size, method
                    )
                
                # Mostrar métricas de éxito
                st.markdown("### 📊 Métricas de Precisión")
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("R²", f"{r2:.4f}", 
                             delta="Excelente" if r2 > 0.8 else "Bueno" if r2 > 0.6 else "Moderado",
                             delta_color="normal")
                with col_metrics[1]:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col_metrics[2]:
                    st.metric("MAE", f"{mae:.4f}")
                with col_metrics[3]:
                    correlation = np.corrcoef(y_estimated, y_real)[0, 1]
                    st.metric("Correlación", f"{correlation:.4f}")
                
                # Gráfico de comparación Real vs Estimado
                fig_comp = analyzer.create_estimation_plot(
                    target_var, y_estimated, y_real, method_name, features_used, r2, rmse, mae
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Gráfico con profundidad si está disponible
                if analyzer.depth_col:
                    st.markdown("### 📏 Comparación con Profundidad")
                    depth_data = analyzer.data_clean[analyzer.depth_col].values
                    fig_depth = analyzer.create_depth_plot(target_var, y_estimated, y_real, depth_data, r2)
                    st.plotly_chart(fig_depth, use_container_width=True)
                
                # Importancia de variables (para Random Forest)
                if method in ['rf', 'ensemble']:
                    st.markdown("### 🔍 Importancia de Variables")
                    try:
                        # Reentrenar RF para obtener importancia
                        X = analyzer.data_clean[selected_features].values
                        y = analyzer.data_clean[target_var].values
                        rf_temp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        rf_temp.fit(X, y)
                        
                        importance_df = pd.DataFrame({
                            'Variable': selected_features,
                            'Importancia': rf_temp.feature_importances_
                        }).sort_values('Importancia', ascending=True)
                        
                        fig_importance = px.bar(importance_df, x='Importancia', y='Variable', 
                                                orientation='h', title='Importancia de Variables Predictoras',
                                                color='Importancia', color_continuous_scale='Blues')
                        st.plotly_chart(fig_importance, use_container_width=True)
                    except:
                        pass
                
                # Mostrar correlaciones con la variable objetivo
                st.markdown("### 📈 Correlaciones con la Variable Objetivo")
                corr_with_target = []
                for feat in selected_features:
                    corr_val = analyzer.calculate_dependence_measures(feat, target_var)
                    corr_with_target.append({
                        'Variable': feat,
                        'Pearson r': corr_val['Pearson_r'],
                        'Spearman ρ': corr_val['Spearman_rho'],
                        'Kendall τ': corr_val['Kendall_tau']
                    })
                corr_df = pd.DataFrame(corr_with_target)
                st.dataframe(corr_df, use_container_width=True)
                
            else:
                if len(selected_features) == 0:
                    st.warning("Selecciona al menos una variable predictora para la estimación")
                else:
                    st.warning("Selecciona una variable objetivo")
        
        # Tab 3: Box Plots
        with tab3:
            st.markdown('<div class="sub-header">📊 Box Plots y Violin Plots</div>', unsafe_allow_html=True)
            
            selected_vars_box = st.multiselect(
                "Selecciona variables para visualizar:",
                analyzer.available_cols,
                default=analyzer.available_cols[:min(4, len(analyzer.available_cols))]
            )
            
            if selected_vars_box:
                plot_type = st.radio(
                    "Tipo de gráfico:",
                    ["Box Plot", "Violin Plot"],
                    horizontal=True
                )
                
                if plot_type == "Box Plot":
                    fig_box = analyzer.create_box_plots(selected_vars_box)
                    if fig_box:
                        st.plotly_chart(fig_box, use_container_width=True)
                else:
                    fig_violin = analyzer.create_violin_plots(selected_vars_box)
                    if fig_violin:
                        st.plotly_chart(fig_violin, use_container_width=True)
                
                # Estadísticas descriptivas
                st.markdown("### 📊 Estadísticas Descriptivas")
                stats_df = analyzer.data_clean[selected_vars_box].describe()
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.warning("Selecciona al menos una variable para visualizar")
        
        # Tab 4: Matriz de Correlación
        with tab4:
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
        
        # Tab 5: Matriz de Dispersión
        with tab5:
            st.markdown('<div class="sub-header">📊 Matriz de Dispersión</div>', unsafe_allow_html=True)
            
            selected_vars_matrix = st.multiselect(
                "Variables a incluir:",
                analyzer.available_cols,
                default=analyzer.available_cols[:min(4, len(analyzer.available_cols))],
                key="matrix_vars"
            )
            
            if len(selected_vars_matrix) >= 2:
                sample_size = st.slider("Tamaño de muestra:", 100, 1000, 500)
                fig_matrix = analyzer.create_pairplot_matrix(selected_vars_matrix, sample_size)
                if fig_matrix:
                    st.plotly_chart(fig_matrix, use_container_width=True)
            else:
                st.warning("Selecciona al menos 2 variables para generar la matriz de dispersión")
        
        # Tab 6: Tabla de Dependencias
        with tab6:
            st.markdown('<div class="sub-header">📋 Tabla de Dependencias</div>', unsafe_allow_html=True)
            
            selected_vars_table = st.multiselect(
                "Variables a analizar (dejar vacío para todas):",
                analyzer.available_cols,
                default=[],
                key="table_vars"
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
    
    else:
        st.info("👈 Por favor, carga un archivo CSV o usa los datos sintéticos para comenzar el análisis")
        
        st.markdown("### 📝 Formato esperado del archivo CSV")
        st.markdown("""
        El archivo CSV debe contener columnas numéricas con registros de pozo como:
        - **DEPTH**: Profundidad (opcional)
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


if __name__ == "__main__":
    main()
