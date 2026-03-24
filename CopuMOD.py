"""
Streamlit App para Análisis de Dependencia en Datos de Pozo
===========================================================
Aplicación interactiva para explorar relaciones entre registros de pozo
con heatmaps, scatter plots y matriz de dispersión.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, kendalltau, spearmanr, pearsonr
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
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
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class WellLogDependenceAnalyzer:
    """
    Clase para análisis de dependencia de datos de pozo
    """
    
    def __init__(self, data):
        self.data_original = data.copy()
        self.data_clean = self._clean_data(data)
        self.available_cols = self.data_clean.columns.tolist()
        self.correlation_matrices = {}
        
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
    
    def create_scatter_plot(self, col1, col2, add_regression=True, add_density=True):
        """Crea scatter plot interactivo con Plotly"""
        x = self.data_clean[col1].values
        y = self.data_clean[col2].values
        dep = self.calculate_dependence_measures(col1, col2)
        
        fig = go.Figure()
        
        # Scatter plot principal
        if add_density:
            # Calcular densidad para colorear
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
                    opacity=0.7
                ),
                text=[f'{col1}: {xi:.2f}<br>{col2}: {yi:.2f}' for xi, yi in zip(x, y)],
                hoverinfo='text',
                name='Datos'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(size=8, color='steelblue', opacity=0.6),
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
        
        # Regresión no lineal (spline)
        try:
            idx = np.argsort(x)
            x_sorted = x[idx]
            y_sorted = y[idx]
            s = len(x_sorted) * 0.05
            spline = UnivariateSpline(x_sorted, y_sorted, s=s)
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
            y_smooth = spline(x_smooth)
            
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth,
                mode='lines',
                line=dict(color='green', width=2),
                name='Tendencia no lineal'
            ))
        except:
            pass
        
        # Configuración del layout
        fig.update_layout(
            title=f'{col1} vs {col2}',
            xaxis_title=col1,
            yaxis_title=col2,
            hovermode='closest',
            width=800,
            height=600,
            showlegend=True
        )
        
        # Añadir anotaciones con estadísticas
        annotation_text = (
            f"Pearson r = {dep['Pearson_r']:.4f}<br>"
            f"Spearman ρ = {dep['Spearman_rho']:.4f}<br>"
            f"Kendall τ = {dep['Kendall_tau']:.4f}<br>"
            f"R² = {dep['Pearson_R2']:.4f}"
        )
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def create_rank_plot(self, col1, col2):
        """Crea rank plot (P-P plot) en espacio uniforme"""
        u = stats.rankdata(self.data_clean[col1]) / (len(self.data_clean) + 1)
        v = stats.rankdata(self.data_clean[col2]) / (len(self.data_clean) + 1)
        
        fig = go.Figure()
        
        # Scatter de rangos
        fig.add_trace(go.Scatter(
            x=u, y=v,
            mode='markers',
            marker=dict(size=6, color='steelblue', opacity=0.6),
            name='Rangos'
        ))
        
        # Línea de independencia
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Independencia'
        ))
        
        fig.update_layout(
            title=f'Rank Plot: {col1} vs {col2}',
            xaxis_title=f'Rank({col1})',
            yaxis_title=f'Rank({col2})',
            width=600,
            height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_heatmap(self, correlation_type='spearman', figsize=(10, 8)):
        """Crea heatmap de correlación con matplotlib (para mayor control)"""
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
            "📈 Scatter Plot", 
            "🔥 Matriz de Correlación", 
            "📊 Matriz de Dispersión",
            "📋 Tabla de Dependencias",
            "ℹ️ Información del Dataset"
        ])
        
        # Tab 1: Scatter Plot Interactivo
        with tab1:
            st.markdown('<div class="sub-header">📈 Scatter Plot Interactivo</div>', unsafe_allow_html=True)
            st.markdown("Selecciona las variables para visualizar su relación")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                var_x = st.selectbox("Variable X:", analyzer.available_cols, index=0)
            with col2:
                var_y = st.selectbox("Variable Y:", analyzer.available_cols, index=min(1, len(analyzer.available_cols)-1))
            with col3:
                add_regression = st.checkbox("Mostrar regresión", value=True)
                add_density = st.checkbox("Mostrar densidad", value=True)
            
            if var_x and var_y:
                fig_scatter = analyzer.create_scatter_plot(var_x, var_y, add_regression, add_density)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Rank plot
                st.markdown("### 📊 Rank Plot (Espacio Uniforme)")
                fig_rank = analyzer.create_rank_plot(var_x, var_y)
                st.plotly_chart(fig_rank, use_container_width=True)
                
                # Mostrar medidas de dependencia
                dep = analyzer.calculate_dependence_measures(var_x, var_y)
                st.markdown("### 📈 Medidas de Dependencia")
                
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Pearson r", f"{dep['Pearson_r']:.4f}")
                with col_metrics[1]:
                    st.metric("Spearman ρ", f"{dep['Spearman_rho']:.4f}")
                with col_metrics[2]:
                    st.metric("Kendall τ", f"{dep['Kendall_tau']:.4f}")
                with col_metrics[3]:
                    st.metric("R²", f"{dep['Pearson_R2']:.4f}")
                
                # Interpretación
                st.markdown("### 💡 Interpretación")
                non_linearity = abs(dep['Spearman_rho'] - dep['Pearson_r'])
                
                if non_linearity > 0.2:
                    st.warning(f"⚠️ Posible relación no lineal detectada (|Spearman - Pearson| = {non_linearity:.3f} > 0.2)")
                else:
                    st.info(f"✅ Relación principalmente lineal (|Spearman - Pearson| = {non_linearity:.3f})")
                
                if abs(dep['Kendall_tau']) > 0.7:
                    st.success("✅ Fuerte concordancia entre las variables")
                elif abs(dep['Kendall_tau']) > 0.3:
                    st.info("📊 Concordancia moderada entre las variables")
                else:
                    st.warning("⚠️ Débil concordancia entre las variables")
        
        # Tab 2: Matriz de Correlación
        with tab2:
            st.markdown('<div class="sub-header">🔥 Matriz de Correlación</div>', unsafe_allow_html=True)
            
            # Selector de tipo de correlación
            corr_type = st.selectbox(
                "Tipo de correlación:",
                ["spearman", "pearson", "kendall"],
                format_func=lambda x: {
                    'pearson': 'Pearson (Correlación Lineal)',
                    'spearman': 'Spearman (Correlación de Rangos)',
                    'kendall': "Kendall τ (Concordancia)"
                }[x]
            )
            
            # Generar heatmap
            fig_heatmap = analyzer.create_heatmap(corr_type, figsize=(10, 8))
            st.pyplot(fig_heatmap)
            
            # Mostrar matriz numérica
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
            st.markdown("Selecciona las variables para incluir en la matriz de dispersión")
            
            # Selección de variables para la matriz
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
                    st.warning("No se pudo generar la matriz de dispersión")
            else:
                st.warning("Selecciona al menos 2 variables para generar la matriz de dispersión")
        
        # Tab 4: Tabla de Dependencias
        with tab4:
            st.markdown('<div class="sub-header">📋 Tabla de Dependencias</div>', unsafe_allow_html=True)
            
            # Selección de variables para la tabla
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
                
                # Botón para descargar
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
            
            # Estadísticas descriptivas
            st.markdown("### 📊 Estadísticas Descriptivas")
            st.dataframe(analyzer.get_data_summary(), use_container_width=True)
            
            # Resumen de correlaciones
            st.markdown("### 📈 Resumen de Correlaciones (Top 10)")
            corr_summary = analyzer.get_correlation_summary()
            if not corr_summary.empty:
                # Ordenar por valor absoluto de Spearman
                corr_summary['abs_Spearman'] = corr_summary['Spearman'].abs()
                corr_summary = corr_summary.sort_values('abs_Spearman', ascending=False).head(10)
                corr_summary = corr_summary.drop('abs_Spearman', axis=1)
                st.dataframe(corr_summary, use_container_width=True)
            
            # Información de calidad de datos
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
        # Mensaje cuando no hay datos
        st.info("👈 Por favor, carga un archivo CSV o usa los datos sintéticos para comenzar el análisis")
        
        # Mostrar ejemplo
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


if __name__ == "__main__":
    main()
