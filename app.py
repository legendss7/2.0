import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import os
import datetime
import pydeck as pdk

# Configuración de la página
st.set_page_config(page_title="Dashboard Power BI Style", layout="wide")
st.title("📊 Dashboard Interactivo - Estilo Power BI")

# Directorio de configuración
config_dir = "configs"
os.makedirs(config_dir, exist_ok=True)

# Funciones para manejar configuraciones
def save_config(name, config):
    with open(os.path.join(config_dir, f"{name}.json"), "w") as f:
        json.dump(config, f)

def load_config(name):
    path = os.path.join(config_dir, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def list_configs():
    return [f[:-5] for f in os.listdir(config_dir) if f.endswith(".json")]

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, None)
        sheet = st.selectbox("Selecciona la hoja", list(df.keys()))
        df = df[sheet]
        
        # Vista previa de los datos
        st.write("Vista previa de datos:")
        st.dataframe(df.head())

        columns = df.columns.tolist()

        # Filtros y segmentadores
        st.sidebar.header("🔀 Segmentadores")
        segment_filters = {}
        for col in columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 20:
                selected_vals = st.sidebar.multiselect(f"{col}", unique_vals.tolist(), default=unique_vals.tolist())
                segment_filters[col] = selected_vals

        for col, vals in segment_filters.items():
            df = df[df[col].isin(vals)]

        # Filtros por texto
        st.sidebar.header("🔍 Filtros por texto")
        filters = {}
        for col in columns:
            val = st.sidebar.text_input(f"Filtrar por {col}")
            if val:
                filters[col] = val

        for col, val in filters.items():
            df = df[df[col].astype(str).str.lower() == val.lower()]

        # Personalización de gráficos
        st.sidebar.header("🎨 Personalización del gráfico")
        theme = st.sidebar.selectbox("Tema de colores", ["Default", "Plotly", "Seaborn", "GGPlot", "Simple White", "Dark"])
        template = {
            "Default": "plotly",
            "Plotly": "plotly",
            "Seaborn": "seaborn",
            "GGPlot": "ggplot2",
            "Simple White": "simple_white",
            "Dark": "plotly_dark",
        }[theme]

        x_axis = st.selectbox("Selecciona eje X", columns)
        y_axis = st.multiselect("Selecciona ejes Y", columns, default=columns[1:2])

        secondary_y = st.multiselect("Selecciona ejes Y secundarios (doble eje)", columns)

        chart_type = st.selectbox("Tipo de gráfico", [
            "Línea", "Barras", "Radar", "Dispersión", "Área", "Pastel", "Histograma",
            "Combinado", "Temporal doble eje", "Mapa Plotly", "Mapa DeckGL", "Treemap", "Gantt", "Boxplot", "Violin",
            "Heatmap", "Waterfall", "Funnel", "Sunburst", "Box Plot", "Scatter Plot"
        ])

        show_labels = st.checkbox("Mostrar etiquetas de datos")
        show_legend = st.checkbox("Mostrar leyenda", value=True)
        show_trendline = st.checkbox("Agregar línea de tendencia (solo Dispersión)")

        # Títulos personalizados
        st.sidebar.header("✏️ Personalización de títulos")
        chart_title = st.sidebar.text_input("Título del gráfico", value="Visualización de datos")
        x_title = st.sidebar.text_input("Título eje X", value=x_axis)
        y_title = st.sidebar.text_input("Título eje Y", value=", ".join(y_axis))

        # Gestión de configuraciones
        st.sidebar.header("💾 Gestión de configuraciones")
        config_name = st.sidebar.text_input("Nombre de configuración")
        if st.sidebar.button("💾 Guardar configuración") and config_name:
            config_data = {
                "x_axis": x_axis,
                "y_axis": y_axis,
                "chart_type": chart_type,
                "show_labels": show_labels,
                "show_legend": show_legend,
                "show_trendline": show_trendline,
                "chart_title": chart_title,
                "x_title": x_title,
                "y_title": y_title,
                "theme": theme
            }
            save_config(config_name, config_data)
            st.success(f"Configuración '{config_name}' guardada.")

        selected_config = st.sidebar.selectbox("📂 Cargar configuración", [""] + list_configs())
        if selected_config:
            config = load_config(selected_config)
            st.session_state.update(config)
            st.success(f"Configuración '{selected_config}' cargada.")

        # Visualización de gráficos
        st.subheader("📈 Visualización de datos")
        
        # Crear subgráficas si se usa doble eje
        fig = make_subplots(specs=[[{"secondary_y": bool(secondary_y)}]]) if secondary_y else go.Figure()

        for y in y_axis:
            if chart_type == "Línea":
                trace = go.Scatter(x=df[x_axis], y=df[y], name=y, mode="lines+markers" if show_labels else "lines")
            elif chart_type == "Barras":
                trace = go.Bar(x=df[x_axis], y=df[y], name=y, text=df[y] if show_labels else None, textposition="auto")
            elif chart_type == "Área":
                trace = go.Scatter(x=df[x_axis], y=df[y], name=y, fill="tozeroy")
            elif chart_type == "Dispersión":
                trace = go.Scatter(x=df[x_axis], y=df[y], name=y, mode="markers")
            elif chart_type == "Boxplot":
                trace = go.Box(x=df[x_axis], y=df[y], name=y)
            elif chart_type == "Violin":
                trace = go.Violin(x=df[x_axis], y=df[y], name=y, box_visible=True)
            elif chart_type == "Heatmap":
                trace = go.Heatmap(z=df[y], x=df[x_axis], y=y)
            elif chart_type == "Waterfall":
                trace = go.Waterfall(x=df[x_axis], y=df[y], name=y)
            elif chart_type == "Funnel":
                trace = go.Funnel(y=df[x_axis], x=df[y], name=y)
            elif chart_type == "Sunburst":
                trace = go.Sunburst(labels=df[x_axis], parents=["" for _ in df[x_axis]], values=df[y])
            elif chart_type == "Treemap":
                trace = go.Treemap(labels=df[x_axis], values=df[y], parents=["" for _ in df[x_axis]])
            else:
                trace = go.Scatter(x=df[x_axis], y=df[y], name=y)

            fig.add_trace(trace, secondary_y=(y in secondary_y))

        fig.update_layout(title=chart_title, template=template, showlegend=show_legend, xaxis_title=x_title, yaxis_title=y_title)
        st.plotly_chart(fig)

        # Resumen del gráfico
        if y_axis:
            summary = df[y_axis].describe().T
            st.markdown("### 📌 Resumen del gráfico")
            for col in summary.index:
                st.markdown(f"**{col}** - Media: {summary.loc[col, 'mean']:.2f}, Mediana: {summary.loc[col, '50%']:.2f}, Máx: {summary.loc[col, 'max']:.2f}, Mín: {summary.loc[col, 'min']:.2f}, Desv: {summary.loc[col, 'std']:.2f}")

        # KPIs
        st.subheader("📌 KPIs")
        kpi_cols = st.columns(len(y_axis) if len(y_axis) > 0 else 1)  # Solucionar el error de columnas vacías
        for i, y in enumerate(y_axis):
            try:
                col_data = pd.to_numeric(df[y], errors='coerce')
                avg = col_data.mean()
                total = col_data.sum()
                global_avg = pd.to_numeric(pd.read_excel(uploaded_file, sheet_name=sheet)[y], errors='coerce').mean()
                variation = ((avg - global_avg) / global_avg) * 100 if global_avg != 0 else 0
                kpi_cols[i].metric(label=f"{y} (media)", value=f"{avg:.2f}", delta=f"{variation:.2f}%")
            except Exception:
                st.warning(f"No se pudo calcular KPIs para {y}")

        st.subheader("📄 Exportar datos")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos filtrados')
        st.download_button(label="📅 Descargar Excel", data=output.getvalue(), file_name="App_Dashboard.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
