"""Dashboard interactivo de demanda de ride-hailing en Chicago.

Capa "Use" del pipeline Big Data del proyecto (ver notebooks/ y README.md).
Consume la muestra analítica exportada por el pipeline PySpark
(data/final_dataset.csv) y las métricas reales de los modelos
(data/model_metrics.csv, generado por src/train_models.py).
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import vl_convert as vlc

DATA_DIR = Path(__file__).parent / "data"

# day_of_week viene del pipeline Spark con la convención dayofweek():
# 1=Domingo, 2=Lunes, ..., 7=Sábado. Se muestra siempre con nombres,
# ordenados de lunes a domingo.
DAY_NAMES = {
    1: "Domingo",
    2: "Lunes",
    3: "Martes",
    4: "Miércoles",
    5: "Jueves",
    6: "Viernes",
    7: "Sábado",
}
DAY_ORDER = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

st.set_page_config(
    page_title="Ride-Hailing Demand Intelligence",
    page_icon="🚖",
    layout="wide",
)


# =========================
# CARGA DE DATOS (cacheada)
# =========================
@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "final_dataset.csv")
    df["day_name"] = df["day_of_week"].map(DAY_NAMES)
    return df


@st.cache_data
def load_geojson() -> dict:
    with open(DATA_DIR / "chicago_geo.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_metrics.csv")


@st.cache_data(show_spinner=False)
def choropleth_png(demand_items: tuple) -> bytes:
    """Renderiza el mapa coroplético de demanda por community area a PNG.

    Incrustamos la demanda media en las propiedades del GeoJSON y renderizamos
    con vl-convert en el servidor en lugar de dejarlo al Vega del navegador:
    el geoshape con GeoJSON inline no compone de forma fiable en el Vega que
    embebe Streamlit, y así el mapa se ve idéntico en local y en producción sin
    depender de ningún CDN de tiles. Cacheado por los valores de demanda.
    """
    demand = dict(demand_items)
    features = [
        {
            **feat,
            "properties": {
                **feat["properties"],
                "demand": demand.get(int(feat["properties"]["area_num_1"])),
            },
        }
        for feat in load_geojson()["features"]
    ]

    chart = (
        alt.Chart(alt.Data(values=features))
        .mark_geoshape(stroke="white", strokeWidth=0.4)
        .encode(
            color=alt.Color(
                "properties.demand:Q",
                scale=alt.Scale(scheme="yelloworangered"),
                title="Demanda media",
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("properties.community:N", title="Zona"),
                alt.Tooltip("properties.demand:Q", title="Demanda media", format=".1f"),
            ],
        )
        .project(type="mercator")
        .properties(width=720, height=560)
        .configure_view(strokeWidth=0)
    )
    return vlc.vegalite_to_png(chart.to_json(), scale=2)


df = load_dataset()

# =========================
# SIDEBAR FILTROS
# =========================
st.sidebar.header("🎛️ Filtros")

available_days = [d for d in DAY_ORDER if d in set(df["day_name"])]
selected_days = st.sidebar.multiselect(
    "Día de la semana",
    options=available_days,
    default=available_days,
)

selected_hours = st.sidebar.slider("Rango horario", 0, 23, (0, 23))

all_zones = sorted(df["zone_name"].dropna().unique())
selected_zones = st.sidebar.multiselect(
    "Zonas",
    options=all_zones,
    default=all_zones,
)

df_filtered = df[
    (df["day_name"].isin(selected_days))
    & (df["hour"].between(selected_hours[0], selected_hours[1]))
    & (df["zone_name"].isin(selected_zones))
]

if df_filtered.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# =========================
# PORTADA
# =========================
st.title("🚖 Ride-Hailing Demand Intelligence")

st.markdown("""
Este dashboard analiza la demanda de transporte en Chicago integrando variables:
- ⏱️ Temporales
- 🌍 Espaciales
- 🌦️ Climáticas

👉 Objetivo: entender patrones y optimizar decisiones operativas.
""")

# =========================
# KPIs DINÁMICOS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros", f"{len(df_filtered):,}")
col2.metric("Demanda promedio", f"{df_filtered['demand'].mean():.2f}")
col3.metric("Temp media", f"{df_filtered['temperature'].mean():.1f}°C")
col4.metric("Zonas activas", df_filtered["pickup_community_area"].nunique())

st.caption(f"📆 Periodo de la muestra: {df['date'].min()} → {df['date'].max()} ({df['date'].nunique()} días)")

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📘 Resumen",
    "📊 Exploración",
    "🗺️ Espacial",
    "🌦️ Variables",
    "🤖 Modelos",
    "🧪 Feature Engineering",
    "📌 Conclusiones",
])

# ==========================================================
# TAB 1 — RESUMEN
# ==========================================================
with tab1:
    st.header("📘 Contexto del proyecto")

    st.markdown("""
Este proyecto integra múltiples fuentes de datos para modelar la demanda de ride-hailing:

### 🔧 Pipeline
- **Capture:** Taxi Trips, NOAA Weather, Holidays
- **Ingest:** Limpieza con PySpark
- **Store:** Data Lake (Parquet)
- **Compute:** Feature Engineering + ML
- **Use:** Dashboard interactivo

### 🎯 Problema
La demanda es altamente variable → requiere modelos predictivos robustos.

### 💡 Enfoque
Se construyó un dataset analítico combinando:
- Tiempo (hora, día)
- Espacio (zona)
- Clima (temperatura)
- Eventos (festivos)

📓 El pipeline completo (descarga de datos, Data Lake, Spark y modelado)
está documentado en el notebook del repositorio: `notebooks/ride_hailing_demand_pipeline.ipynb`.
""")

# ==========================================================
# TAB 2 — EXPLORACIÓN
# ==========================================================
with tab2:
    st.header("📊 Exploración de la demanda")

    st.markdown("""
Analizamos cómo se distribuye la demanda en el tiempo y el espacio,
identificando patrones clave de movilidad urbana.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏱️ Demanda promedio por hora")

        chart_hour = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X("hour:O", title="Hora del día"),
            y=alt.Y("mean(demand):Q", title="Demanda promedio"),
        )

        st.altair_chart(chart_hour, use_container_width=True)

        st.caption("Se observan picos en horas punta → comportamiento típico de commuting urbano.")

    with col2:
        st.subheader("📅 Demanda por día de la semana")

        chart_day = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X("day_name:N", title="Día de la semana", sort=DAY_ORDER,
                    scale=alt.Scale(domain=DAY_ORDER)),
            y=alt.Y("mean(demand):Q", title="Demanda promedio"),
        )

        st.altair_chart(chart_day, use_container_width=True)

        caption_day = "Permite comparar patrones entre días laborales y fines de semana."
        missing_days = [d for d in DAY_ORDER if d not in set(df["day_name"])]
        if missing_days:
            caption_day += (
                f" La muestra actual ({df['date'].min()} → {df['date'].max()}) "
                f"no incluye datos de: {', '.join(missing_days)}."
            )
        st.caption(caption_day)

    st.markdown("---")

    st.subheader("🔥 Mapa de calor: día vs hora")

    heatmap_data = df_filtered.pivot_table(
        index="day_name",
        columns="hour",
        values="demand",
        aggfunc="mean",
    )

    heatmap_long = heatmap_data.reset_index().melt(
        id_vars="day_name",
        var_name="hour",
        value_name="avg_demand",
    )

    chart_heatmap = alt.Chart(heatmap_long).mark_rect().encode(
        x=alt.X("hour:O", title="Hora"),
        y=alt.Y("day_name:N", title="Día", sort=DAY_ORDER,
                scale=alt.Scale(domain=DAY_ORDER)),
        color=alt.Color("avg_demand:Q", scale=alt.Scale(scheme="reds"), title="Demanda media"),
    )

    st.altair_chart(chart_heatmap, use_container_width=True)

    pico = heatmap_data.stack().idxmax()
    pico_val = heatmap_data.stack().max()

    st.caption(f"Pico de demanda: {pico[0]} a las {pico[1]}h → {pico_val:.2f} viajes/hora/zona")

    st.markdown("""
📌 **Insight:**
La demanda no es uniforme → existen ventanas críticas de alta concentración.
""")

# ==========================================================
# TAB 3 — ESPACIAL
# ==========================================================
with tab3:
    st.subheader("🏙️ Top 15 zonas por demanda")

    top_zones = (
        df_filtered.groupby("zone_name")["demand"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    chart_zones = alt.Chart(top_zones).mark_bar().encode(
        x=alt.X("demand:Q", title="Demanda promedio"),
        y=alt.Y("zone_name:N", sort="-x", title="Zona"),
        color=alt.value("steelblue"),
    )

    st.altair_chart(chart_zones, use_container_width=True)

    st.caption("""
Las zonas con mayor actividad reflejan concentración de demanda
y posibles puntos críticos operativos.
""")

    st.markdown("---")

    st.subheader("🌍 Mapa coroplético de demanda por community area")

    st.markdown("Se observa un clúster central dominante y zonas periféricas con menor actividad.")

    demand_by_area = (
        df_filtered.groupby("pickup_community_area")["demand"].mean().round(2).to_dict()
    )

    st.image(choropleth_png(tuple(sorted(demand_by_area.items()))), use_container_width=True)

    st.info("""
La visualización revela concentración espacial de la demanda,
lo que sugiere oportunidades de optimización operativa y posibles sesgos de cobertura.
""")

# ==========================================================
# TAB 4 — VARIABLES
# ==========================================================
with tab4:
    st.header("🌦️ Impacto de variables explicativas")

    st.markdown("Analizamos cómo factores externos influyen en la demanda de transporte.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Temperatura vs Demanda")

        chart_temp = alt.Chart(df_filtered).mark_circle(size=40).encode(
            x=alt.X("temperature:Q", title="Temperatura (°C)"),
            y=alt.Y("demand:Q", title="Demanda"),
            color=alt.Color("day_name:N", title="Día semana", sort=DAY_ORDER),
            tooltip=["temperature", "demand", "day_name"],
        ).interactive()

        st.altair_chart(chart_temp, use_container_width=True)

        st.caption("""
La temperatura influye en la movilidad:
- Climas extremos → menor demanda
- Climas moderados → mayor actividad
""")

    with col2:
        st.subheader("📅 Festivos vs Demanda")

        if "is_holiday" in df_filtered.columns:
            chart_holiday = alt.Chart(df_filtered).mark_bar().encode(
                x=alt.X("is_holiday:O", title="Es festivo"),
                y=alt.Y("mean(demand):Q", title="Demanda promedio"),
                color=alt.Color("is_holiday:N", legend=None),
            )

            st.altair_chart(chart_holiday, use_container_width=True)

            st.caption("""
Los festivos alteran el patrón de movilidad:
- Mayor concentración en zonas recreativas
- Menor patrón commuting
""")
        else:
            st.info("La columna 'is_holiday' no está disponible.")

    # Variables de tráfico (V3): solo si el export del pipeline las incluye
    if {"n_crashes", "congestion_speed"} <= set(df_filtered.columns):
        st.markdown("---")
        st.subheader("🚧 Variables de tráfico (V3)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Siniestros viales vs demanda**")

            df_crash = df_filtered.assign(
                con_siniestro=lambda d: (d["n_crashes"] > 0).map({True: "Con siniestro", False: "Sin siniestro"})
            )
            chart_crash = alt.Chart(df_crash).mark_bar().encode(
                x=alt.X("con_siniestro:N", title="Franja zona-hora"),
                y=alt.Y("mean(demand):Q", title="Demanda promedio"),
                color=alt.Color("con_siniestro:N", legend=None),
            )
            st.altair_chart(chart_crash, use_container_width=True)

            st.caption("""
Compara la demanda media entre franjas zona-hora con y sin siniestros registrados.
""")

        with col2:
            st.markdown("**Congestión (velocidad media) vs demanda**")

            df_cong = (
                df_filtered.groupby("hour")
                .agg(demanda=("demand", "mean"), velocidad=("congestion_speed", "mean"))
                .reset_index()
            )
            base_cong = alt.Chart(df_cong).encode(x=alt.X("hour:O", title="Hora"))
            chart_cong = alt.layer(
                base_cong.mark_line(point=True, color="#d62728").encode(
                    y=alt.Y("demanda:Q", title="Demanda promedio"),
                ),
                base_cong.mark_line(point=True, color="#1f77b4", strokeDash=[4, 3]).encode(
                    y=alt.Y("velocidad:Q", title="Velocidad media (mph)"),
                ),
            ).resolve_scale(y="independent")
            st.altair_chart(chart_cong, use_container_width=True)

            st.caption("""
En horas punta la demanda sube y la velocidad media baja → correlación negativa
esperable entre congestión y demanda.
""")

    st.markdown("""
📌 **Insight clave:**
Las variables externas no solo afectan la demanda, sino que introducen variabilidad
→ justificando su inclusión en modelos predictivos.
""")

# ==========================================================
# TAB 5 — MODELOS
# ==========================================================
with tab5:
    st.header("🤖 Modelos predictivos")

    st.markdown("""
Se evaluaron tres modelos con **split temporal** (el último tramo del periodo
se reserva como test: la última semana completa si la muestra lo permite,
el último día en caso contrario) y features de retardo (`lag_1h`, `lag_24h`)
calculadas zona a zona. Las métricas son reales y reproducibles con
`python src/train_models.py`.
""")

    metrics = load_metrics()
    metrics_lags = metrics[metrics["Escenario"] == "con_lags"].drop(columns="Escenario")

    st.dataframe(metrics_lags.set_index("Modelo").style.format("{:.2f}"))

    chart_metrics = alt.Chart(metrics_lags).transform_fold(
        ["MAE", "RMSE"],
        as_=["Métrica", "Valor"],
    ).mark_bar().encode(
        x=alt.X("Modelo:N", title=None),
        y=alt.Y("Valor:Q", title="Error (viajes/hora/zona)"),
        color=alt.Color("Métrica:N"),
        xOffset="Métrica:N",
    ).properties(height=400)

    st.altair_chart(chart_metrics, use_container_width=True)

    best = metrics_lags.sort_values("MAE").iloc[0]
    st.caption(f"""
**{best['Modelo']}** obtiene el mejor MAE ({best['MAE']:.2f} viajes/hora/zona).
El MAPE se muestra en la tabla pero debe interpretarse con cautela: muchas
combinaciones zona-hora tienen demanda cercana a cero, lo que infla el error porcentual.
""")

# ==========================================================
# TAB 6 — FEATURE ENGINEERING
# ==========================================================
with tab6:
    st.header("🧪 Estudio comparativo: impacto de las features de lag")

    st.markdown("""
Se entrenó una segunda versión de los tres modelos **eliminando** las features
de retardo (`lag_1h`, `lag_24h`) sobre exactamente el mismo train/test,
para aislar su aporte real a la precisión.
""")

    metrics = load_metrics()
    df_lag = metrics.pivot(index="Modelo", columns="Escenario", values="MAE").reset_index()
    df_lag = df_lag.rename(columns={"con_lags": "Con lags", "sin_lags": "Sin lags"})

    chart_lag = alt.Chart(df_lag).transform_fold(
        ["Sin lags", "Con lags"],
        as_=["Escenario", "MAE"],
    ).mark_bar().encode(
        x=alt.X("Modelo:N", title=None),
        y=alt.Y("MAE:Q", title="MAE (viajes/hora/zona)"),
        color=alt.Color("Escenario:N"),
        xOffset="Escenario:N",
    ).properties(height=400)

    st.altair_chart(chart_lag, use_container_width=True)

    st.caption("""
La demanda pasada es el mejor predictor de la demanda futura: incluir los lags
reduce el MAE de forma consistente, con especial impacto en Linear Regression
y Gradient Boosting.
""")

# ==========================================================
# TAB 7 — CONCLUSIONES
# ==========================================================
with tab7:
    st.header("📌 Conclusiones")

    st.markdown("""
### 🔍 Hallazgos clave del proyecto

1. La **demanda presenta patrones espacio-temporales claros**: picos en horas punta y concentración en el clúster central de Chicago.
2. **O'Hare** funciona como un polo aislado de altísima demanda, desconectado del clúster urbano.
3. Las **variables meteorológicas** (temperatura) y los **festivos** influyen significativamente en la movilidad.
4. El modelo **Gradient Boosting** fue el más preciso en MAE y RMSE.
5. Las features de retardo (**lag_1h**, **lag_24h**) son las que más aportan a la predicción.
6. Existe un **sesgo espacial**: el modelo refuerza zonas de alta renta y turismo, dejando infrarepresentado el sur de Chicago.

---

### ⚖️ Consideraciones

El modelo puede amplificar desigualdades espaciales →
se recomienda integrar criterios de equidad.
Esto abre un debate sobre **equidad espacial** y la necesidad de criterios de cobertura mínima en despliegues reales.

---

### 🚀 Aplicación

- Optimización de flotas
- Planificación urbana
- Reducción de tiempos de espera

Este dashboard permite explorar patrones de demanda y apoyar decisiones operativas en transporte urbano, integrando datos de movilidad, clima y calendario.
Su uso responsable debe considerar tanto la eficiencia como la equidad territorial.
""")
