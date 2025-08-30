# app.py — PCA + Clustering + Clasificación (presentación Streamlit, versión con Caracterización PC)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import lightgbm as lgb

# ================== Config general & estilo ==================
st.set_page_config(page_title="Fenotipos Hemodinámicos (PCA)", page_icon="🧬", layout="wide")
st.title("🧬 Fenotipos Hemodinámicos con PCA: Clustering y Clasificación")

# Tabs fijas (sticky) al hacer scroll
STICKY_TABS_CSS = """
<style>
.stTabs [role="tablist"]{
  position: sticky; top: 0; z-index: 999;
  background: var(--background-color);
  padding-top: .5rem; margin-top: -0.5rem;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}
.block-container { padding-top: 0.5rem; }
</style>
"""
st.markdown(STICKY_TABS_CSS, unsafe_allow_html=True)

# ================== Sidebar: Fuente de datos y parámetros ==================
st.sidebar.header("⚙️ Datos & Parámetros")

DATA_SOURCE = st.sidebar.radio(
    "Fuente de datos (componentes PCA)",
    ["Auto (archivo local)", "Subir CSV"],
    index=0,
)
DEFAULT_PATHS = [Path("datos_final7_pca_components.csv"), Path("data/datos_final7_pca_components.csv")]

@st.cache_data(show_spinner=False)
def load_csv_from_path(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def load_csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)

df = None
used_source_desc = ""

if DATA_SOURCE == "Auto (archivo local)":
    found = next((p for p in DEFAULT_PATHS if p.exists()), None)
    if found is not None:
        df = load_csv_from_path(found)
        used_source_desc = f"📁 Cargado automáticamente desde `{found}`"
    else:
        st.warning("No se encontró `datos_final7_pca_components.csv` (ni en `./` ni en `./data/`). "
                   "Selecciona **Subir CSV** en la barra lateral.")
else:  # Subir CSV
    uploaded_file = st.sidebar.file_uploader("Sube tu CSV (componentes PCA)", type=["csv"])
    if uploaded_file:
        df = load_csv_from_upload(uploaded_file)
        used_source_desc = "⬆️ Cargado por subida de archivo"

if df is None:
    st.stop()

st.success(f"✅ Datos cargados. {used_source_desc}")


# ======== Detección de columnas PCA y elección de PCs a usar ========
def detectar_pc_cols(_df: pd.DataFrame):
    pc_named = [c for c in _df.columns if str(c).upper().startswith("PC")]
    if pc_named:
        def pc_order(c):
            s = ''.join(ch for ch in str(c) if ch.isdigit())
            return int(s) if s.isdigit() else 9999
        return sorted(pc_named, key=pc_order)
    return [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]

pc_cols_all = detectar_pc_cols(df)
if len(pc_cols_all) < 2:
    st.error("No se detectaron suficientes columnas de componentes principales (PCs).")
    st.stop()

default_pcs = [c for c in pc_cols_all if any(c.upper() == f"PC{i}" for i in range(1, 6))]
if not default_pcs:
    default_pcs = pc_cols_all[:5] if len(pc_cols_all) >= 5 else pc_cols_all

pcs_to_use = st.sidebar.multiselect(
    "PCs a usar para clustering y modelos",
    options=pc_cols_all,
    default=default_pcs,
)
if len(pcs_to_use) < 2:
    st.warning("Selecciona al menos 2 componentes para proceder.")
    st.stop()

# Parámetros de clustering/modelado
k_optimo = st.sidebar.slider("Número de Clusters (k)", 2, 10, 4)
test_size = st.sidebar.slider("Proporción Test", 0.1, 0.5, 0.25, step=0.05)
random_state = 42

# ======== Preparación base ========
df_pca = df.copy()
df_pca = df_pca.dropna(subset=pcs_to_use)  # PCA suele venir limpio
X_pca_all = df_pca[pc_cols_all].values
X_pca = df_pca[pcs_to_use].values

# ================== NUEVO MÓDULO 1: Caracterización PC ==================
# Matriz de loadings tomada de tu gráfico (PC1..PC8 x variables hemodinámicas)
LOADINGS_DEFAULT = pd.DataFrame(
    data=[
        [-0.10, -0.03,  0.49,  0.19,  0.49,  0.30,  0.43,  0.45],  # PC1
        [ 0.53,  0.50,  0.06,  0.46,  0.06,  0.40, -0.25, -0.20],  # PC2
        [ 0.44,  0.53, -0.03, -0.47,  0.02, -0.37,  0.30,  0.29],  # PC3
        [-0.10,  0.05, -0.50, -0.45,  0.41,  0.51, -0.27,  0.17],  # PC4
        [ 0.66, -0.64, -0.10,  0.07,  0.13, -0.13, -0.16,  0.27],  # PC5
        [-0.27,  0.24, -0.16,  0.40,  0.16, -0.42, -0.46,  0.52],  # PC6
        [-0.00,  0.00, -0.40,  0.24,  0.60, -0.32,  0.35, -0.45],  # PC7
        [ 0.00, -0.00,  0.56, -0.34,  0.43, -0.23, -0.49, -0.32],  # PC8
    ],
    index=[f"PC{i}" for i in range(1, 9)],
    columns=[
        "fc_mean","fc_night_mean","sbp_mean","dbp_mean",
        "sbp_night_mean","dbp_night_mean","pulse_pressure_mean","pulse_pressure_night_mean"
    ],
)



# ================== Tabs  ==================
tabA, tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 Caracterización PC",
    "👀 Datos",
    "📊 Diagnóstico & Clustering",
    "🔎 Exploración en PCA",
    "🤖 Modelado (sobre PCs)",
    "📈 Evaluación",
    "🏆 Conclusiones",
])

# ============ TAB A: Caracterización PC ============
with tabA:
    st.subheader("🧩 PCA Loadings Heatmap (excluyendo edad, IMC y género)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        LOADINGS_DEFAULT, annot=True, fmt=".2f", cmap="viridis",
        vmin=-0.7, vmax=0.7, cbar_kws={"label":"Loading"}, ax=ax
    )
    ax.set_xlabel("Variables originales"); ax.set_ylabel("Componentes principales")
    ax.set_title("PCA Loadings Heatmap")
    st.pyplot(fig)
    st.caption("Los *loadings* indican la contribución (y signo) de cada variable original a cada componente principal.")

    st.markdown("""
### Caracterización de los componentes principales (PC1–PC5)

- **PC1 (Componente de "Presión de Pulso y Sistólica"):**  
  → Contribución alta y positiva: pulse_pressure_mean, pulse_pressure_night_mean, sbp_mean, sbp_night_mean. Significado: Sigue siendo el componente principal de la hipertensión sistólica y de pulso elevado. Un valor alto aquí indica presiones "altas" muy fuertes.

- **PC2 (Componente de "Presión Diastólica y Frecuencia Cardíaca"):**  
  → Contribución alta y positiva: dbp_mean, dbp_night_mean, fc_mean, fc_night_mean. Significado: Ahora representa de forma más clara la hipertensión diastólica y la taquicardia. Un valor alto en PC2 significa DBP y FC elevadas.

- **PC3 (Componente de "Frecuencia Cardíaca vs. Presión de Pulso"):**
  → Contribución alta y positiva: fc_mean y fc_night_mean (Frecuencia Cardíaca).

  → Contribución negativa: pulse_pressure_mean y pulse_pressure_night_mean (Presión de Pulso). Significado Clínico: Este componente representa un contraste o una relación inversa. Un valor alto y positivo en PC3 describe a pacientes que tienen una frecuencia cardíaca elevada pero, al mismo tiempo, una presión de pulso relativamente baja. Un valor bajo y negativo en PC3 describiría el perfil opuesto: una presión de pulso alta con una frecuencia cardíaca más controlada.

- **PC4 (Componente de "Patrón Nocturno Anormal - No Dipper"):**  
  → Contribución alta y positiva: dbp_night_mean y sbp_night_mean (Presiones Nocturnas).

  → Contribución alta y negativa: dbp_mean y sbp_mean (Presiones Diurnas). Significado Clínico: Este es un componente muy importante en el análisis de MAPA. Representa el patrón circadiano de la presión arterial. Un valor alto y positivo en PC4 identifica claramente a los pacientes con un patrón "Non-Dipper" (o "Riser"). Es decir, su presión arterial es más alta durante la noche que durante el día, lo cual es un indicador de mayor riesgo cardiovascular.

- **PC5 (Componente de "Patrón de Pulso Diurno - Dipper Fuerte"):** 
  → Contribución alta y positiva: pulse_pressure_mean y sbp_mean (Presión de Pulso y Sistólica Diurnas).

  → Contribución alta y negativa: pulse_pressure_night_mean y sbp_night_mean (Presión de Pulso y Sistólica Nocturnas). Significado Clínico: Este componente es el opuesto al PC4, pero enfocado en la presión de pulso y sistólica. Un valor alto y positivo en PC5 describe a los pacientes con un patrón "Dipper" muy marcado. Tienen una presión de pulso y sistólica diurna significativamente más alta que la nocturna, lo que indica un descenso nocturno saludable y pronunciado.


Mientras que PC1 y PC2 dan la severidad y el tipo general de hipertensión (Sistólica vs. Diastólica/FC), los componentes PC3, PC4 y PC5 permiten añadir una capa de detalle clínico mucho más rica, describiendo las relaciones entre variables (FC vs. Pulso) y los patrones circadianos (Dipper vs. Non-Dipper), que son fundamentales en el diagnóstico y pronóstico de la hipertensión.

    """)

# ============ TAB 0: Datos ============
with tab0:
    st.subheader("👀 Vista previa de los datos (componentes PCA)")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Filas", f"{df_pca.shape[0]:,}")
    with c2: st.metric("Columnas", f"{df_pca.shape[1]:,}")
    with c3: st.metric("PCs detectados", f"{len(pc_cols_all)}")
    st.caption(f"PCs seleccionados para el flujo: {pcs_to_use}")
    st.dataframe(df_pca.head(50), use_container_width=True)


    st.markdown("""
###  Definición del Problema y Descubrimiento de Fenotipos con Clustering
**Objetivo:** Nuestro primer paso es descubrir estos grupos latentes en los datos. Usaremos K-Means para agrupar a los pacientes y cada cluster será considerado un "fenotipo hemodinámico". El resultado de este paso será nuestro DataFrame con una nueva columna fenotipo que usaremos como variable objetivo. 
    """)

# ============ TAB 1: Diagnóstico & Clustering ============
with tab1:
    st.subheader("📊 Diagnóstico: Codo e 'Accuracy' de K-Means")

    # --- Parámetro para la "verdad" de referencia (ground truth) ---
    # Usamos por defecto el mismo k que elegiste en el sidebar (k_optimo),
    # pero puedes cambiarlo aquí para comparar.
    k_ref = st.number_input(
        "k de referencia (ground truth para accuracy)",
        min_value=2, max_value=20, value=int(k_optimo), step=1,
        help="Se usa para generar y_true con KMeans(k_ref) y calcular el 'accuracy' relativo."
    )

    # --- 1) Cálculo de inercia (codo) y accuracy para k=2..20 ---
    k_range = range(2, 21)

    # Codo: lo calculamos con TODOS los PCs detectados (X_pca_all)
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_pca_all)
        inertia.append(km.inertia_)

    # Accuracy: lo calculamos en el espacio seleccionado (X_pca)
    # y_true proviene del clustering con k_ref
    kmeans_ref = KMeans(n_clusters=int(k_ref), n_init=10, random_state=random_state)
    y_true = kmeans_ref.fit_predict(X_pca)

    acc_list = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        y_pred = km.fit_predict(X_pca)

        # Mapear etiquetas de y_pred a y_true (por mayor frecuencia) sin usar scipy
        labels_mapped = np.zeros_like(y_pred)
        for i in range(k):
            mask = (y_pred == i)
            if np.any(mask):
                # etiqueta "verdadera" más frecuente dentro de ese cluster
                majority = np.bincount(y_true[mask]).argmax()
                labels_mapped[mask] = majority

        acc = (labels_mapped == y_true).mean()
        acc_list.append(acc)

    # --- 2) Mostrar las dos gráficas al mismo nivel ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.plot(list(k_range), inertia, marker="o", linestyle="--")
        ax1.axvline(x=int(k_optimo), color="red", linestyle="--", label=f"k elegido = {int(k_optimo)}")
        ax1.set_title("Método del Codo (Inercia)")
        ax1.set_xlabel("Número de Clusters (k)")
        ax1.set_ylabel("Inercia")
        ax1.grid(True, alpha=.3)
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.plot(list(k_range), acc_list, marker="o", linestyle="--", color="purple")
        ax2.axvline(x=int(k_ref), color="red", linestyle="--", label=f"k de referencia = {int(k_ref)}")
        ax2.set_title("K-Means Accuracy (vs. k de referencia)")
        ax2.set_xlabel("Número de Clusters (k)")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=.3)
        ax2.legend()
        st.pyplot(fig2)

    # --- 3) Clustering final con el k seleccionado en el sidebar (k_optimo) ---
    kmeans_final = KMeans(n_clusters=int(k_optimo), random_state=random_state, n_init=50)
    df_pca["fenotipo"] = kmeans_final.fit_predict(X_pca)

    dist_df = (
        df_pca["fenotipo"]
        .value_counts()
        .rename_axis("fenotipo")
        .reset_index(name="count")
        .sort_values("fenotipo")
        .reset_index(drop=True)
    )

    st.success(f"✅ Se han descubierto {int(k_optimo)} fenotipos")
    st.dataframe(dist_df, use_container_width=True)

    st.markdown("""
**Análisis de la Distribución La observación principal es que los cuatro grupos no tienen el mismo tamaño, lo cual es un hallazgo muy realista y significativo.**

  → Fenotipos Dominantes: Los grupos 1 (8433) y 3 (6682) son los más comunes, representando juntos aproximadamente el 68% de todos los pacientes.

  → Fenotipos Minoritarios: Los grupos 0 (3731) y 2 (3467) son menos frecuentes, constituyendo el 32% restante.
    """)

    

# ============ TAB 2: Exploración en PCA ============
with tab2:

    st.markdown("""
**Objetivo:** Ahora que tenemos la variable fenotipo, exploraremos cómo se relacionan las características fisiológicas con cada grupo. Veremos si los grupos son distintos y qué variables los definen.

    """)


    st.subheader("🔎 Exploración en el espacio PCA")

    # 1) Scatter PC1 vs PC2 por fenotipo (útil, lo mantenemos)
    pcs = detectar_pc_cols(df_pca)
    PC1 = next((c for c in pcs if c.upper()=="PC1"), pcs[0])
    PC2 = next((c for c in pcs if c.upper()=="PC2"), pcs[1] if len(pcs)>1 else pcs[0])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    sns.scatterplot(data=df_pca, x=PC1, y=PC2, hue="fenotipo", palette="viridis", s=18, alpha=0.8, ax=ax)
    ax.set_title("Fenotipos en PC1 vs PC2"); ax.grid(True, alpha=.3)
    st.pyplot(fig)

    # 2) “Comportamiento de Fenotipos en el Espacio PCA” (pairplot del código original)
    st.markdown("**Comportamiento de Fenotipos en el Espacio PCA (pairplot)**")
    # Para evitar plots muy pesados, muestreamos hasta N filas
    max_pts = st.slider("Máx. puntos para pairplot (para rendimiento)", 500, 5000, 2000, 500)
    df_pair = df_pca.copy()
    if len(df_pair) > max_pts:
        df_pair = df_pair.sample(max_pts, random_state=42)

    vars_pair = [c for c in pc_cols_all if any(c.upper() == f"PC{i}" for i in range(1,6))]
    if not vars_pair:
        vars_pair = pc_cols_all[:5]

    pair = sns.pairplot(
        df_pair,
        hue="fenotipo",
        palette="viridis",
        vars=vars_pair,
        plot_kws=dict(s=12, alpha=0.7),
        diag_kws=dict(fill=True, alpha=0.7),
    )
    pair.fig.suptitle('Comportamiento de Fenotipos en el Espacio PCA', y=1.02)
    st.pyplot(pair.fig)


    st.markdown("""
### Interpretación Clínica de la Prevalencia

** Fenotipo 1 (El más común: 37.8%): Hipertensión Leve o Basal **

  → Perfil Numérico: PC1 Negativo (-0.94), PC2 Negativo (-0.73).

  → Interpretación Clínica: Este grupo, que es el más grande de la muestra, tiene valores consistentemente bajos en los componentes que miden la severidad de la presión (tanto sistólica como diastólica). Representa el perfil hemodinámico más cercano a la normalidad dentro de la población de hipertensos, correspondiendo a casos de hipertensión leve o pacientes cuya condición está bien controlada.

** Fenotipo 3 (El segundo más común: 29.9%): Hipertensión Diastólica y taquicardia **

  → Perfil Numérico: PC1 Negativo (-1.08), PC2 Alto (+1.11).

  → Interpretación Clínica: El problema principal para este grupo no es la presión sistólica 
(que es relativamente baja en comparación con los otros). Su perfil se define por una presión diastólica y una frecuencia cardíaca elevadas (PC2 alto). Representa un tipo de hipertensión donde el sistema cardiovascular parece estar en un estado de "sobreactivación" constante.

** Fenotipo 0 (Menos común: 16.7%): Hipertensión Sistólica-Diastólica **

  → Perfil Numérico: PC1 Muy Alto (+2.42), PC2 Alto (+1.34).

  → Interpretación Clínica: Este grupo representa el perfil de hipertensión más severo en todas las métricas. El valor extremadamente alto en PC1 indica presiones sistólicas y de pulso muy elevadas, mientras que el valor alto en PC2 se traduce en presiones diastólicas y frecuencias cardíacas también altas. Es un fenotipo de "carga hemodinámica total".

** Fenotipo 2 (El menos común: 15.5%): Hipertensión Sistólica Aislada con Bradicardia Relativa **

  → Perfil Numérico: PC1 Alto (+1.80), PC2 Muy Bajo (-1.81).

  → Interpretación Clínica: Este es un perfil muy específico y de gran interés clínico. Se caracteriza por una presión sistólica y de pulso muy alta (PC1 alto) pero con una presión diastólica y frecuencia cardíaca notablemente bajas (PC2 muy bajo). Esta gran diferencia entre la presión sistólica y la diastólica (pulso amplio) es un indicador clásico de rigidez arterial.


    """)

# ============ TAB 3: Modelado (sobre PCs) ============
with tab3:
    st.subheader("🤖 Modelado supervisado usando PCs seleccionados")
    X = df_pca[pcs_to_use]
    y = df_pca["fenotipo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    classifiers = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
        "SVC": (SVC(), {'C': [1, 10], 'kernel': ['rbf']}),
        "DecisionTree": (DecisionTreeClassifier(random_state=42), {'max_depth': [5, 10, None]}),
        "RandomForest": (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {'n_estimators': [100]}),
        "LightGBM": (lgb.LGBMClassifier(random_state=42), {'n_estimators': [100]}),
        "GaussianNB": (GaussianNB(), {}),  # sin hiperparámetros
    }

    best_models = {}
    rows = []
    for name, (model, params) in classifiers.items():
        with st.spinner(f"Entrenando {name}..."):
            grid = GridSearchCV(model, params, cv=3, scoring="f1_macro", n_jobs=-1)
            grid.fit(X_train, y_train)
            best_models[name] = grid.best_estimator_
            rows.append({
                "Modelo": name,
                "F1 (CV)": grid.best_score_,
                "Mejores parámetros": str(grid.best_params_)
            })

    results_cv_df = pd.DataFrame(rows).sort_values("F1 (CV)", ascending=False).reset_index(drop=True)
    st.markdown("**Resultados (validación cruzada)**")
    st.dataframe(results_cv_df.style.format({"F1 (CV)": "{:.3f}"}), use_container_width=True)

    st.session_state["split"] = (X_train, X_test, y_train, y_test)
    st.session_state["best_models"] = best_models

# ============ TAB 4: Evaluación ============
with tab4:
    st.subheader("📈 Evaluación en test")
    if "best_models" not in st.session_state:
        st.info("Primero entrena los modelos en la pestaña **Modelado (sobre PCs)**.")
    else:
        X_train, X_test, y_train, y_test = st.session_state["split"]
        best_models = st.session_state["best_models"]

        results = []
        for name, model in best_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = classification_report(y_test, y_pred, output_dict=True)["macro avg"]["f1-score"]
            results.append({"Modelo": name, "Accuracy (test)": acc, "F1-Macro (test)": f1})

        results_test_df = pd.DataFrame(results).sort_values("F1-Macro (test)", ascending=False).reset_index(drop=True)
        st.markdown("**Ranking en Test**")
        st.dataframe(results_test_df.style.format({"Accuracy (test)": "{:.3f}", "F1-Macro (test)": "{:.3f}"}), use_container_width=True)

        st.markdown("---")
        st.markdown("**Matrices de confusión**")
        for name, model in best_models.items():
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{name}")
            ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
            st.pyplot(fig)


    st.markdown("""

La conclusión más importante es que los fenotipos hemodinámicos que se identifificaron mediante clustering en el espacio PCA están bien definidos y son muy fáciles de separar. Un rendimiento de 99.8% es prácticamente perfecto e indica que los grupos no se solapan entre sí.

El rendimiento es alto, porque la combinación de PCA + Clustering fue muy efectiva. El PCA transformó los datos de manera que las diferencias entre los grupos de pacientes se maximizaron, y el K-Means pudo trazar fronteras muy claras entre ellos. En esencia, se creó un problema de clasificación "ideal", donde cada fenotipo ocupa su propio espacio distintivo, como se veía en el gráfico de dispersión.

Análisis del Ranking de Modelos El modelo más simple, LogisticRegression, ha superado a todos los demás, incluidos los más complejos como RandomForest o GradientBoosting. Esto es un hallazgo muy significativo: Significa que los grupos son Linealmente Separables: La razón por la que un modelo lineal simple funciona tan bien es que las fronteras entre los fenotipos en el espacio PCA son, en su mayoría, líneas rectas (o planos). No se necesitan modelos complejos para aprender a diferenciarlos.

    """)

# ============ TAB 5: Conclusiones ============
with tab5:
    st.subheader("🏆 Conclusiones")
    st.markdown("""
**Hallazgos principales**

1) **Separabilidad alta.** El *clustering* en el espacio PCA produce fenotipos bien delimitados. Los modelos supervisados alcanzan **F1-Macro** y **Accuracy** muy altos, lo que indica que los grupos son aprendibles y equilibrados.

2) **Mapeo clínico interpretable (1..4).** La organización de los fenotipos en PC1/PC2 captura severidad y patrón circadiano, facilitando la interpretación clínica.

3) **Modelos simples triunfan.** El buen desempeño de **LogisticRegression** sugiere **separabilidad casi lineal** en el espacio PCA.

**Implicaciones**

- Soporta **estratificación de riesgo** y diseño de **rutas de manejo** por fenotipo.

- Permite **monitoreo longitudinal** y priorización de variables hemodinámicas clave.

**Limitaciones & próximos pasos**

- Validación en cohortes externas y **validación temporal** con series longitudinales.

- Analizar **estabilidad del número de clusters** (silhouette, gap statistic) y robustez del mapeo si cambian signos/escala.
- Añadir **explicabilidad** (SHAP/permutación) para auditar decisiones sobre PCs.

- Considerar un panel de **loadings** dinámico (ya incluido) y, si se dispone, visualizar **varianza explicada** por PC.
    """)