# Predicción de resultados de fútbol — LaLiga y Premier League

Trabajo práctico de Inteligencia Artificial (Grado en Ingeniería Informática, curso 2025/2026).  
Competición de Kaggle: predicción de resultados de partidos de fútbol (1_local, X_empate, 2_visitante).

## Estructura del proyecto

```
proyecto/
├── data/
│   ├── raw/                  # train.csv y test_con_resultados.csv (no incluidos en el repo)
│   └── processed/            # submission.csv y resultados intermedios
├── datasources/              # Carga de datos
├── features/
│   ├── internal/             # Calidad, ELO, H2H, estado, presión, condiciones
│   ├── derived/              # Diferencias home-away
│   ├── utils/                # Utilidades (teams_df)
│   └── selection.py          # RFE y permutation importance
├── models/                   # LightGBM, CatBoost, XGBoost, ensemble, calibración, Optuna
├── validation/               # Walk-forward y métricas (AUC-PR, F1)
├── 01_eda_features.ipynb     # EDA + ingeniería de variables + PCA
├── 02_modelado_walkforward.ipynb  # Tuning, walk-forward, selección de features (RFE vs PI)
├── 03_modelo_final_submission.ipynb  # Modelo final + submission.csv
├── main.py                   # Pipeline completo equivalente a los notebooks
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

## Configurar el kernel de Jupyter

Los notebooks deben ejecutarse con el mismo Python donde están instaladas las librerías. Después de instalar los requisitos, registrar el kernel una sola vez:

```bash
python -m ipykernel install --user --name proyecto-ia --display-name "Proyecto IA"
```

Luego, al abrir un notebook:
- **VS Code**: clic en el nombre del kernel (esquina superior derecha) → seleccionar **Proyecto IA**
- **Jupyter Notebook/Lab**: Kernel → Change kernel → **Proyecto IA**

## Cómo reproducir los resultados

Ejecutar los notebooks **en orden** con el kernel **Proyecto IA**:

1. `01_eda_features.ipynb` — exploración y construcción del dataset enriquecido
2. `02_modelado_walkforward.ipynb` — optimización y selección de variables (guarda `resultados_fase1.pkl`)
3. `03_modelo_final_submission.ipynb` — modelo final y generación de `submission.csv`

Alternativamente, ejecutar `python main.py` desde la raíz del proyecto hace todo el proceso en un solo paso.

**Nota**: los archivos `data/raw/train.csv` y `data/raw/test_con_resultados.csv` deben descargarse desde la página de la competición de Kaggle y colocarse en `data/raw/` antes de ejecutar.

## Métricas

- **Competición de probabilidades**: AUC-PR macro (Average Precision)
- **Competición de clasificación**: F1 macro

## Selección de variables

Se comparan dos técnicas (cuaderno 2):
- **RFE** (Recursive Feature Elimination) con LightGBM como estimador base
- **Permutation Importance** basada directamente en AUC-PR

## Notas sobre data leakage

Todas las variables temporales (ELO, forma rolling, H2H) se calculan usando únicamente información anterior a cada partido. El walk-forward validation asegura que la validación respeta el orden temporal.
