from feature_engine.selection import DropConstantFeatures
from feature_engine.selection import DropCorrelatedFeatures
from feature_engine.selection import ProbeFeatureSelection
from sklearn.ensemble import RandomForestClassifier


class BaseFiltering:
    """
    Clase que encapsula el pipeline de seleccion de features en 3 etapas:
      1. Eliminar features constantes / cuasi-constantes (DropConstantFeatures)
      2. Eliminar features altamente correlacionadas (DropCorrelatedFeatures)
      3. Eliminar features menos importantes que ruido aleatorio (ProbeFeatureSelection)

    Sigue el patron fit/transform para poder ajustar en train y aplicar en test
    sin data leakage.
    """

    def __init__(self,
                 constant_tol=0.9,
                 correlation_threshold=0.8,
                 correlation_method='pearson',
                 probe_n_probes=10,
                 probe_scoring='roc_auc',
                 probe_cv=3,
                 probe_n_estimators=50,
                 probe_max_depth=10,
                 random_state=42):

        # Paso 1: eliminar features cuasi-constantes
        self.drop_constant = DropConstantFeatures(tol=constant_tol)

        # Paso 2: eliminar features correlacionadas
        self.drop_correlated = DropCorrelatedFeatures(
            variables=None,
            method=correlation_method,
            threshold=correlation_threshold
        )

        # Paso 3: ProbeFeatureSelection
        self.probe_selection = ProbeFeatureSelection(
            estimator=RandomForestClassifier(
                n_estimators=probe_n_estimators,
                max_depth=probe_max_depth,
                random_state=random_state,
                n_jobs=-1
            ),
            variables=None,
            scoring=probe_scoring,
            n_probes=probe_n_probes,
            distribution="normal",
            cv=probe_cv,
            random_state=random_state,
            confirm_variables=False
        )

    def fit(self, X_data, y_data):
        """
        Ajusta los 3 filtros secuencialmente sobre los datos de entrenamiento.
        Cada filtro aprende que features eliminar y guarda esa informacion
        para aplicarla luego en transform().
        """
        # Paso 1: fit + transform para que el paso 2 reciba datos ya filtrados
        self.drop_constant.fit(X_data)
        X_no_constant = self.drop_constant.transform(X_data)

        self.n_dropped_constant = X_data.shape[1] - X_no_constant.shape[1]

        # Paso 2: fit + transform para que el paso 3 reciba datos ya filtrados
        self.drop_correlated.fit(X_no_constant, y_data)
        X_no_correlated = self.drop_correlated.transform(X_no_constant)

        self.n_dropped_correlated = X_no_constant.shape[1] - X_no_correlated.shape[1]

        # Paso 3: fit (el transform se hara cuando el usuario lo pida)
        self.probe_selection.fit(X_no_correlated, y_data)

        # Guardamos info para el resumen
        X_final = self.probe_selection.transform(X_no_correlated)
        self.n_dropped_probe = X_no_correlated.shape[1] - X_final.shape[1]
        self.n_features_initial = X_data.shape[1]
        self.n_features_final = X_final.shape[1]
        self.selected_features = X_final.columns.tolist()

    def transform(self, X_data):
        """
        Aplica los 3 filtros secuencialmente.
        Usa los parametros aprendidos en fit(), NO re-aprende nada.
        """
        X_out = self.drop_constant.transform(X_data)
        X_out = self.drop_correlated.transform(X_out)
        X_out = self.probe_selection.transform(X_out)
        return X_out

    def print_summary(self):
        """Imprime un resumen del pipeline de filtrado."""
        print("=" * 60)
        print("RESUMEN DEL PIPELINE DE FILTRADO")
        print("=" * 60)
        print(f"  Features iniciales:              {self.n_features_initial}")
        print(f"  Eliminadas cuasi-constantes:     -{self.n_dropped_constant}")
        print(f"  Eliminadas por correlacion:       -{self.n_dropped_correlated}")
        print(f"  Eliminadas por ProbeFeature:      -{self.n_dropped_probe}")
        print(f"  Features seleccionadas finales:  {self.n_features_final}")
        print("=" * 60)
