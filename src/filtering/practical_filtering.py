from feature_engine.selection import DropConstantFeatures
from feature_engine.selection import DropDuplicateFeatures
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


class PracticalFiltering:
    """
    Pipeline de filtrado alternativo al base.

    Etapas:
      1. Eliminar features constantes / cuasi-constantes
      2. Eliminar features duplicadas
      3. Eliminar features altamente correlacionadas
      4. Selección final con importancia de modelo (ExtraTrees + SelectFromModel)

    Mantiene el patrón fit/transform del profesor para evitar data leakage.
    """

    def __init__(
        self,
        constant_tol=0.995,
        correlation_threshold=0.90,
        correlation_method="spearman",
        n_estimators=200,
        max_depth=None,
        threshold="median",
        random_state=42,
    ):
        self.drop_constant = DropConstantFeatures(tol=constant_tol)
        self.drop_duplicates = DropDuplicateFeatures()

        self.drop_correlated = DropCorrelatedFeatures(
            variables=None,
            method=correlation_method,
            threshold=correlation_threshold,
        )

        self.selector_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        self.select_from_model = SelectFromModel(
            estimator=self.selector_model,
            threshold=threshold,
            prefit=False,
        )

        self.selected_features = []
        self.n_features_initial = None
        self.n_features_final = None
        self.n_dropped_constant = None
        self.n_dropped_duplicates = None
        self.n_dropped_correlated = None
        self.n_dropped_model = None

    def fit(self, X_data, y_data):
        self.n_features_initial = X_data.shape[1]

        self.drop_constant.fit(X_data)
        X_step1 = self.drop_constant.transform(X_data)
        self.n_dropped_constant = X_data.shape[1] - X_step1.shape[1]

        self.drop_duplicates.fit(X_step1)
        X_step2 = self.drop_duplicates.transform(X_step1)
        self.n_dropped_duplicates = X_step1.shape[1] - X_step2.shape[1]

        self.drop_correlated.fit(X_step2)
        X_step3 = self.drop_correlated.transform(X_step2)
        self.n_dropped_correlated = X_step2.shape[1] - X_step3.shape[1]

        self.select_from_model.fit(X_step3, y_data)
        support_mask = self.select_from_model.get_support()
        self.selected_features = X_step3.columns[support_mask].tolist()

        X_final = self.select_from_model.transform(X_step3)
        self.n_features_final = X_final.shape[1]
        self.n_dropped_model = X_step3.shape[1] - X_final.shape[1]

        return self

    def transform(self, X_data):
        X_out = self.drop_constant.transform(X_data)
        X_out = self.drop_duplicates.transform(X_out)
        X_out = self.drop_correlated.transform(X_out)
        X_out = X_out.loc[:, self.selected_features].copy()
        return X_out

    def fit_transform(self, X_data, y_data):
        self.fit(X_data, y_data)
        return self.transform(X_data)

    def print_summary(self):
        print("=" * 60)
        print("RESUMEN DEL PIPELINE DE FILTRADO PRACTICAL")
        print("=" * 60)
        print(f"Features iniciales:                {self.n_features_initial}")
        print(f"Eliminadas cuasi-constantes:      -{self.n_dropped_constant}")
        print(f"Eliminadas duplicadas:            -{self.n_dropped_duplicates}")
        print(f"Eliminadas por correlación:       -{self.n_dropped_correlated}")
        print(f"Eliminadas por modelo:            -{self.n_dropped_model}")
        print(f"Features seleccionadas finales:    {self.n_features_final}")
        print("=" * 60)
