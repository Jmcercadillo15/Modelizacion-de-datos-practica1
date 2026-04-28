import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class PracticalPreprocess:
    """
    Preprocesamiento alternativo al BasePreprocess del profesor.

    Cambios principales respecto al base:
      - usa variables_withExperts.xlsx
      - imputación con SimpleImputer
      - escalado robusto en numéricas
      - ordinal encoding para variables naturalmente ordenadas
      - target encoding para categóricas nominales
      - TF-IDF para emp_title
      - nuevas variables de negocio / ratios financieros
    """

    def __init__(
        self,
        var_to_process,
        target,
        null_threshold=0.98,
        text_max_features=30,
        random_state=42,
    ):
        self.raw_predictors_vars = pd.read_excel(var_to_process)
        self.raw_predictors_vars = (
            self.raw_predictors_vars
            .query("posible_predictora == 'si'")
            .variable
            .tolist()
        )

        self.target_var = target
        self.null_threshold = null_threshold
        self.text_max_features = text_max_features
        self.random_state = random_state

        self.var_with_most_nulls = []
        self.numeric_vars = []
        self.ordinal_vars = []
        self.nominal_vars = []
        self.text_vars = []

        self.num_imputer = None
        self.ord_imputer = None
        self.nom_imputer = None
        self.text_imputer = None

        self.scaler = None
        self.ordinal_encoder = None
        self.target_encoder = None
        self.text_vectorizers = {}

        self.reference_year = None

    def _read_data(self, data):
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.read_csv(data)

    def _safe_divide(self, numerator, denominator):
        return numerator / (denominator.replace(0, np.nan) + 1e-6)

    def _clean_text(self, series):
        return (
            series.astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _add_date_features(self, X, fit=False):
        if "earliest_cr_line" in X.columns:
            date_col = pd.to_datetime(
                X["earliest_cr_line"],
                format="%b-%Y",
                errors="coerce"
            )
            X["earliest_cr_line_year"] = date_col.dt.year
            X["earliest_cr_line_month"] = date_col.dt.month

            if fit:
                valid_years = X["earliest_cr_line_year"].dropna()
                self.reference_year = int(valid_years.max()) if not valid_years.empty else 2020

            if self.reference_year is None:
                self.reference_year = 2020

            X["credit_history_age"] = self.reference_year - X["earliest_cr_line_year"]
            X = X.drop(columns=["earliest_cr_line"])

        return X

    def _add_domain_features(self, X):
        if {"fico_range_low", "fico_range_high"}.issubset(X.columns):
            X["fico_avg"] = (X["fico_range_low"] + X["fico_range_high"]) / 2.0
            X["fico_span"] = X["fico_range_high"] - X["fico_range_low"]

        if {"installment", "annual_inc"}.issubset(X.columns):
            X["installment_to_monthly_income"] = self._safe_divide(
                X["installment"], X["annual_inc"] / 12.0
            )

        if {"loan_amnt", "annual_inc"}.issubset(X.columns):
            X["loan_to_annual_income"] = self._safe_divide(
                X["loan_amnt"], X["annual_inc"]
            )

        if {"revol_bal", "total_rev_hi_lim"}.issubset(X.columns):
            X["revol_bal_to_total_rev_limit"] = self._safe_divide(
                X["revol_bal"], X["total_rev_hi_lim"]
            )

        if {"tot_cur_bal", "tot_hi_cred_lim"}.issubset(X.columns):
            X["total_bal_to_total_limit"] = self._safe_divide(
                X["tot_cur_bal"], X["tot_hi_cred_lim"]
            )

        if {"total_bc_limit", "annual_inc"}.issubset(X.columns):
            X["bc_limit_to_income"] = self._safe_divide(
                X["total_bc_limit"], X["annual_inc"]
            )

        return X

    def _prepare_features(self, X, fit=False):
        X = X.copy()

        if self.var_with_most_nulls:
            existing_to_drop = [c for c in self.var_with_most_nulls if c in X.columns]
            X = X.drop(columns=existing_to_drop)

        if "term" in X.columns:
            X["term"] = (
                X["term"]
                .astype("string")
                .str.replace(" months", "", regex=False)
                .str.strip()
            )

        X = self._add_date_features(X, fit=fit)
        X = self._add_domain_features(X)

        return X

    def _ensure_columns(self, X, columns):
        X = X.copy()
        for col in columns:
            if col not in X.columns:
                X[col] = np.nan
        return X[columns]

    def fit(self, data):
        df = self._read_data(data)
        X = df[self.raw_predictors_vars].copy()

        nulls_perc = X.isnull().mean().sort_values(ascending=False)
        self.var_with_most_nulls = nulls_perc[nulls_perc > self.null_threshold].index.tolist()

        X = self._prepare_features(X, fit=True)

        self.text_vars = [col for col in ["emp_title"] if col in X.columns]

        self.ordinal_vars = [
            col for col in ["term", "grade", "sub_grade", "emp_length"]
            if col in X.columns
        ]

        all_categoricals = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.nominal_vars = [
            col for col in all_categoricals
            if col not in self.text_vars + self.ordinal_vars
        ]

        self.numeric_vars = [
            col for col in X.columns
            if col not in self.text_vars + self.ordinal_vars + self.nominal_vars
        ]

        y = (df[self.target_var] != "Fully Paid").astype(int).values.ravel()

        if self.numeric_vars:
            self.num_imputer = SimpleImputer(strategy="median")
            X_num = pd.DataFrame(
                self.num_imputer.fit_transform(X[self.numeric_vars]),
                columns=self.numeric_vars,
                index=X.index,
            )

            self.scaler = RobustScaler()
            self.scaler.fit(X_num)

        if self.ordinal_vars:
            self.ord_imputer = SimpleImputer(strategy="most_frequent")
            X_ord = pd.DataFrame(
                self.ord_imputer.fit_transform(X[self.ordinal_vars]),
                columns=self.ordinal_vars,
                index=X.index,
            )

            ordered_categories = {
                "term": ["36", "60"],
                "grade": ["A", "B", "C", "D", "E", "F", "G"],
                "sub_grade": [f"{letter}{n}" for letter in "ABCDEFG" for n in range(1, 6)],
                "emp_length": [
                    "< 1 year", "1 year", "2 years", "3 years", "4 years",
                    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
                ],
            }

            categories = [ordered_categories[var] for var in self.ordinal_vars]

            self.ordinal_encoder = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            self.ordinal_encoder.fit(X_ord)

        if self.nominal_vars:
            self.nom_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")
            X_nom = pd.DataFrame(
                self.nom_imputer.fit_transform(X[self.nominal_vars]),
                columns=self.nominal_vars,
                index=X.index,
            )

            self.target_encoder = TargetEncoder(
                target_type="binary",
                smooth="auto",
                cv=5,
                shuffle=True,
                random_state=self.random_state,
            )
            self.target_encoder.fit(X_nom, y)

        if self.text_vars:
            self.text_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")

            X_text = pd.DataFrame(
                self.text_imputer.fit_transform(X[self.text_vars]),
                columns=self.text_vars,
                index=X.index,
            )

            for col in self.text_vars:
                vectorizer = TfidfVectorizer(
                    max_features=self.text_max_features,
                    ngram_range=(1, 2),
                    min_df=10,
                )
                vectorizer.fit(self._clean_text(X_text[col]))
                self.text_vectorizers[col] = vectorizer

        return self

    def transform(self, data):
        df = self._read_data(data)
        X = df[self.raw_predictors_vars].copy()
        y = (df[self.target_var] != "Fully Paid").astype(int)

        X = self._prepare_features(X, fit=False)

        output_blocks = []

        if self.numeric_vars:
            X_num = self._ensure_columns(X, self.numeric_vars)
            X_num = pd.DataFrame(
                self.num_imputer.transform(X_num),
                columns=self.numeric_vars,
                index=X.index,
            )
            X_num_scaled = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=self.numeric_vars,
                index=X.index,
            )
            output_blocks.append(X_num_scaled)

        if self.ordinal_vars:
            X_ord = self._ensure_columns(X, self.ordinal_vars)
            X_ord = pd.DataFrame(
                self.ord_imputer.transform(X_ord),
                columns=self.ordinal_vars,
                index=X.index,
            )
            X_ord_enc = pd.DataFrame(
                self.ordinal_encoder.transform(X_ord),
                columns=[f"{col}_ord" for col in self.ordinal_vars],
                index=X.index,
            )
            output_blocks.append(X_ord_enc)

        if self.nominal_vars:
            X_nom = self._ensure_columns(X, self.nominal_vars)
            X_nom = pd.DataFrame(
                self.nom_imputer.transform(X_nom),
                columns=self.nominal_vars,
                index=X.index,
            )
            X_nom_enc = pd.DataFrame(
                self.target_encoder.transform(X_nom),
                columns=[f"{col}_te" for col in self.nominal_vars],
                index=X.index,
            )
            output_blocks.append(X_nom_enc)

        if self.text_vars:
            X_text = self._ensure_columns(X, self.text_vars)
            X_text = pd.DataFrame(
                self.text_imputer.transform(X_text),
                columns=self.text_vars,
                index=X.index,
            )

            for col in self.text_vars:
                vectorizer = self.text_vectorizers[col]
                text_matrix = vectorizer.transform(self._clean_text(X_text[col])).toarray()
                text_df = pd.DataFrame(
                    text_matrix,
                    columns=[f"{col}_tfidf_{token}" for token in vectorizer.get_feature_names_out()],
                    index=X.index,
                )
                output_blocks.append(text_df)

        X_output = pd.concat(output_blocks, axis=1)
        X_output = X_output.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X_output, y

    def print_summary(self):
        print("=" * 60)
        print("RESUMEN DEL PREPROCESAMIENTO PRACTICAL")
        print("=" * 60)
        print(f"Variables iniciales:              {len(self.raw_predictors_vars)}")
        print(f"Variables > {self.null_threshold:.0%} nulos:      {len(self.var_with_most_nulls)}")
        print(f"Variables numericas finales:      {len(self.numeric_vars)}")
        print(f"Variables ordinales finales:      {len(self.ordinal_vars)}")
        print(f"Variables nominales finales:      {len(self.nominal_vars)}")
        print(f"Variables de texto finales:       {len(self.text_vars)}")
        print("=" * 60)
