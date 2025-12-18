from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import os
import tempfile

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt


@dataclass
class ExperimentConfig:
    experiment_name: str
    tracking_uri: str = "mlruns"
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tags:
            self.tags = {
                "project": "vnpt-credit-scoring",
                "team": "data-science",
            }


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "credit-scoring",
        tracking_uri: str = "mlruns",
        auto_log: bool = True
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.auto_log = auto_log
        self.client = None
        self.run_id = None
        self._setup_tracking()

    def _setup_tracking(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.experiment_name,
                tags={"project": "vnpt-credit-scoring"}
            )

        mlflow.set_experiment(self.experiment_name)

        if self.auto_log:
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                silent=True
            )

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id

        if tags:
            mlflow.set_tags(tags)

        if description:
            mlflow.set_tag("mlflow.note.content", description)

        mlflow.set_tag("run_timestamp", datetime.now().isoformat())

        return self.run_id

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status=status)
        self.run_id = None

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(key, float(value), step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        X_sample: Optional[pd.DataFrame] = None,
        registered_name: Optional[str] = None,
        conda_env: Optional[Dict] = None
    ):
        signature = None
        if X_sample is not None:
            try:
                y_pred = model.predict_proba(X_sample)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_sample)
                signature = infer_signature(X_sample, y_pred)
            except Exception:
                pass

        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=registered_name,
            conda_env=conda_env
        )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: Dict, artifact_file: str):
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure: plt.Figure, artifact_file: str):
        mlflow.log_figure(figure, artifact_file)

    def log_dataframe(
        self,
        df: pd.DataFrame,
        artifact_name: str,
        file_format: str = "csv"
    ):
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'.{file_format}',
            delete=False
        ) as f:
            if file_format == "csv":
                df.to_csv(f.name, index=False)
            elif file_format == "parquet":
                df.to_parquet(f.name, index=False)
            elif file_format == "json":
                df.to_json(f.name, orient='records')

            mlflow.log_artifact(f.name, artifact_name)
            os.unlink(f.name)

    def log_credit_scoring_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        prefix: str = ""
    ):
        from modeling.evaluation import DiscriminationMetrics, CalibrationMetrics

        disc = DiscriminationMetrics()
        calib = CalibrationMetrics()

        roc_result = disc.calculate_auc(y_true, y_prob)
        ks_result = disc.calculate_ks(y_true, y_prob)
        brier = calib.calculate_brier_score(y_true, y_prob)

        metrics = {
            f"{prefix}auc": roc_result.auc,
            f"{prefix}gini": roc_result.gini,
            f"{prefix}ks": ks_result.ks_statistic,
            f"{prefix}ks_decile": ks_result.ks_decile,
            f"{prefix}brier_score": brier,
            f"{prefix}optimal_threshold": roc_result.optimal_threshold,
        }

        self.log_metrics(metrics)

        return metrics

    def log_woe_analysis(
        self,
        woe_results: Dict[str, Any],
        artifact_path: str = "woe_analysis"
    ):
        summary_data = []
        for feature, result in woe_results.items():
            summary_data.append({
                "feature": feature,
                "iv": result.iv if hasattr(result, 'iv') else result.get('iv', 0),
                "n_bins": len(result.bins) if hasattr(result, 'bins') else result.get('n_bins', 0),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('iv', ascending=False)

        self.log_dataframe(summary_df, f"{artifact_path}/iv_summary")

        total_iv = summary_df['iv'].sum()
        self.log_metrics({"total_iv": total_iv, "n_features": len(summary_df)})

    def log_decile_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        artifact_path: str = "decile_analysis"
    ):
        from modeling.evaluation import DecileAnalysis

        analysis = DecileAnalysis()
        decile_table = analysis.generate_decile_table(y_true, y_prob)
        decile_df = analysis.to_dataframe(decile_table, format_percentages=False)

        self.log_dataframe(decile_df, artifact_path)

        self.log_metrics({
            "max_ks": decile_table.max_ks,
            "max_ks_decile": decile_table.max_ks_decile,
            "overall_bad_rate": decile_table.overall_bad_rate,
        })

    def log_stability_metrics(
        self,
        base_scores: np.ndarray,
        current_scores: np.ndarray,
        artifact_path: str = "stability"
    ):
        from modeling.evaluation import StabilityMetrics

        stability = StabilityMetrics()
        psi_result = stability.calculate_psi(base_scores, current_scores)

        self.log_metrics({
            "psi": psi_result.psi,
            "is_stable": int(psi_result.is_stable),
        })

        self.log_dict({
            "psi": psi_result.psi,
            "interpretation": psi_result.interpretation,
            "is_stable": psi_result.is_stable,
        }, f"{artifact_path}/psi_result.json")

        self.log_dataframe(psi_result.bin_details, f"{artifact_path}/psi_bins")

    def log_model_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        artifact_path: str = "plots"
    ):
        from modeling.evaluation import ModelEvaluationReport

        report = ModelEvaluationReport()

        fig = report.plot_full_report(y_true, y_prob, title="Model Evaluation")
        self.log_figure(fig, f"{artifact_path}/full_evaluation.png")
        plt.close(fig)

    def get_run(self, run_id: Optional[str] = None) -> mlflow.entities.Run:
        run_id = run_id or self.run_id
        if run_id is None:
            raise ValueError("No run_id specified")
        return self.client.get_run(run_id)

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        if order_by is None:
            order_by = ["metrics.auc DESC"]

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )

        return runs

    def get_best_run(self, metric: str = "auc", ascending: bool = False) -> Dict:
        order = "ASC" if ascending else "DESC"
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )

        if len(runs) == 0:
            return None

        return runs.iloc[0].to_dict()

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = ["auc", "gini", "ks", "brier_score"]

        data = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {"run_id": run_id, "run_name": run.info.run_name}

            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)

            data.append(row)

        return pd.DataFrame(data)


def get_or_create_experiment(
    experiment_name: str,
    tracking_uri: str = "mlruns",
    tags: Optional[Dict[str, str]] = None
) -> str:
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags=tags or {"project": "vnpt-credit-scoring"}
        )
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def log_model_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    run_name: Optional[str] = None,
    experiment_name: str = "credit-scoring",
    additional_params: Optional[Dict] = None,
    additional_metrics: Optional[Dict] = None
) -> Dict[str, float]:
    tracker = MLflowTracker(experiment_name=experiment_name, auto_log=False)

    tracker.start_run(run_name=run_name)

    if additional_params:
        tracker.log_params(additional_params)

    metrics = tracker.log_credit_scoring_metrics(y_true, y_prob)

    if additional_metrics:
        tracker.log_metrics(additional_metrics)
        metrics.update(additional_metrics)

    tracker.end_run()

    return metrics


def log_feature_importance(
    feature_importance: Dict[str, float],
    experiment_name: str = "credit-scoring",
    run_id: Optional[str] = None
) -> None:
    tracker = MLflowTracker(experiment_name=experiment_name, auto_log=False)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            _log_importance(tracker, feature_importance)
    else:
        _log_importance(tracker, feature_importance)


def _log_importance(tracker: MLflowTracker, feature_importance: Dict[str, float]):
    importance_df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in sorted(feature_importance.items(), key=lambda x: -abs(x[1]))
    ])

    tracker.log_dataframe(importance_df, "feature_importance")

    top_10 = list(feature_importance.items())[:10]
    for i, (feature, importance) in enumerate(top_10):
        tracker.log_metrics({f"importance_rank_{i+1}": importance})
