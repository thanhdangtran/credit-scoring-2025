from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion as MLflowModelVersion
import pandas as pd


class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelVersion:
    name: str
    version: str
    stage: ModelStage
    run_id: str
    source: str
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mlflow(cls, mv: MLflowModelVersion) -> "ModelVersion":
        return cls(
            name=mv.name,
            version=mv.version,
            stage=ModelStage(mv.current_stage),
            run_id=mv.run_id,
            source=mv.source,
            creation_timestamp=datetime.fromtimestamp(mv.creation_timestamp / 1000),
            last_updated_timestamp=datetime.fromtimestamp(mv.last_updated_timestamp / 1000),
            description=mv.description,
            tags=mv.tags or {},
        )


class ModelRegistry:
    def __init__(self, tracking_uri: str = "mlruns"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> ModelVersion:
        result = mlflow.register_model(model_uri, name)

        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, result.version, key, value)

        if description:
            self.client.update_model_version(
                name=name,
                version=result.version,
                description=description
            )

        return ModelVersion.from_mlflow(
            self.client.get_model_version(name, result.version)
        )

    def register_from_run(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> ModelVersion:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return self.register_model(model_uri, model_name, tags, description)

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        mv = self.client.get_model_version(name, version)
        return ModelVersion.from_mlflow(mv)

    def get_latest_versions(
        self,
        name: str,
        stages: Optional[List[str]] = None
    ) -> List[ModelVersion]:
        if stages is None:
            stages = ["None", "Staging", "Production"]

        versions = self.client.get_latest_versions(name, stages)
        return [ModelVersion.from_mlflow(v) for v in versions]

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        versions = self.get_latest_versions(name, stages=["Production"])
        return versions[0] if versions else None

    def get_staging_model(self, name: str) -> Optional[ModelVersion]:
        versions = self.get_latest_versions(name, stages=["Staging"])
        return versions[0] if versions else None

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True
    ) -> ModelVersion:
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage.value,
            archive_existing_versions=archive_existing
        )

        return self.get_model_version(name, version)

    def promote_to_staging(
        self,
        name: str,
        version: str,
        archive_existing: bool = True
    ) -> ModelVersion:
        return self.transition_stage(
            name, version, ModelStage.STAGING, archive_existing
        )

    def promote_to_production(
        self,
        name: str,
        version: str,
        archive_existing: bool = True
    ) -> ModelVersion:
        return self.transition_stage(
            name, version, ModelStage.PRODUCTION, archive_existing
        )

    def archive_model(self, name: str, version: str) -> ModelVersion:
        return self.transition_stage(name, version, ModelStage.ARCHIVED, False)

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Any:
        if stage:
            model_uri = f"models:/{name}/{stage.value}"
        elif version:
            model_uri = f"models:/{name}/{version}"
        else:
            model_uri = f"models:/{name}/latest"

        return mlflow.sklearn.load_model(model_uri)

    def load_production_model(self, name: str) -> Any:
        return self.load_model(name, stage=ModelStage.PRODUCTION)

    def load_staging_model(self, name: str) -> Any:
        return self.load_model(name, stage=ModelStage.STAGING)

    def search_model_versions(
        self,
        name: str,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[ModelVersion]:
        filter_str = f"name='{name}'"
        if filter_string:
            filter_str = f"{filter_str} and {filter_string}"

        versions = self.client.search_model_versions(
            filter_string=filter_str,
            max_results=max_results,
            order_by=order_by or ["version_number DESC"]
        )

        return [ModelVersion.from_mlflow(v) for v in versions]

    def list_registered_models(self) -> List[str]:
        models = self.client.search_registered_models()
        return [m.name for m in models]

    def delete_model_version(self, name: str, version: str):
        self.client.delete_model_version(name, version)

    def delete_registered_model(self, name: str):
        self.client.delete_registered_model(name)

    def get_model_version_metrics(self, name: str, version: str) -> Dict[str, float]:
        mv = self.client.get_model_version(name, version)
        run = self.client.get_run(mv.run_id)
        return run.data.metrics

    def compare_model_versions(
        self,
        name: str,
        versions: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = ["auc", "gini", "ks", "brier_score"]

        data = []
        for version in versions:
            mv = self.get_model_version(name, version)
            run_metrics = self.get_model_version_metrics(name, version)

            row = {
                "version": version,
                "stage": mv.stage.value,
                "created": mv.creation_timestamp,
            }

            for metric in metrics:
                row[metric] = run_metrics.get(metric, None)

            data.append(row)

        return pd.DataFrame(data)

    def get_model_lineage(self, name: str, version: str) -> Dict[str, Any]:
        mv = self.client.get_model_version(name, version)
        run = self.client.get_run(mv.run_id)

        return {
            "model_name": name,
            "model_version": version,
            "run_id": mv.run_id,
            "experiment_id": run.info.experiment_id,
            "parameters": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
            "artifacts": [a.path for a in self.client.list_artifacts(mv.run_id)],
            "source": mv.source,
            "creation_time": mv.creation_timestamp,
        }

    def set_model_version_tag(
        self,
        name: str,
        version: str,
        key: str,
        value: str
    ):
        self.client.set_model_version_tag(name, version, key, value)

    def set_model_description(
        self,
        name: str,
        version: str,
        description: str
    ):
        self.client.update_model_version(
            name=name,
            version=version,
            description=description
        )


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    tracking_uri: str = "mlruns",
    tags: Optional[Dict[str, str]] = None,
    description: Optional[str] = None
) -> ModelVersion:
    registry = ModelRegistry(tracking_uri)
    return registry.register_from_run(
        run_id=run_id,
        model_name=model_name,
        artifact_path=artifact_path,
        tags=tags,
        description=description
    )


def load_production_model(
    model_name: str,
    tracking_uri: str = "mlruns"
) -> Any:
    registry = ModelRegistry(tracking_uri)
    return registry.load_production_model(model_name)


def transition_model_stage(
    model_name: str,
    version: str,
    stage: Union[str, ModelStage],
    tracking_uri: str = "mlruns",
    archive_existing: bool = True
) -> ModelVersion:
    registry = ModelRegistry(tracking_uri)

    if isinstance(stage, str):
        stage = ModelStage(stage)

    return registry.transition_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing=archive_existing
    )
