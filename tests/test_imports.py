"""Smoke test: every public symbol of the refactor must import cleanly."""

import importlib

import pytest


PACKAGE_MODULES = [
    "ddcomb",
    "ddcomb.config",
    "ddcomb.runtime",
    "ddcomb.pipeline",
    "ddcomb.data.datasets",
    "ddcomb.data.dataloaders",
    "ddcomb.data.embeddings",
    "ddcomb.data.yolo",
    "ddcomb.data.kshot",
    "ddcomb.models.attention",
    "ddcomb.models.fusion",
    "ddcomb.models.plots",
    "ddcomb.models.embedding_models",
    "ddcomb.training.build",
    "ddcomb.training.train",
    "ddcomb.evaluation.evaluate",
    "ddcomb.kshot_results",
]


@pytest.mark.parametrize("module", PACKAGE_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_public_surface_names():
    """The intended package-root public API must remain importable."""
    from ddcomb import (  # noqa: F401
        DeepWeedsCSVDataset,
        FusionPipelineConfig,
        _checkpoint_path,
        build_fusion_model,
        evaluate_fusion_classifier,
        init_runtime,
        load_models_and_dataloaders_for_postfix,
        run_experiment,
    )


def test_structured_config_schemas_instantiate():
    """TrainFusionConfig and SingleBackboneConfig must instantiate with valid defaults / required fields."""
    from ddcomb.config import (
        FusionPipelineConfig,
        SingleBackboneConfig,
        SingleBackboneKShotSweepConfig,
        TrainFusionConfig,
    )

    fusion = TrainFusionConfig()
    assert fusion.mode == "stage_a"
    assert isinstance(fusion.pipeline, FusionPipelineConfig)

    sb = SingleBackboneConfig(model="resnet18", dataset_name="plantvillage", mode="head_only")
    assert sb.preprocess == "basic"

    sb_kshot = SingleBackboneKShotSweepConfig(
        model="resnet18",
        dataset_name="plantvillage",
        mode="head_only",
        kshot_dataset_key="plantvillage_color_split",
        kshot_shot=1,
        kshot_run_idx=1,
    )
    assert sb_kshot.output_dir.endswith("single_backbone_kshot")
    assert sb_kshot.kshot_root == "kshot/manifests"
    assert sb_kshot.kshot_results_root == "kshot/results_single_backbone"


def test_structured_config_schemas_reject_invalid_values():
    """Validation in __post_init__ must reject out-of-vocab choices."""
    import pytest

    from ddcomb.config import FusionPipelineConfig, SingleBackboneConfig, TrainFusionConfig

    with pytest.raises(ValueError):
        TrainFusionConfig(mode="stage_z")

    with pytest.raises(ValueError):
        FusionPipelineConfig(attention_method="unsupported_attention")
    with pytest.raises(ValueError):
        FusionPipelineConfig(classifier_method="unsupported_classifier")

    with pytest.raises(ValueError):
        SingleBackboneConfig(model="bogus", dataset_name="plantvillage", mode="head_only")
    with pytest.raises(ValueError):
        SingleBackboneConfig(model="resnet18", dataset_name="bogus", mode="head_only")
    with pytest.raises(ValueError):
        SingleBackboneConfig(model="resnet18", dataset_name="plantvillage", mode="bogus")


def test_register_configs_populates_config_store():
    """register_configs() must make the schemas resolvable via Hydra's ConfigStore."""
    from hydra.core.config_store import ConfigStore

    from ddcomb.config import register_configs

    register_configs()
    repo = ConfigStore.instance().repo
    assert "train_fusion_schema.yaml" in repo
    assert "train_single_backbone_schema.yaml" in repo
    assert "single_backbone_kshot_sweep_schema.yaml" in repo
    assert "pipeline" in repo
    assert "fusion_pipeline_schema.yaml" in repo["pipeline"]
