"""Public CLI and experiment-output boundaries for independent model suites."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import write_dataset
from dreamq.cli import main
from dreamq.experiments import comparison_root
from dreamq.utils import read_config


def test_config_inheritance_and_graph_seed_grid():
    root = Path(__file__).parents[1]
    config = read_config(root / 'configs/all_models.yaml', dev=True)
    assert len(config['zoo']['enabled']) == 19
    assert config['zoo']['grids']['M15']['seeds'] == [[0]]
    assert config['zoo']['grids']['M15']['epochs'] == [1]
    assert config['features']['morgan_bits'] == 2048
    assert config['artifact_dir'].endswith('/artifacts/dev')


def test_config_cycle_and_unsafe_run_name(tmp_path, config):
    a, b = tmp_path / 'a.yaml', tmp_path / 'b.yaml'
    a.write_text('extends: b.yaml\n')
    b.write_text('extends: a.yaml\n')
    with pytest.raises(ValueError, match='Cyclic'):
        read_config(a)
    config['zoo'] = {'output_name': '../escape'}
    with pytest.raises(ValueError, match='Run name'):
        comparison_root(config)


def test_new_cli_run_reload_subset_and_protect_existing_results(config, molecules, tmp_path):
    write_dataset(config, molecules)
    config['cv']['strategies'] = ['stratified']
    config['zoo'] = {'output_name': 'test_suite', 'enabled': ['M0', 'M4', 'M7', 'M9', 'M16', 'M17'],
                     'ensemble_members': ['M0', 'M4', 'M7'],
                     'grids': {'M9': {'n_estimators': [8], 'max_depth': [2]}},
                     'selection': {'k': 3, 'weights': [0, .25]}}
    path = tmp_path / 'comparison.yaml'
    path.write_text(yaml.safe_dump(config))
    legacy_config = Path(config['artifact_dir']) / 'resolved_config.json'
    legacy_config.write_text('legacy provenance sentinel')
    main(['run-models', '--config', str(path)])
    assert legacy_config.read_text() == 'legacy provenance sentinel'
    root = Path(config['prediction_dir']) / 'comparison/test_suite'
    assert pd.read_csv(root / 'M4_submission.csv').CatalogID.tolist() == ['leader1', 'leader2']
    main(['predict-models', '--config', str(path), '--models', 'M0'])
    assert len(pd.read_csv(root / 'M0_submission.csv')) == 2
    with pytest.raises(ValueError, match='new --run-name'):
        main(['compare', '--config', str(path), '--models', 'M0'])
    main(['fit-models', '--config', str(path), '--models', 'M0', '--run-name', 'separate'])
    assert (Path(config['artifact_dir']) / 'comparison/separate/final/M0.joblib').exists()
