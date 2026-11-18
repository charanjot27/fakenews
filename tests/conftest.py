import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fakenews.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """Config whose data/ and outputs/ live in a temporary folder, with small settings for speed."""
    c = load_config(overrides=["evaluation.bootstrap=100", "hypotheses.bootstrap=100", "style_sweep.n_rows=300",
                               "style_sweep.values=[0.5,0.7,0.9]"])
    c["paths"]["data_dir"] = str(tmp_path / "data")
    c["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    return c


@pytest.fixture
def prepared(cfg):
    from fakenews.data import prepare
    from fakenews.synthetic import make_dataset

    path = make_dataset(cfg, n_rows=200)
    prepare(cfg, "synthetic", path, "tiny")
    return cfg


slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 to run transformer tests")
