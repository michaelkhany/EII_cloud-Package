import matplotlib
import pytest

matplotlib.use("Agg")

from eii_cloud.experiments import make_reference_scenario  # noqa: E402


@pytest.fixture(scope="session")
def scenario():
    return make_reference_scenario(seed=0)


@pytest.fixture(scope="session")
def engine_and_result(scenario):
    from eii_cloud import EIIEngine
    from eii_cloud.experiments.runner import default_detectors

    engine = EIIEngine(envelopes=scenario.envelopes, detectors=default_detectors(list(scenario.envelopes)),
                       rules=scenario.rules, context=scenario.context)
    result = engine.run(scenario.dataset, scenario.reference)
    return engine, result
