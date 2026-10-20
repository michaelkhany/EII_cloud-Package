import json

from eii_cloud.cli import main


def test_cli_demo_and_reproduce(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--formats", "svg,png", "--dpi", "40"]) == 0
    assert (out / "result.json").exists()
    assert main(["reproduce", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out.split("\n{", 1)[1].join(["{", ""]))
    assert rep["identical"]


def test_cli_validate_and_list(tmp_path, capsys, scenario):
    p = tmp_path / "rules.yaml"
    scenario.rules.to_yaml(p)
    assert main(["validate-rules", str(p)]) == 0
    assert main(["list"]) == 0
    text = capsys.readouterr().out
    assert "threshold" in text and "EIICloudChart" in text


def test_cli_protocols(capsys):
    assert main(["protocols", "exp00"]) == 0
    assert '"all_intervals_exact": true' in capsys.readouterr().out
