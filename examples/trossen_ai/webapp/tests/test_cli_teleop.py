from typer.testing import CliRunner
from cli import app

runner = CliRunner()


def test_teleop_in_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "teleop" in res.output


def test_teleop_help_lists_mode_flag():
    res = runner.invoke(app, ["teleop", "--help"])
    assert res.exit_code == 0
    assert "--mode" in res.output
    assert "detached" in res.output
