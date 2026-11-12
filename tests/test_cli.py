import cli
from tests.conftest import FakeLLM


def test_cli_writes_report(pdf_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "GeminiClient", lambda *a, **k: FakeLLM())
    from gap_assistant import config
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    code = cli.main([str(pdf_dir), "--topic", "retina", "--out", str(out), "--json", str(js), "--no-novelty-check"])
    assert code == 0
    assert "Pediatric retinal images are unstudied" in out.read_text(encoding="utf-8")
    assert js.exists()


def test_cli_needs_two_pdfs(pdf_dir, capsys):
    assert cli.main([str(pdf_dir / "paper_a.pdf")]) == 2
