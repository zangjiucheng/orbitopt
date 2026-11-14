"""Translation orchestration checks without API calls or compute dependencies."""

import json
from pathlib import Path
import runpy
import subprocess
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/translate_docs.py"
MODULE = runpy.run_path(str(SCRIPT))


def tracked(root, name, text):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    subprocess.run(["git", "add", name], cwd=root, check=True)


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked(tmp_path, "README.md", "# Hello\n\n[Guide](docs/guide.md)\n")
    tracked(tmp_path, "docs/guide.md", "# Hello\n\nUse `orbitopt`.\n\n```bash\norbitopt --help\n```\n")
    return tmp_path


def translate_stub(text):
    return text.replace("Hello", "你好")


def test_dry_run_needs_no_engine_and_writes_nothing(repository, monkeypatch):
    def unexpected():
        raise AssertionError('dry run must not load the engine')
    monkeypatch.setitem(MODULE['translate'].__globals__, 'load_translator', unexpected)
    assert len(MODULE["translate"](repository, dry_run=True)) == 2
    assert not (repository / "translations").exists()


def test_incremental_translation_and_source_updates(repository):
    with patch.dict(MODULE["translate"].__globals__, load_translator=lambda: translate_stub):
        MODULE["translate"](repository)
        output = repository / "translations/zh/README.md"
        assert "你好" in output.read_text()
        output.write_text(output.read_text() + "\nReviewed translation.\n")
        assert MODULE["translate"](repository) == []
        assert "Reviewed translation." in output.read_text()
        (repository / "README.md").write_text("# Hello again\n")
        assert MODULE["translate"](repository) == ["README.md"]
        assert "again" in output.read_text()


@pytest.mark.parametrize("before,after", [
    ("Use `orbitopt`.", "Use `wrong`."),
    ("```bash\norbitopt\n```", "```bash\nwrong\n```"),
    ("~~~bash\norbitopt\n~~~", "~~~bash\nwrong\n~~~"),
    ("[Guide](guide.md)", "[Guide](wrong.md)"),
    ("---\npermalink: /api/\n---\nHello", "---\npermalink: /wrong/\n---\nHello"),
    ('<span id="entry"></span>', '<span id="wrong"></span>'),
    ("{{ site.baseurl }}", "{{ wrong }}"),
    ("Altitude 400 km", "Altitude 500 km"),
])
def test_rejects_technical_changes(before, after):
    with pytest.raises(ValueError):
        MODULE["validate"](before, after)


def test_failure_does_not_replace_review_copies(repository):
    def broken(text):
        raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        MODULE["translate"](repository, translator=broken)
    assert not (repository / "translations").exists()


def test_links_use_translated_pages_and_english_assets_without_touching_code():
    text = '[Guide](guide.md) ![Image](assets/a.png) [Install](guide.md#install)\n\n`[Guide](guide.md#install)`\n'
    result = MODULE["rewrite_links"](text, "docs/quickstart.md", {"docs/guide.md"})
    assert "[Guide](guide.md)" in result
    assert "https://raw.githubusercontent.com/zangjiucheng/orbitopt/main/docs/assets/a.png" in result
    assert "https://github.com/zangjiucheng/orbitopt/blob/main/docs/guide.md#install" in result
    assert "`[Guide](guide.md#install)`" in result


def test_reference_links_and_repository_directories():
    text = '[catalog](../src/vehicles/)\n\n[guide]: guide.md#install\n'
    result = MODULE["rewrite_links"](text, "docs/quickstart.md", {"docs/guide.md"})
    assert 'https://github.com/zangjiucheng/orbitopt/tree/main/src/vehicles/' in result
    assert '[guide]: https://github.com/zangjiucheng/orbitopt/blob/main/docs/guide.md#install' in result


def test_jekyll_api_links_become_published_urls():
    source = '[Python]({{ site.baseurl }}/api/python/)\n`{{ site.baseurl }}`\n'
    result = MODULE['rewrite_links'](source, 'docs/api.md', set())
    assert '[Python](https://zangjiucheng.github.io/orbitopt/api/python/)' in result
    assert '`{{ site.baseurl }}`' in result


def test_argos_sees_only_prose_and_glossary_keeps_technical_meaning():
    calls = []
    def translator(text):
        calls.append(text)
        return text.replace('Check', '检查').replace('Status', '状态').replace('Earth', '地球')
    source = '# Models\n\nCheck **3 models** at epoch `t0`.\n\n| Model | Status |\n|---|---|\n| Earth | Not checked |\n\n```bash\norbitopt --help\n```\n'
    result = MODULE["translate_markdown"](source, translator)
    assert '# 模型\n' in result
    assert '**3 模型**' in result
    assert '| 模型 |' in result
    assert '|---|---|' in result
    assert '未检查' in result
    assert '`t0`' in result and 'orbitopt --help' in result
    assert not any('t0' in text or 'orbitopt' in text for text in calls)
    assert any('检查' not in text and 'Check' in text and ' at ' in text for text in calls)


def test_frontmatter_indented_code_and_liquid_never_enter_translator():
    source = '---\npermalink: /api/\n---\n# Hello\n\n    print("hello")\n\n{{ site.baseurl }}\n'
    result = MODULE["translate_markdown"](source, lambda text: text.upper())
    assert result.startswith('---\npermalink: /api/\n---\n')
    assert '    print("hello")' in result
    assert '{{ site.baseurl }}' in result


def test_missing_placeholders_retain_english_and_are_reported():
    source = 'Use `orbitopt` at 400 km.\n'
    retained = []
    result = MODULE['translate_markdown'](source, lambda text: '译文', fallbacks=retained)
    assert result == source
    assert retained == [source.strip()]
    with pytest.raises(ValueError, match='placeholder'):
        MODULE['translate_markdown'](source, lambda text: '译文')


def test_shortened_placeholder_wrappers_preserve_their_original_values():
    source = 'Use `orbitopt` at 400 km.\n'
    result = MODULE['translate_markdown'](
        source, lambda text: text.replace('ZXQ', 'ZQQ').replace('QXZ', 'Z'))
    assert '`orbitopt`' in result
    assert '400' in result
    assert 'ZXQ' not in result and 'ZQQ' not in result


def test_invented_numbers_retain_original_paragraph():
    source = 'Check the result.\n'
    retained = []
    result = MODULE['translate_markdown'](source, lambda text: '检查 8 个结果。', fallbacks=retained)
    assert result == source
    assert retained == [source.strip()]


def test_retained_english_is_listed_in_review_report(repository):
    MODULE['translate'](repository, translator=lambda text: '译文')
    review = json.loads((repository / 'translations/zh/review.json').read_text())
    assert 'Use `orbitopt`.' in review['docs/guide.md']
    output = (repository / 'translations/zh/docs/guide.md').read_text()
    assert 'Use `orbitopt`.' in output


def test_deleted_sources_remove_only_generated_copy(repository):
    with patch.dict(MODULE["translate"].__globals__, load_translator=lambda: translate_stub):
        MODULE["translate"](repository)
    subprocess.run(["git", "rm", "-f", "docs/guide.md"], cwd=repository, check=True)
    MODULE["translate"](repository)
    assert not (repository / "translations/zh/docs/guide.md").exists()
    manifest = json.loads((repository / "translations/zh/.sources.json").read_text())
    assert "docs/guide.md" not in manifest
