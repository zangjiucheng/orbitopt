"""Model download checks without downloading or importing translation engines."""

from io import BytesIO
import hashlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_argos_model.py'))


@pytest.mark.parametrize('valid', [True, False])
def test_model_checksum_is_checked_before_install(monkeypatch, valid):
    payload = b'test model archive'
    installed = []
    package = SimpleNamespace(get_installed_packages=lambda: [],
                              install_from_path=lambda path: installed.append(path.read_bytes()))
    monkeypatch.setitem(sys.modules, 'argostranslate', SimpleNamespace(package=package))
    context = MODULE['install'].__globals__
    monkeypatch.setitem(context, 'urlopen', lambda request, **kwargs: BytesIO(payload))
    monkeypatch.setitem(context, 'SHA256', hashlib.sha256(payload).hexdigest() if valid else 'wrong')
    if valid:
        MODULE['install']()
        assert installed == [payload]
    else:
        with pytest.raises(ValueError, match='checksum mismatch'):
            MODULE['install']()
        assert installed == []


def test_cached_model_needs_no_download(monkeypatch):
    package = SimpleNamespace(get_installed_packages=lambda: [
        SimpleNamespace(from_code='en', to_code='zh', package_version='1.9')])
    monkeypatch.setitem(sys.modules, 'argostranslate', SimpleNamespace(package=package))
    def unexpected(*args, **kwargs):
        raise AssertionError('cached model should not be downloaded')
    monkeypatch.setitem(MODULE['install'].__globals__, 'urlopen', unexpected)
    MODULE['install']()
