#!/usr/bin/env python3
"""Install the checksum-pinned English-to-Chinese Argos 1.9 model."""

import hashlib
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

URL = "https://argos-net.com/v1/translate-en_zh-1_9.argosmodel"
SHA256 = "433e7c4f034d87fbe2353161e05f18646d7999452f801a4e1f0378522b9850ab"


def install():
    from argostranslate import package

    if any(p.from_code == "en" and p.to_code == "zh" and str(p.package_version) == "1.9"
           for p in package.get_installed_packages()):
        print("English-to-Chinese Argos 1.9 model already installed")
        return
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "en_zh.argosmodel"
        digest = hashlib.sha256()
        request = Request(URL, headers={"User-Agent": "orbitopt-docs/1.0"})
        with urlopen(request, timeout=120) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError("Argos model checksum mismatch")
        package.install_from_path(path)
    print("Installed English-to-Chinese Argos 1.9 model")


if __name__ == "__main__":
    install()
