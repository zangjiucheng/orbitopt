#!/usr/bin/env python3
"""Translate English Markdown review copies with offline Argos Translate."""

import argparse
from collections import Counter
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin


ENGINE_REVISION = "argos-1.11.0-en_zh-1.9-markdown-3"
OUTPUT = Path("translations/zh")
# These must survive machine translation exactly, including code and URLs.
TECHNICAL = re.compile(
    r"\A---\n.*?\n---(?=\n|$)|"
    r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*(?=\n|$)|"
    r"(?:^(?: {4}|\t)[^\n]*(?:\n|$))+|"
    r"(`+)[^`\n]*?\2|"
    r"\]\([^\n]*?\)|^\s*\[[^\]\n]+\]:[^\n]+|"
    r"<[^>]+>|\{\{.*?\}\}|\{%.*?%\}",
    re.MULTILINE | re.DOTALL,
)
LINK = re.compile(r"(\]\()((?:\{\{.*?\}\}|[^\s)])+)([^)]*\))")
REFERENCE = re.compile(r"^(\s*\[[^\]\n]+\]:\s*)(\S+)([^\n]*)$", re.MULTILINE)
GLOSSARY = {
    "models": "模型", "model": "模型", "verification": "验证",
    "trajectories": "轨迹", "trajectory": "轨迹", "propagation": "轨道传播",
    "flight-path angle": "航迹角", "inclination": "轨道倾角",
    "periapsis": "近拱点", "apoapsis": "远拱点",
    "perigee": "近地点", "apogee": "远地点", "epoch": "历元",
    "reference frames": "参考系", "reference frame": "参考系",
    "booster": "助推器", "thrust": "推力", "ascent": "上升段",
    "not checked": "未检查", "not assessed": "未评估",
    "numerical consistency": "数值一致性", "model agreement": "模型一致性",
    "historical validation": "历史数据验证",
    "Python": "Python", "CuPy": "CuPy", "NumPy": "NumPy",
    "pykep": "pykep", "pygmo": "pygmo", "tudatpy": "tudatpy",
    "Mission Control": "Mission Control", "Flight Lab": "Flight Lab",
    "OpenTsiolkovsky": "OpenTsiolkovsky", "GitHub": "GitHub",
}
GLOSSARY = {term.lower(): value for term, value in GLOSSARY.items()}
TERMS = re.compile(r"\b(" + "|".join(re.escape(term) for term in
                   sorted(GLOSSARY, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def protected(text):
    return Counter(match.group() for match in TECHNICAL.finditer(text))


def validate(source, translated):
    if not translated.strip() or protected(source) != protected(translated):
        raise ValueError("Translation changed code, links, HTML/Liquid or frontmatter")
    if Counter(re.findall(r"\d+", source)) != Counter(re.findall(r"\d+", translated)):
        raise ValueError("Translation changed numeric values")


def load_translator():
    from argostranslate import translate as argos

    engine = argos.get_translation_from_codes("en", "zh")
    return lru_cache(maxsize=16384)(engine.translate)


def translate_markdown(source, translate_text, *, fallbacks=None):
    """Translate complete paragraphs with validated, restorable placeholders."""
    inline = re.compile(
        r"\*\*[^*\n]+\*\*|(?<!\*)\*[^*\n]+\*(?!\*)|"
        r"!?\[[^\]\n]*\]\([^)]*\)|" + TECHNICAL.pattern + "|(?i:" + TERMS.pattern + ")" +
        r"|https?://[^\s)]+|[|*_\[\]#!]|\b[A-Z][A-Z0-9_-]+\b|"
        r"[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?", re.MULTILINE | re.DOTALL)
    # Argos sometimes shortens the alphabetic wrapper (e.g. ZXQ6QXZ → ZQQ6Z).
    # Recover only this wrapper alphabet; numeric IDs must still occur once each.
    marker = re.compile(r"(?<![A-Za-z])[ZXQ]{1,5}\s*(\d+)\s*[ZXQ]{1,5}(?![A-Za-z])", re.IGNORECASE)

    def prose(text):
        leading = text[:len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()):]
        body = text.strip().replace("\n", " ")
        if not re.search(r"[A-Za-z]", body):
            return text
        saved = []
        def protect(match):
            value = GLOSSARY.get(match.group().lower(), match.group())
            if value.startswith("**") and value.endswith("**"):
                value = "**" + prose(value[2:-2]).strip() + "**"
            elif value.startswith("*") and value.endswith("*"):
                value = "*" + prose(value[1:-1]).strip() + "*"
            saved.append(value)
            return f" ZXQ{len(saved) - 1}QXZ "
        masked = inline.sub(protect, body)
        # Labels consisting solely of protected terms need no inference.
        if not re.search(r"[A-Za-z]", marker.sub("", masked)):
            result = inline.sub(lambda m: GLOSSARY.get(m.group().lower(), m.group()), body)
        else:
            result = translate_text(masked).strip()
            indices = [int(m[1]) for m in marker.finditer(result)]
            if sorted(indices) != list(range(len(saved))):
                sentences = re.split(r"(?<=[.!?;])\s+(?=\S)", masked)
                if len(sentences) > 1:
                    result = " ".join(translate_text(sentence).strip() for sentence in sentences)
                    indices = [int(m[1]) for m in marker.finditer(result)]
                if sorted(indices) != list(range(len(saved))):
                    if fallbacks is None:
                        raise ValueError("Argos changed or dropped a protected placeholder")
                    fallbacks.append(body)
                    return text
            result = marker.sub(lambda m: saved[int(m[1])], result)
            if not result or "\n" in result:
                raise ValueError("Argos returned empty or multiline paragraph text")
        try:
            validate(body, result)
        except ValueError:
            if fallbacks is None:
                raise
            fallbacks.append(body)
            return text
        return leading + result + trailing

    def line(text):
        if re.fullmatch(r"[ \t|:\-\r\n]+", text):
            return text
        if text.lstrip().startswith("|"):
            # Do not split a pipe inside inline code.
            cells = re.split(r"(\|)(?=(?:[^`]*`[^`]*`)*[^`]*$)", text)
            return "".join(cell if i % 2 else prose(cell) for i, cell in enumerate(cells))
        prefix = re.match(r"[ \t]*(?:#{1,6}|[-+*>]|\d+[.)])[ \t]+", text)
        return (prefix[0] + prose(text[prefix.end():])) if prefix else prose(text)

    def paragraphs(text):
        chunks = re.split(r"(\n[ \t]*\n)", text)
        result = []
        for i, chunk in enumerate(chunks):
            if i % 2 or not chunk.strip():
                result.append(chunk)
            elif re.search(r"^[ \t]*(?:#{1,6} |[-+*>] |\d+[.)] |\|)", chunk, re.MULTILINE):
                result.append("".join(line(part) for part in chunk.splitlines(keepends=True)))
            else:
                result.append(prose(chunk))
        return "".join(result)

    parts = []
    offset = 0
    for match in TECHNICAL.finditer(source):
        value = match.group()
        is_block = (match.start() == 0 and value.startswith("---\n")) or bool(
            re.match(r"(?:[ \t]*(?:`{3,}|~{3,})| {4}|\t)", value))
        if is_block:
            parts.extend((paragraphs(source[offset:match.start()]), value))
            offset = match.end()
    parts.append(paragraphs(source[offset:]))
    result = "".join(parts)
    validate(source, result)
    return result


def rewrite_links(text, source_path, available):
    """Keep translated Markdown readable on GitHub, including image links."""
    def replace(match):
        target = match[2]
        if target.startswith("{{ site.baseurl }}"):
            return match[1] + target.replace("{{ site.baseurl }}", "https://zangjiucheng.github.io/orbitopt", 1) + match[3]
        if re.match(r"(?:[a-zA-Z][\w+.-]*:|/)", target):
            return match[0]
        absolute = urljoin(f"https://source.invalid/{source_path}", target)
        relative = absolute.removeprefix("https://source.invalid/")
        path = relative.split("#", 1)[0].split("?", 1)[0]
        if "#" not in relative and "?" not in relative and path in available:
            target = os.path.relpath(path, Path(source_path).parent)
        else:
            # Fragment IDs change with translated headings; use the English anchor.
            kind = "raw.githubusercontent.com" if path.endswith((".png", ".jpg", ".svg")) else "github.com"
            prefix = "main" if kind.startswith("raw") else ("tree/main" if path.endswith("/") else "blob/main")
            target = f"https://{kind}/zangjiucheng/orbitopt/{prefix}/{relative}"
        return match[1] + target + match[3]
    # Code and frontmatter stay byte-for-byte intact during link rewriting too.
    parts = []
    offset = 0
    for match in TECHNICAL.finditer(text):
        if match.group().startswith("](") or REFERENCE.fullmatch(match.group()):
            continue
        parts.append(REFERENCE.sub(replace, LINK.sub(replace, text[offset:match.start()])))
        parts.append(match.group())
        offset = match.end()
    parts.append(REFERENCE.sub(replace, LINK.sub(replace, text[offset:])))
    return "".join(parts)


def translate(root, *, dry_run=False, force=False, translator=None):
    root = Path(root).resolve()
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "README.md", "docs/**/*.md", "docs/*.md"],
        cwd=root,
    ).decode().split("\0")
    sources = {name: (root / name).read_text() for name in files if name}
    manifest_path = root / OUTPUT / ".sources.json"
    review_path = root / OUTPUT / "review.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    hashes = {name: hashlib.sha256((ENGINE_REVISION + text).encode()).hexdigest()
              for name, text in sources.items()}
    pending = [name for name in sources if force or manifest.get(name) != hashes[name]
               or not (root / OUTPUT / name).exists()]
    print(f"English → zh: {len(pending)} documents, {sum(len(sources[p]) for p in pending)} source characters")
    if dry_run:
        return pending
    if pending and translator is None:
        translator = load_translator()
    available = set(sources)
    outputs = {}
    # Validate the whole batch in memory before replacing any review copies.
    for name in pending:
        print(f"Translating {name}", flush=True)
        retained = []
        try:
            translated = translate_markdown(sources[name], translator, fallbacks=retained)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        outputs[name] = rewrite_links(translated, name, available).rstrip() + "\n"
        review[name] = retained
        if retained:
            print(f"  Retained {len(retained)} English paragraphs for review", flush=True)
    for name, text in outputs.items():
        target = root / OUTPUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for name in manifest.keys() - sources.keys():
        # Only remove generated paths from our manifest, never arbitrary paths.
        if name == "README.md" or (name.startswith("docs/") and ".." not in Path(name).parts):
            (root / OUTPUT / name).unlink(missing_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    review_path.write_text(json.dumps({name: paragraphs for name, paragraphs in review.items()
                                     if name in sources and paragraphs},
                                    indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return pending


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Count pending files without loading a model")
    parser.add_argument("--force", action="store_true", help="Retranslate unchanged sources too")
    args = parser.parse_args()
    translate(Path(__file__).resolve().parents[1],
              dry_run=args.dry_run, force=args.force)
