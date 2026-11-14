# Documentation translation

English in `README.md` and `docs/` is authoritative. The
[translation workflow](../.github/workflows/translate.yml) uses
[Argos Translate](https://github.com/argosopentech/argos-translate) on the CPU
to create Simplified Chinese review copies under `translations/zh/`.
No translation API key or paid translation service is required. The first run
downloads the English-to-Chinese model; translation then runs locally.

Enable **Allow GitHub Actions to create and approve pull requests** under
**Settings → Actions → General**. Run **Translate documentation** manually for
the initial batch; later English Markdown changes trigger it automatically.
The workflow updates one review PR. Review orbital terminology, units and
numerical claims before merging. Copies are readable on GitHub; generated
Python API pages and landing-page HTML are outside this Markdown pipeline.
The published Jekyll site uses the English sources.

Preview pending work without installing the engine:

```bash
python scripts/translate_docs.py --dry-run
```

To translate locally in a separate virtual environment:

```bash
python3.12 -m venv /tmp/orbitopt-translation
source /tmp/orbitopt-translation/bin/activate
pip install -r requirements-translation.txt
python scripts/install_argos_model.py
python scripts/translate_docs.py
```

The engine version and model checksum are pinned. CI caches model data.
Code, URLs, frontmatter, HTML/Liquid, Markdown delimiters and numbers are kept
outside the translator; a structural check precedes writing. Relative images
use English source assets; document links stay local when a translated page
exists, and fragment links use English headings.

Complete paragraphs are translated with protected placeholders. If Argos loses
a placeholder, the workflow retries by sentence and then retains the original
English paragraph if needed. `zh/review.json` lists these excerpts for human
review. This prevents corrupt code or numbers from entering a translation.

The committed `.sources.json` tracks source hashes and engine revision.
Unchanged documents retain reviewed translations. Changed documents are
translated again in full, replacing that document's earlier edits. Merging
the review PR persists new hashes; unmerged runs may repeat the same work.
Deleted sources remove their generated copies. The manual **force** option
retranslates everything. The workflow does not merge its own PR; inspect its
diff before merging because GitHub may suppress workflows triggered by PRs
opened using `GITHUB_TOKEN`.
