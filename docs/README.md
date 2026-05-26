# Docs

This folder contains the presentation slide deck and architecture diagrams.

## Files

| File | Description |
| :--- | :--- |
| [`presentation.md`](presentation.md) | Marp source — edit this file to update slides |
| [`presentation.pdf`](presentation.pdf) | Exported PDF (re-generate after edits) |
| [`presentation.pptx`](presentation.pptx) | Exported PowerPoint (re-generate after edits) |
| [`diagrams.md`](diagrams.md) | Standalone system architecture diagram |
| [`diagrams/`](diagrams/) | Pre-rendered SVG diagrams embedded in slides |

---

## Export the Presentation

### Prerequisites

```bash
# Install Marp CLI (one-time)
npm install -g @marp-team/marp-cli
```

### Generate PDF

```bash
npx @marp-team/marp-cli docs/presentation.md --pdf --allow-local-files --output docs/presentation.pdf
```

### Generate PowerPoint (PPTX)

```bash
npx @marp-team/marp-cli docs/presentation.md --pptx --allow-local-files --output docs/presentation.pptx
```

### Generate both at once

```bash
npx @marp-team/marp-cli docs/presentation.md --pdf --allow-local-files --output docs/presentation.pdf && \
npx @marp-team/marp-cli docs/presentation.md --pptx --allow-local-files --output docs/presentation.pptx
```

> **Note:** `--allow-local-files` is required because the slides embed local SVG diagrams from `docs/diagrams/`.

### VS Code (alternative)

Install the **Marp for VS Code** extension, open `presentation.md`, click the Marp icon in the top-right toolbar, then select **Export slide deck**.

---

## Regenerate Diagrams

If you edit any `.mmd` file in `docs/diagrams/`, re-render the SVG before exporting:

```bash
# Regenerate a single diagram
npx @mermaid-js/mermaid-cli -i docs/diagrams/phase1.mmd -o docs/diagrams/phase1.svg -b white

# Regenerate all diagrams at once
for f in architecture phase1 phase2 phase3 phase4 cicd; do
  npx @mermaid-js/mermaid-cli -i docs/diagrams/${f}.mmd -o docs/diagrams/${f}.svg -b white
done
```
