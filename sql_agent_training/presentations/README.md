# Presentation Materials

This directory contains the source, templates, and release-ready PowerPoint files for the SQL agent training project.
It is deliberately separate from `../docs/`, which is the GitHub Pages website source.

## Layout

- `source/` contains the deck authoring script and the interview transcript.
- `templates/` contains the original template and the prepared starter deck required by the authoring script.
- `output/` contains the release-ready deck: `interview_presentation.pptx`.
- `.build/` contains generated layouts, renders, inspections, QA reports, and local runtime dependencies.
- `archive/` contains local backup copies that are intentionally not version-controlled.

## Version-Control Policy

The source files, both templates, and the release-ready deck are committed so they remain available after a fresh clone.
Generated build output and local backup copies remain ignored.

## Rebuilding

Run `source/build_interview_deck.mjs` with the presentation artifact runtime. The script resolves all paths relative to this directory, imports `templates/template-starter.pptx`, writes the finished deck to `output/`, and writes temporary inspection and rendering artifacts to `.build/`.
