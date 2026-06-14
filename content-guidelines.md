# Content Guidelines — Agentic AI Course Website

Read this file before creating or editing any HTML content or verification code.
This is the single source of truth for all authoring decisions.

---

## 1. Theme: Solarized Light

All pages use the Solarized Light palette exactly as defined below.
No other color scheme is used anywhere on the site.

```
--sol-base3:   #fdf6e3   page background
--sol-base2:   #eee8d5   panel / card background
--sol-base1:   #93a1a1   secondary text, borders
--sol-base0:   #657b83   body text
--sol-base00:  #657b83   (same as base0 in light mode)
--sol-base01:  #586e75   emphasis text
--sol-base02:  #073642   headings, strong
--sol-yellow:  #b58900   highlight accent
--sol-orange:  #cb4b16   Intuition layer label
--sol-blue:    #268bd2   Formal layer label, links, accent
--sol-cyan:    #2aa198   Applied layer label
--sol-green:   #859900   success / verified badge
--sol-red:     #dc322f   error / warning badge
```

Typography:
- Headings: `'Georgia', serif`
- Body: `-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
- Code / math display: `"SF Mono", "Fira Code", "Fira Mono", monospace`
- Body size: 17px, line-height: 1.75
- Max content width: 780px, centered

Math rendering: MathJax 3, loaded from CDN.
  `<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>`
Inline: `\(...\)`. Display: `\[...\]`.

---

## 2. File Naming

- Topic pages: `topics/<slug>.html` — slug is lowercase, hyphenated, topic-descriptive.
  No week numbers in filenames or page titles.
  Examples: `transformers-self-attention.html`, `peft-lora.html`, `rag.html`
- Verification: `verification/<slug>/` — one subfolder per topic with a visualization.
- Never rename existing files without updating all links.

---

## 3. Topic Page Structure

Every topic page follows this exact layout. Do not add or remove sections.

```
<nav>           course name | back to index
<header>        topic title, one-line description
                prerequisites (linked) | leads to (linked)
<section>       concept 1
  .intuition    layer 1 — plain language
  .formal       layer 2 — mathematics (MathJax)
  .applied      layer 3 — practical context
  .viz          visualization (if applicable)
<section>       concept 2   (repeat)
<section>       concept N   (repeat)
<footer>        references
```

Layers are stacked vertically — all visible, no tabs.
Order is always: Intuition first, then Formal, then Applied.

Layer label styling:
- Small all-caps label above each layer: "INTUITION", "FORMAL", "APPLIED"
- Left border: orange for Intuition, blue for Formal, cyan for Applied
- Background: slightly tinted version of the border color

---

## 4. Content Standards

### Source requirements
Every factual claim, equation, and definition must have an inline citation `[N]`.
Acceptable sources:
  - Primary research papers (arXiv, conference proceedings)
  - Deep-dive technical posts from: Jay Alammar, Lilian Weng, Andrej Karpathy,
    Sebastian Ruder, Chris Olah, Distill.pub, Harvard NLP Annotated Transformer
  - Official documentation (PyTorch, HuggingFace) for implementation facts

Not acceptable as primary source:
  - Random blog posts, Medium articles without academic grounding
  - Wikipedia
  - AI-generated summaries without original source verification
  - Claude's training knowledge used directly without web-verified source

### Before writing any topic page
Claude must perform web research to retrieve the authoritative sources for that topic.
Do not rely on training data alone. Fetch the actual pages and cite specific content.

### No hallucination rule
Mathematical formulations must match exactly what is in the cited source.
If sources disagree on notation, note it explicitly.
Specific numbers (layer counts, dimensions, parameter counts) require a citation.
Do not invent equations or state consensus where there is none.

### Citations
Inline: superscript `[N]` linked to the reference entry at the bottom.
Reference format:
  `[N] Author(s). "Title." Venue/Journal, Year. URL.`

---

## 5. Visualization Verification Pipeline

See `course-website-plan.md` Section 7 for the full pipeline.

Summary of rules:
- Never add a visualization without a passing pytest suite in `verification/<slug>/`.
- Never publish a page where the JS math badge shows a failure.
- Ground truth JSON is embedded inline in the HTML `<script>` block — not fetched.
- Tolerance: 1e-6 default, overridable per-visualization in ground_truth.json metadata.
- Run `bash verification/run_all_tests.sh` to re-verify all topics at once.
- Regenerate ground_truth.json by running `export_ground_truth.py` after any
  change to `reference.py`.

Verification workflow for a new visualization:
  1. Write `reference.py` (numpy only, no ML framework)
  2. Write `test_reference.py` and run pytest — fix until all pass
  3. Run `export_ground_truth.py` → `ground_truth.json`
  4. Implement same math in JS; embed ground_truth.json inline
  5. Open in browser; confirm "Math verified ✓" badge appears
  6. Do not publish until badge passes

---

## 6. Visualization Font Sizes

All text inside a visualization block (`.viz-body` and everything inside it) must use
the same font-size as the body: **17px**. This applies to every element without exception:

- `.viz-desc` — the description paragraph
- `.head-tab` — head selector buttons
- `.steps-toggle` — the show/hide button
- `.hm-row-label`, `.hm-col-label` — token labels on the heatmap axes
- `.hm-cell` — heatmap cell values
- `.hm-legend`, `.hm-axis-label` — legend and axis annotation
- `.matrix-table`, `.matrix-table th` — computation step matrices
- `.step-label` — step headings inside the steps panel
- Any `<p>` or inline `style` inside JS template literals that renders into the viz

The only elements exempt from this rule are `.viz-title` and `.math-badge`, which are
UI chrome in the viz header bar (not content).

When adding a new visualization, set all font sizes to 17px from the start.
Never use inline `font-size` overrides inside viz content — let elements inherit from body.

---

## 7. Interactive Visualizations and Knobs

Every interactive visualization should have at least one knob (slider, toggle, or selector)
that teaches something a static diagram cannot. The lesson can be any of:

- A failure mode (e.g., attention collapse when scaling is removed, reward over-optimization)
- A trade-off (e.g., rank vs. parameter count in LoRA)
- A parameter sensitivity (e.g., how temperature shifts the sampling distribution)
- An emergent property (e.g., how head specialization depends on the input)
- A conceptual connection between two course topics

The key requirement: moving the knob produces a clear, observable lesson. The type of lesson
is determined by what is pedagogically most valuable for that specific topic.

### Walkthrough mode
Visualizations with multiple computation steps should display all steps simultaneously as a
vertical stack (not hidden behind a toggle). Steps that are affected by a knob get a visual
marker ("← changes with T") and a blue border. Steps that are unaffected get a "unaffected"
tag. Every knob change must recompute and rerender all affected steps simultaneously — the
student should see the entire chain update in real time.

### Insight text
Each affected step should include a short "insight box" (styled `.step-insight`) that
explains what the current knob value implies for that step. This text should update live with
the knob and use concrete numbers from the current computation (e.g., score range, entropy).

### Framing for random weights
When a visualization uses randomly initialized weights (not trained), include a `.note` block
near the top of the visualization stating this explicitly. The framing: "The mathematics is
real, the semantics is not yet." This prevents students from over-interpreting the patterns.

---

## 8. What Not To Do

- No week numbers in filenames, headings, or navigation.
- No AI filler phrases ("it's worth noting", "this is important", "let's explore").
- No equations without inline citations.
- No visualizations without a passing verification suite.
- No external JS libraries beyond MathJax — all visualization code is vanilla JS.
- No inline styles that override the Solarized Light theme.
- No tabs for layer switching — layers are always stacked.
- No content in the Applied layer that is also in the Formal layer — they are distinct.
- No copy-paste of equations without checking they match the cited source exactly.
