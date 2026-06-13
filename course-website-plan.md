# Agentic AI — Course Website Plan
*Living document. Update as decisions are made.*

---

## 1. Project Overview

A static HTML course website for the graduate Agentic AI course at BITS Pilani.
Each topic is a standalone HTML page. No backend, no build tooling, no frameworks —
pure HTML, CSS, and vanilla JS.

The goal is a Living Textbook: content that is layered, cited, interactive, and
mathematically honest. Not a PDF converted to a webpage.

---

## 2. File Structure

```
m26-agentic-ai-cm/
├── CLAUDE.md                          # auto-read by Claude Code; points to content-guidelines.md
├── content-guidelines.md              # all authoring rules (theme, layers, citations, etc.)
├── course-website-plan.md             # this file
├── index.html                         # course homepage
├── topics/
│   ├── transformers-self-attention.html
│   ├── beyond-transformers.html
│   ├── peft-lora.html
│   ├── inference-decoding.html
│   ├── test-time-compute.html
│   ├── post-training-rlhf.html
│   ├── direct-alignment-dpo.html
│   ├── interpretability.html
│   ├── rag.html
│   ├── agent-security.html
│   ├── evolutionary-methods.html
│   └── multimodal-llms.html
└── verification/
    └── transformers-self-attention/   # one subfolder per topic that has a visualization
        ├── reference.py
        ├── test_reference.py
        ├── export_ground_truth.py
        └── ground_truth.json
```

---

## 3. Theme: Solarized Light

All topic pages and the index use the Solarized Light color palette. No exceptions.

| Role                  | Hex       |
|-----------------------|-----------|
| Page background       | `#FDF6E3` |
| Panel / card bg       | `#EEE8D5` |
| Base text             | `#657B83` |
| Headings / strong     | `#073642` |
| Accent — blue         | `#268BD2` |
| Accent — cyan         | `#2AA198` |
| Accent — yellow       | `#B58900` |
| Accent — orange       | `#CB4B16` |
| Code block bg         | `#EEE8D5` |
| Code text             | `#586E75` |
| Border / divider      | `#93A1A1` |

Typography:
- Headings: Georgia serif
- Body: system sans-serif stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`)
- Code / math display: `"SF Mono", "Fira Code", monospace`
- Body font size: 17px, line-height 1.75
- Max content width: 780px, centered

Math rendering: MathJax 3 loaded from CDN. Inline `\(...\)` and display `\[...\]` notation.

---

## 4. Index Page (`index.html`)

- Course title and brief description at the top
- Topics organized by theme, not by week
- Each topic is a card with: title, one-line description, status badge (Published / Coming Soon)
- "Coming Soon" cards are visually muted but present, so students can see the full scope

### Topic groupings

| Theme                    | Topics |
|--------------------------|--------|
| Foundations              | Transformers & Self-Attention |
| Advanced Architectures   | SSMs / Mamba, MoE, Diffusion LMs |
| Efficient Training       | PEFT: LoRA, QLoRA, DoRA, AdaLoRA, Quantization |
| Inference & Serving      | Decoding strategies, KV-cache, PagedAttention, vLLM |
| Test-Time Compute        | Self-consistency, Tree/Graph-of-Thought |
| Post-Training            | RLHF, PPO, GRPO, Rejection Sampling |
| Direct Alignment         | DPO, SimPO, KTO, IPO, Instruction Tuning |
| Interpretability         | Mechanistic interpretability, Probing, Circuits |
| RAG                      | IR foundations, HyDE, RAPTOR, CRAG, DSPy |
| Agent Security           | Threat models, Red/Blue teaming, MCP security |
| Evolutionary Methods     | Prompt and agent evolution |
| Multimodal               | LLaVA, MM-ReAct, Visual instruction tuning |

---

## 5. Topic Page Structure

Every topic page follows the same layout. No exceptions.

### Header
- Topic title
- One-line description
- Prerequisite topics (linked)
- "Leads to" topics (linked)

### Concept sections (one per major concept in the topic)

Each concept section has three layers, stacked vertically in this order:

**Layer 1 — Intuition**
Plain language. Analogy. "What problem does this solve and why does the solution
look the way it does?" No equations. A student with a CS background but no ML
experience should be able to follow. Grounded in cited sources.

**Layer 2 — Formal**
Full mathematical treatment. Notation consistent with the primary paper for that
concept (e.g., Vaswani et al. 2017 for attention). Every symbol defined. Every
equation derived, not just stated. Rendered with MathJax. Inline citations at
every claim that comes from a source.

**Layer 3 — Applied**
Where this shows up in real systems. What breaks without it. Practical implications.
Connections forward to later course topics. Numbers where available (e.g., actual
dₖ values in GPT-2, actual head counts).

### Visualization (where applicable)
Placed after the concept section it illustrates. See Section 6 for verification rules.

### References
Full reference list at the bottom of every page. Format:
`[N] Author(s). "Title." Venue, Year. URL.`

Inline citation format: `[N]` as superscript, linked to the reference entry.

---

## 6. Content Standards

### Source requirements
- Every factual claim, equation, and definition must be cited.
- Acceptable sources: primary research papers; deep-dive technical blog posts from
  known authoritative authors (Jay Alammar, Lilian Weng, Andrej Karpathy,
  Sebastian Ruder, Chris Olah / Distill.pub, Harvard NLP Annotated Transformer).
- Not acceptable: random blog posts, Stack Overflow, Wikipedia as primary source,
  AI-generated summaries.
- Before writing any topic page, Claude must perform web research to retrieve the
  actual authoritative sources for that topic, read them, and cite specific content
  — not recall from training data.

### No hallucination rule
All mathematical formulations must match exactly what is in the cited source.
If there is a discrepancy between sources, note it explicitly in the text.
Claude must not invent equations, claim consensus where there is none, or state
specific numbers without a citation.

---

## 7. Visualization Verification Pipeline

*This section is being finalized. See discussion notes below.*

### Philosophy
A visualization that computes math incorrectly is worse than no visualization —
it confidently misteaches. Every interactive visualization on the site must pass
a verification pipeline before the HTML is published.

### The verification chain

```
Paper / authoritative source
        |
        v
Hand-computed toy example (verified by hand against paper)
        |
        v
Python reference implementation  (reference.py)
        |
        v
pytest test suite                (test_reference.py)
  - structural checks (e.g., softmax sums to 1)
  - numerical checks against hand-computed values
  - cross-checks against scipy/numpy built-ins where applicable
        |
        v
Ground truth export              (export_ground_truth.py)
  - runs reference.py on fixed toy inputs
  - writes exact float64 values to ground_truth.json
        |
        v
JavaScript visualization
  - implements the same math
  - embeds ground_truth.json values inline (no fetch; works offline/static)
        |
        v
JS assertions on page load
  - re-runs computation on the same toy inputs
  - compares to embedded ground_truth values within tolerance (1e-6)
  - if any assertion fails: visible warning banner in the UI
  - if all pass: "Math verified" badge shown
```

### File layout (one subfolder per visualization)
```
verification/
└── <topic-slug>/
    ├── reference.py           # canonical numpy implementation, line-by-line from paper
    ├── test_reference.py      # pytest; run with: pytest verification/<topic-slug>/
    ├── export_ground_truth.py # run to regenerate ground_truth.json
    └── ground_truth.json      # embedded verbatim in the HTML
```

### What gets tested in pytest

For each visualization, at minimum:
1. **Structural checks** — properties that must hold regardless of input
   (e.g., attention weights per row sum to 1; all weights >= 0)
2. **Tiny hand-computable case** — e.g., 2 tokens, dₖ = 2, so a human can
   verify the numbers with a calculator. Expected values hardcoded in the test.
3. **Cross-check against scipy/numpy built-in** where one exists
   (e.g., compare custom softmax against `scipy.special.softmax`)
4. **Numerical stability check** — verify max-subtraction trick in softmax
   gives same result as naive version on non-extreme inputs

### Handling interactive (user-driven) inputs
The JS assertions do not verify all possible user inputs. They verify a fixed set
of representative test cases (the toy inputs from ground_truth.json). This is
sufficient to confirm the implementation is correct — if it's correct for the
verified cases it's correct for all inputs, since the same code path runs.

### Workflow when adding or updating a visualization
1. Write `reference.py` (numpy, no ML framework)
2. Write `test_reference.py` and run `pytest` — fix until green
3. Run `export_ground_truth.py` to generate `ground_truth.json`
4. Implement the same math in JS in the HTML file
5. Embed `ground_truth.json` values inline in the `<script>` block
6. Open the page in a browser — confirm "Math verified" badge appears
7. If badge fails: debug JS against reference.py, do not publish

### Decisions

- **ground_truth.json placement**: Embedded inline in the HTML `<script>` block.
  Page is fully self-contained; works on any static host, GitHub Pages, USB, offline.
- **Float tolerance**: 1e-6 default. Can be overridden per-visualization via a
  `"tolerance"` field in the `ground_truth.json` metadata object.
- **Top-level test runner**: `verification/run_all_tests.sh` runs `pytest verification/`
  to re-verify all topics in one command.
- **Python environment**: `numpy`, `scipy`, `pytest` pinned in
  `verification/requirements.txt`.

---

## 8. CLAUDE.md and content-guidelines.md

`CLAUDE.md` will be minimal — just a pointer:

```
# Agentic AI Course Website

Read content-guidelines.md before creating or editing any course content.
```

`content-guidelines.md` will be the authoritative rules document covering:
- Theme (Section 3 above)
- Topic page structure (Section 5 above)
- Content standards / source requirements (Section 6 above)
- Visualization verification workflow (Section 7 above)
- File naming conventions
- What not to do (no week numbers in filenames, no AI filler phrases, no
  equations without citations, no visualizations without verification)

---

## 9. Build Sequence for First Topic

Once this plan is fully finalized:

1. Web research — fetch and read authoritative sources for Transformers /
   Self-Attention / Multi-Head Attention (Vaswani et al., Jay Alammar,
   Lilian Weng, Harvard Annotated Transformer, etc.)
2. Create `CLAUDE.md` and `content-guidelines.md`
3. Create `index.html`
4. Write `verification/transformers-self-attention/reference.py`
5. Write and run `test_reference.py`
6. Run `export_ground_truth.py` → `ground_truth.json`
7. Create `topics/transformers-self-attention.html` with full content
   and embedded verified visualization
8. Open in browser, confirm math badge passes, review content
