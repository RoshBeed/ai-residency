# AI Residency Portfolio — Working Notes

> **Current state (2026-08-23) — read this first, it supersedes everything below.**
>
> **The eleven repos are now one: `/Users/rosh/mlx/ai-residency`**, pushed private to
> `github.com/RoshBeed/ai-residency`. 259 commits, oldest 2025-06-09 — every original history was
> path-rewritten with `git filter-repo --to-subdirectory-filter` and merged, so authors, dates and
> messages are intact (week 1 still shows all five of its authors on `services/word-embeddings`).
>
> ```
> ai-residency/
> ├── libs/common/        was ai-residency-common — a uv workspace member now, not a git-tagged dependency
> ├── services/           one directory per service: package, tests, Dockerfiles, and its write-up as README.md
> │   ├── _template/      was ai-residency-repo-template; what `make new-service NAME=x` copies
> │   └── frontend/       was ai-residency-frontend
> ├── deploy/             was ai-residency-infrastructure — compose/, helm/, terraform/
> ├── notebooks/          one folder per service
> ├── pyproject.toml      the uv workspace: ONE lockfile, ONE ruff + mypy config
> ├── Dockerfile.base     roshbeed/ai-residency-base — no private dep, so no BuildKit secret any more
> ├── Makefile            install / check / build / push, dispatching on SERVICE=
> └── .github/workflows/  path-filtered CI, tagging all nine images from one commit
> ```
>
> **The week numbering is gone everywhere** — directories, images, Kubernetes Deployments, compose
> services, frontend page keys and env vars (`WEEK3_API_URL` is `VISION_TRANSFORMERS_API_URL`). The
> mapping is: week1→`word-embeddings`, week1-content-scoring→`content-scoring`,
> week2→`vector-search-retrieval`, week3→`vision-transformers`, week4→`multimodal-captioning`,
> week5→`audio-fine-tuning`, week6→`rlhf-ppo`. Read every `weekN` below as its service name.
>
> **Two long-standing "still open" items in this file were already resolved and the notes were
> wrong:** the `rosh.beed@icloud.com` author-email push blocker is spent (all repos pushed
> successfully on 2026-08-22), and the history blobs are gone (total `.git` was 63MB, not the
> ~256MB claimed for week 1 alone). Neither needed a rewrite.
>
> **How to work in it now:** `make install` once; `make check` runs ruff, strict mypy and pytest
> across every service; `make check SERVICE=x` narrows it. mypy runs *per service* deliberately —
> each has its own top-level `config` package, so one invocation over the tree sees nine modules
> called `config` and refuses to start. Service-specific pipelines stay in
> `make -C services/<name> help`. Images build from the repository root, not the service directory.
>
> **Still open:** nothing is on Docker Hub (0 repos under `roshbeed`), so `deploy/` still pins
> `:latest` rather than a commit SHA; the eleven old repos are pushed but not archived; and the
> monorepo is private, so the public profile README's links 404 until it is flipped.

> **The historical narrative below predates the consolidation.** It is kept for the
> reasoning, not the paths — every repo it names is now a directory in the monorepo.

> **Current state (2026-08-20) — read this first.** Week 1 was split in two:
> `ai-residency-week1-word-embeddings` (CBOW training, now serves a nearest-neighbour/analogy API)
> and `ai-residency-week1-content-scoring` (the regression head, pulls the frozen CBOW checkpoint
> from the word-embeddings repo's Hub). The frontend has two week-1 pages/backends (ports 8001 and
> 8007); `roshbeed/ai-residency-week1-content-scoring` holds `best_predictor.pth` on the Hub.
>
> **Current state (2026-08-17) — read this first.** The whole portfolio was renamed from
> `MLX-*` to `ai-residency-*` (see "Rename pass: MLX → ai-residency" at the end of this
> file). The historical narrative below still uses the old `MLX-*` names in several places;
> read every `MLX-WeekN-...`, `MLX-Portfolio-...`, `mlx-common`, `mlx_common` and `MLX_`
> as its `ai-residency-*` / `ai_residency_*` / `AI_RESIDENCY_*` equivalent. Not done yet:
> the HF Hub repos still carry the old `roshbeed/mlx-*` ids (rename is create-new +
> re-upload + delete-old), and `ai-residency-common` v0.7.2 is still local-only.
>
> **Current state (2026-08-12) — read this first.** A large cleanup session on 2026-08-12
> supersedes parts of the historical narrative below (which still uses old names like
> `ai-residency-hub`, `src/mlx_weekN_*`, and "installable package"):
>
> - **Ten sibling repos now.** `ai-residency-hub` was split in two: `ai-residency-frontend`
>   (the Streamlit app + all six `docs/weekN.md` write-ups) and `ai-residency-infrastructure`
>   (Helm chart, `docker-compose.yml`, AWS Terraform, Makefile). The six week repos, `ai-residency-common`,
>   and `ai-residency-repo-template` are unchanged in count.
> - **No more git submodules.** The Infrastructure repo no longer pulls the six week repos in as
>   submodules; `docker-compose.yml` and the Helm chart pull published Docker Hub images
>   (`roshbeed/ai-residency-weekN-serve` / `-train` / `ai-residency-frontend`) instead of building from
>   source. Checkpoint `hostPath` mounts from sibling repos remain as the fast local path.
> - **Every repo is a package** — one top-level package named after the repo's topic
>   (`word_embeddings/`, `content_scoring/`, `multimodal_captioning/`, `audio_fine_tuning/`, `template/`),
>   plus `config/` and `serving/` packages; no `.py` at the repo root, no `src/`, no installable
>   package, `[tool.uv] package = false`.
>   Week 1 was split in two (`ai-residency-week1-word-embeddings` — CBOW, and
>   `ai-residency-week1-content-scoring` — the regression head). Weeks 2/3/6 use several packages each
>   (more modules to group); the convention is "a package, not flat root modules".
> - **`ai-residency-common` is at v0.7.2** (added `sse.py`, `serving.py`, `checkpoint_path`) but **v0.7.2 is
>   local-only, never pushed**. Every repo's `pyproject.toml` pins `@v0.7.2` while its `uv.lock`
>   still resolves `v0.6.0`, so `uv sync` (non-frozen) and every Docker build are broken until
>   `ai-residency-common` v0.7.2 is pushed and each repo re-runs `uv lock`.
> - **Documentation was updated to match** the package layout and the Hub split — all six
>   `docs/weekN.md`, every week README, `ai-residency-common`'s README, and `ai-residency-repo-template`'s README.
> - No source `requirements.txt` exists anywhere — dependencies live in `pyproject.toml`/`uv.lock`
>   (the only `requirements.txt` files left are wandb's auto-generated snapshots in gitignored
>   `wandb/` dirs).

This directory is the working root for standalone repos, each a project of the
[Machine Learning Institute](https://ml.institute/) residency, cloned locally as siblings:

- `ai-residency-week1-word-embeddings` (word2vec CBOW) — **flagship / reference**
- `ai-residency-week1-content-scoring` (regression head over frozen CBOW vectors)
- `ai-residency-week2-vector-search-retrieval` (two-tower retrieval)
- `ai-residency-week3-vision-transformers` (ViT digit recognition)
- `ai-residency-week4-multimodal-captioning` (CLIP + Qwen3 captioning)
- `ai-residency-week5-audio-fine-tuning` (Whisper + emotion tokens)
- `ai-residency-week6-rlhf-ppo` (PPO/RLHF)

Three further siblings were added later, none of them a week of the residency:

- `ai-residency-hub` — a frontend/orchestration repo that pulls the six above in as git submodules
  and puts one Streamlit frontend in front of all six as backend containers. See its section below.
- `ai-residency-common` — a small shared package the six all depend on (device selection, logging,
  checkpoint format, config base, seeding, Hub upload, W&B). **The only repo pushed anywhere**;
  private, consumed by git URL pinned to a tag. See "Code standardization pass" below.
- `ai-residency-repo-template` — the canonical shape a week repo takes. It is a working repo, not a
  skeleton: `make train` trains, `make serve` answers, `make check` passes. Start any new repo
  from it.

All six have README parity (Pass 1) as of 2026-07-26. Each has real artifacts under `assets/` and
its own gitignored `TODO.md` with findings from actually running the pipeline. A later session
(2026-07-30) did a full cross-reference audit of Pass 1's README work against the team
presentations/course slides/team roster and fixed what it found — see "README cross-reference
audit" below; treat Pass 1 as audited-and-corrected, not just "done," from that point on.

Each is its own GitHub repo under `RoshBeed`, not a monorepo — cross-links between them must be
full `https://github.com/RoshBeed/ai-residency-weekN-...` URLs, never relative paths (relative paths only
resolve when browsing local clones side by side, not on GitHub).

**Goal:** use all six as an ML / Forward-Deployed-Engineer job-application portfolio.

## The plan — status as of 2026-08-02: **all of it is done**

1. **README parity across all six repos — done, and since audited/corrected.** All six matched the
   documentation bar established on Week 1 as of Pass 1 (2026-07-26); a later cross-reference audit
   (2026-07-30) checked that work against the real source material (team presentations, course
   slides, the team roster file) and fixed what it found — see "README cross-reference audit" below.
2. **Code hygiene pass, all six repos, in four stages.**
   - **2a — make it run — done, two rounds.** Pure bugfixing. See "Pass 2a" below.
   - **2c — infrastructure standardization — done.** Docker + Makefile + a walkthrough notebook,
     consistent across all six repos. See "Pass 2c" below.
   - **2e — deep repository structure standardization — done.** Physical directory reorg
     (`src/`/`checkpoints/`/`data/`/`notebooks/`/`ui/`) onto one common convention, applied per
     repo. This was originally slated for the single-repo Pass 3 below but the user explicitly
     chose to run it across all six now instead — see "Pass 2e" below.
   - **2b — broader hygiene checklist — SKIPPED, deliberately, 2026-08-02.** By the time it came
     up most of it had evaporated: `requirements.txt` completeness is superseded by uv and
     committed locks, tests and lint/type tooling arrived with the standardization pass, dead code
     was down to seven lines across six repos, and descriptions/topics/demo-link are all blocked on
     the repos not being pushed. What was left was LICENSE (**done — MIT, all nine repos**) and CI
     (**not done**; a workflow needs either a PAT secret per repo or `ai-residency-common` made public,
     since it is a private git dependency).
3. **One deeper architecture pass — done on Week 1, 2026-08-02.** See "Pass 3" below. The only
   item in this plan that was real architectural change rather than documentation, structure,
   tooling or verification.

**Since then, a full code-standardization pass (2026-07-31/08-01) has landed** — all six repos are
now installable packages on one shared standard, depending on a new `ai-residency-common` package, with ruff
and strict mypy passing everywhere. That was the largest single change to this portfolio; see "Code
standardization pass" below for the standard itself and the bugs it uncovered. It also supersedes
parts of Pass 2b's checklist (tests now exist in all six; lint/type tooling is universal), though
CI and LICENSE remain untouched.

Between 2e and 2b, four rounds of user-directed work happened that weren't part of the original
numbered plan — a README audit/fix pass, a full pip→uv dependency migration, a UI/backend-API
build-out for the three repos that didn't have one, and a new 7th repo (`ai-residency-hub`). Each
gets its own section below.

**Then three more, all on 2026-08-02, each its own section at the end of this file:**
**Three-host verification** (macOS, docker-compose and Kubernetes, training *and* serving, eleven
real bugs found), **Pass 3** (Week 1's architecture), and **No data in the repos** (nothing under
`data/` or `datasets/` is committed anywhere any more; it lives in Hub dataset repos).

**Then a fourth, larger one, 2026-08-06: the container-source / documentation-repo split** (own
section at the end of this file). User-directed architectural reversal of how the nine repos divide
labor: the six week repos are stripped down to pure Docker container source (two Dockerfiles each —
`Dockerfile.train` for RunPod/AWS, `Dockerfile.serve` for Kubernetes — both built `FROM` a new shared
`roshbeed/ai-residency-base` image), and every long README, diagram, and the UI move into `ai-residency-hub`
as the one documentation/entry-point repo. Also folded in: Week 3's AWS Terraform, generalized into
a portfolio-wide RunPod alternative.

**What is actually left is not code.** The six week repos cannot be pushed until their history,
authored with `rosh.beed@icloud.com`, is rewritten — which also gates CI, GitHub descriptions,
topics and a demo link. That rewrite is worth combining with purging the data blobs still sitting in
history. After that, the cloud step: the RunPod/AWS/Hetzner runbook is in the hub's `README.md`
under "Deploying to real hosts".

**Commit state as of 2026-08-02.** All nine repos are **committed and clean**, and none of them
tracks any data. Current HEADs:

| Repo | HEAD | Notes |
|---|---|---|
| `ai-residency-common` | `611e868` | **the only code repo pushed** — private, `github.com/RoshBeed/ai-residency-common`, tags `v0.1.0` … `v0.6.0`; depend on **`@v0.6.0`** |
| `ai-residency-repo-template` | `6ba8092` | local only |
| Week 1 | `02094f5` | branch `dev` |
| Week 2 | `be28232` | |
| Week 3 | `fbcb944` | |
| Week 4 | `2611d1a` | |
| Week 5 | `cef2948` | |
| Week 6 | `083d702` | |
| `ai-residency-hub` | `3e4fc51` | all six submodules re-pinned to the commits above |

Also on the Hub (all private): six **model** repos `roshbeed/ai-residency-weekN-*` holding trained weights,
and three **dataset** repos — `ai-residency-week1-hn-posts`, `ai-residency-week2-retrieval-data`,
`ai-residency-week4-sample-images`. Between them these are why no checkpoint or dataset needs to be committed
or bind-mounted from this specific machine any more.

**Nothing else has been pushed.** Two blockers before the six can be: their older history is
authored with `rosh.beed@icloud.com`, which GitHub's email-privacy setting rejects (the global
`user.email` change only affects new commits), and `.gitmodules` still points at local relative
paths, so the hub will not clone correctly for anyone else until the six have real remotes.

Still run `git status` before starting new work — this portfolio's passes routinely leave things
uncommitted mid-flight by design.

**Update, 2026-08-06.** The container-source/documentation-repo split (see that section at the end
of this file) touched eight of the nine repos again and left all eight uncommitted — the HEADs above
are still accurate (nothing was committed, so nothing moved), but "committed and clean" no longer
describes the working tree. Only `ai-residency-common` is untouched.

**Both are now resolved, 2026-08-02.** Pass 2b was skipped at the user's direction once most of its
checklist turned out to be superseded or blocked, and Pass 3 was done on Week 1 — see the section at
the bottom of this file for both. The standing advice this paragraph used to give (don't jump ahead
to Pass 3) is spent; the note worth keeping from it is that the extra rounds above were explicit,
direct user requests rather than a judgment call to take on unplanned scope, which is the
distinction that matters if a future session is weighing something similar.

## README methodology (established on Week 1, apply consistently to Weeks 2–6)

### Structure

Academic-paper scaffolding, technical-article prose. In order:

1. Title + one opening paragraph (no literal "Abstract" label — just an inviting, dense summary
   paragraph: what it predicts/does, the method in one sentence, what's evaluated, built-as-Week-N
   framing).
2. **Table of Contents** — numbered, linking to every top-level section.
3. **1. Introduction** — why the problem matters, motivating diagram(s).
4. **2. System Architecture** — a data-flow diagram and, where a real call sequence exists (see
   "Sequence diagrams" below), a UML sequence diagram. **Comes before Method (changed 2026-07-30,
   now confirmed as the standard across all six repos — see "README voice, rigor, and structure
   pass" below for the reasoning and the per-repo rollout).** Orient the reader with the system at
   a glance first, then go deep on the technical core — the order a person would actually walk
   someone through the repo in person, not the order a paper would.
5. **3. Method** — the technical core, broken into numbered subsections (3.1, 3.2, 3.2.1, ...) as
   needed. This is where most of the diagrams and all the math live.
6. **4. Implementation** — annotated file tree. Its opening line should explicitly point back to
   the Architecture section by number (e.g. "Mapping the architecture from Section 2 onto the
   actual file layout") since Architecture is no longer the immediately preceding section.
7. **5. Usage** — numbered steps matching the actual pipeline stages, runnable commands.
8. **6. Experimental Results** — real numbers, real sample outputs, an honest "what's missing" (or
   "what I'd fix next") list. Prefer numbers/artifacts pulled from the team's original presentation
   or wandb logs over invented ones — and where a number in the repo isn't actually reproducible
   (no fixed random seed, no baseline comparison), fix the underlying code, not just the prose; see
   "README voice, rigor, and structure pass" below for what this looks like in practice.
9. **7. Related Work** — cross-links to the other five repos where genuinely relevant (full GitHub
   URLs).
10. **References** — real numbered citations for any technique borrowed from a paper (e.g.
    Mikolov et al. for word2vec). Don't skip this even though it's a bootcamp project — if a
    technique has a paper, cite it.
11. **Citation** — a BibTeX block citing the repo itself (`author = {RoshBeed}`, correct year from
    `git log`, not guessed).
12. **Acknowledgments** — name the team by name if it was a team project (confirmed with user:
    yes, name them — see Week 1's "Kernel Kittens" credit). **Do not credit course slides or team
    presentation material as a diagram source** (changed 2026-07-30, per direct user instruction —
    the course's own terms allow reusing that material freely, without attribution; treat anything
    derived from it as this repo's own). Only cite real academic papers a figure is reproduced
    from (References-section entries) — see "Diagrams" and "Source material" below for exactly
    what that does and doesn't cover.
13. **Dependencies** — one-line tech stack list.

### Tone

**Voice updated 2026-07-30, now applied across all six repos (see "README voice, rigor, and
structure pass" below for the per-repo rollout and results).** Original register was
"technical-article, not stiff academic, not a lecture" — still true, but the user pushed further
after reading Week 1: cut what reads as AI-generated, cut over-explanation, and write like a
senior forward-deployed engineer describing a system they built, not a paper describing a method.
Concretely:
- **First person where it's about a real decision.** "I trained CBOW because..." / "I went with
  late fusion because it's the only one where stopping early still leaves a coherent system" reads
  as a real engineer's judgment call. "This repo trains CBOW" reads as a paper. Use "this repo"
  for descriptions of what the system does; use "I" when explaining *why* a choice was made.
- **Name tradeoffs and workarounds explicitly, including the unflattering ones.** "That's a
  workaround, not a design choice" (about a `PYTHONPATH` hack) is better than silence or a
  euphemism — a senior engineer reading this trusts self-aware imprecision far more than polish.
  Don't manufacture flaws that aren't there, but don't hide real ones either.
- Direct sentences, contractions, "this repo" not "this implementation adopts."
- No Latin abbreviations (cf., i.e. mid-sentence, etc. as a crutch).
- Explain *why* before *what* — a reader should understand the reasoning, not just the steps.
- **Cut AI-generated tells specifically** — these showed up repeatedly in a first draft and needed
  a dedicated pass to remove: rhetorical-question section-openers ("So, does any of this actually
  work?"), templated transition phrases ("Zoom out one level...", "With the method covered, here's
  how it's actually wired together..."), the word "exactly" as a verbal tic ("is exactly this",
  "made concrete", "exactly the problem X solves" — found 9 uses in one README, cut to near zero),
  and heavy em-dash overuse as the default connective (one README had 64 em-dashes in 613 lines;
  halving that by using periods/commas/parentheses instead reads as far more human). Vary sentence
  length and rhythm — a string of evenly-balanced two-clause sentences ("X does A, but Y does B")
  is itself a tell; let some sentences be short and blunt.
- **Cut self-referential commentary and redundant restatement, but never cut technical reasoning**
  — this is the line the user flagged as easy to get wrong in both directions. Cut: meta-commentary
  about the README's own sourcing process ("not a recreation — which is why it's worth going to the
  primary source instead"), narration of a repo's *previous* architecture that no longer exists,
  and the same point explained twice in a row. Never cut: *why* a technique works, a real tradeoff,
  or the reasoning behind a design choice — that reasoning is the actual evidence of understanding
  a technical hiring audience is reading for, and over-trimming it in the name of "sounding less
  AI" defeats the README's whole purpose. When genuinely unsure which category a sentence falls
  into, err toward keeping it and ask, rather than cutting first.
- Numbers and equations are real, not illustrative — pull them from actual training logs, wandb
  exports, or code, never invent a plausible-sounding metric. Where a number *isn't* reliably
  reproducible (no fixed seed, no baseline to compare against), that's a real bug worth fixing in
  the code, not just a caveat to add in prose — see "README voice, rigor, and structure pass"
  below for a concrete example of this on Week 1.

### Diagrams

**Policy as of 2026-07-30 (superseding the original "Mermaid only, recreate faithfully" rule):
where a real source image exists, use it — extract/screenshot the actual original diagram rather
than recreating it in Mermaid.** This was a deliberate user-directed change, not a refinement of
the original approach; the original Pass 1 methodology recreated diagrams in Mermaid from course
slides/presentations, and that work is now being revisited across all six repos to swap real
screenshots back in where possible.

- **Three categories of diagram, handled differently:**
  - **Paper-sourced diagrams** — anything the README is describing/recreating from a real academic
    paper cited in the References section (e.g. the word2vec CBOW architecture from Mikolov et
    al., the Transformer block from Vaswani et al., CLIP's dual-encoder diagram from Radford et
    al., Whisper's architecture diagram, a PPO/RLHF loop diagram). For these, **extract the real
    image** (see "Source material" below for mechanics), embed it directly with plain Markdown
    (`![alt](assets/figN-description.png)`), and **cite the source and the paper's own figure
    number in the caption** — e.g. "Figure 8. CBOW and Skip-gram architectures, Figure 1 from the
    original word2vec paper [1]." Citing the source figure number specifically (not just the
    paper) matters here for a real reason, not just academic politeness: the extracted image is a
    literal screenshot of the paper's page, so it has that paper's own "Figure 1" (or whatever)
    printed inside the picture itself. Without the caption saying so, a reader sees this
    document's "Figure 8" label sitting right above an image that visibly says "Figure 1" inside
    it — actually confusing, not the caption creating confusion. Follow with prose explaining how
    *this repo* implements what the diagram shows.
  - **Course-slide/team-deck-sourced diagrams** — the same extraction mechanics apply (see "Source
    material" below) and the image still gets embedded the same way, but **do not cite or credit
    the course/slide/deck as the source anywhere in the README** (changed 2026-07-30, per direct
    user instruction — confirmed the course's own terms allow reusing this material freely without
    attribution). Treat it exactly as this repo's own diagram: a plain descriptive caption with no
    "from the course's assignment brief" / "from the same workshop deck" framing, and no mention in
    Acknowledgments either. This is a real, deliberate asymmetry with paper-sourced figures above —
    papers still get cited (that's real intellectual property with its own citation norms), course
    material doesn't need to be (per the course's own stated terms).
  - **Invented diagrams** — ones with no real-world source because they describe this repo's own
    pipeline, not anyone else's: the System Architecture data-flow diagram, UML sequence diagrams
    reflecting the actual call order in this repo's code. These **stay Mermaid** — there is nothing
    to screenshot, and Mermaid is still the right tool for a diagram invented to document this
    codebase.
- Every diagram — Mermaid or extracted screenshot — gets a numbered, sequential **Figure N.**
  caption, in one running sequence for the whole document (not per-section numbering). Renumber
  every figure if a section reorder changes their document order (see the "README voice, rigor,
  and structure pass" section below for a worked example — moving Architecture before Method
  shifted 12 of Week 1's 17 figure numbers).
- **Validate every remaining Mermaid diagram before considering it done.** Extract all
  ` ```mermaid ` blocks and compile each with
  `npx -y @mermaid-js/mermaid-cli -i block.mmd -o block.svg`, confirm no errors. This has caught
  real mistakes every single time — never skip it. (Doesn't apply to extracted screenshots — there's
  nothing to compile.)
- Sequence diagrams (which stay Mermaid, being invented/code-derived) must use real UML notation,
  not an invented pseudo-notation: `actor` for a human/external role vs `participant` for a system
  component, `->>` for synchronous calls, `-->>` for returns, `+`/`-` (or explicit
  `activate`/`deactivate`) for activation bars, `Note over` for annotations. Do **not** use
  Mermaid's `rect` background-coloring to fake grouping — it isn't a real UML combined-fragment
  operator; use `Note over` labels instead, which are.
- Extracted screenshots get committed to that repo's `assets/` folder, same as the existing
  plots/UI-screenshots exception always has — only the *source* documents (slides, decks, papers)
  stay gitignored and uncommitted, not the cropped figures pulled from them.

### Source material: slides, team presentations, and cited papers

- Course slides live in `../Slides/Week N*.pdf` (three PDFs per week typically: assignment brief,
  a workshop deck, and sometimes an infra/tooling deck). Read them for terminology, diagrams, and
  the instructor's own framing of the problem before writing.
- Team presentation decks live in `../Presentations/MLX-weekN_*.pdf` (real team-authored decks;
  exist for weeks 1, 2, 4, 5, 6 — no deck for week 3). Week 1 additionally has the original
  `.pptx` checked out locally at its repo root (`MLX-week1_kernal-kitties.pptx`, gitignored).
- **Extracting a diagram, two methods depending on source format:**
  - **From a `.pptx`** (only Week 1 has one locally): `unzip -o file.pptx -d extracted` gives you
    `extracted/ppt/media/*.png` as the actual embedded raster images at full original quality;
    cross-reference `extracted/ppt/slides/_rels/slideN.xml.rels` to know which image belongs to
    which slide.
  - **From a PDF** (course slides, team decks in `../Presentations/`, or a downloaded paper): the
    diagram is usually vector-drawn directly on the page, not a separate embedded raster, so
    `pdfimages` alone won't reliably get the whole figure. Render the full page instead —
    `pdftoppm -png -r 300 -f <page> -l <page> file.pdf page` — then crop to just the diagram with
    `magick page-N.png -crop WxH+X+Y +repage cropped.png` (find the crop box by opening the
    rendered page image once and reading pixel coordinates). 300 DPI keeps text in the figure
    legible at README width.
  - Real charts/plots/sequence diagrams from any of these sources are worth extracting faithfully;
    treat wandb screenshots in team decks as ground truth for the Results section too.
- **Cited academic papers** (References section entries — e.g. Mikolov et al. for word2vec,
  Vaswani et al. for Transformers, Radford et al. for CLIP/Whisper): where the README describes or
  recreates a specific figure from the paper itself (architecture diagram, an equation figure), find
  the real paper (arXiv PDF is usually cleanest) and extract that figure the same way as above,
  rather than describing it in prose alone or hand-drawing it in Mermaid. Only do this where the
  README is genuinely describing that paper's own diagram — don't fetch a paper just to manufacture
  a figure that wasn't already being discussed.
- **Course slides and team decks get extracted the same way as papers but never cited in the
  README** (see "Diagrams" above for the full reasoning) — this is a source-material-type
  distinction, not a mechanics one; the `pdftoppm`/`magick`/`unzip` extraction steps below apply
  identically regardless of whether the result gets a citation.
- **Never commit slides, team presentations, or downloaded papers.** Add `slides/`, `*.pptx`, and
  any local papers directory to `.gitignore` per repo (copy the pattern from Week 1's
  `.gitignore`). It's fine to keep them locally and to use their content in the README — just don't
  check the source files themselves in. The *extracted, cropped figures* are different — those are
  committed to `assets/`, per the "Diagrams" section above.
- If a repo doesn't have its own copy of the relevant slides yet, check whether the user has them
  in `../Slides/` first before asking.

### Cross-repo links

Always `https://github.com/RoshBeed/ai-residency-weekN-...`, never `../ai-residency-weekN-...`. Each week is a
separate GitHub repo; relative links only work browsing local clones side by side and are dead
links on GitHub itself.

### Getting real artifacts (screenshots, plots) — optional, time-permitting

For Week 1 we actually ran the pipeline end to end to get a real t-SNE plot and a real Streamlit
screenshot (see `assets/` in that repo) rather than only describing results in prose. This is
valuable but expensive (had to fix three undocumented missing dependencies, train on a reduced
corpus for speed, use Playwright to drive and screenshot Streamlit) — treat it as a nice-to-have
per repo, not a blocker for README parity. If doing it:
- Set up a Python 3.11 venv (not the system default — newer Pythons may lack torch wheels).
- Check `requirements*.txt` is actually complete before trusting it — it probably isn't (see
  checklist below).
- Use `WANDB_MODE=offline` to avoid needing a real W&B login for a quick verification run.
- Back up and temporarily shrink any large training-data files / sweep configs for speed, then
  restore them afterward — never leave the repo in a modified state.
- No `chromium-cli` is installed in this environment; use Python `playwright` directly
  (`pip install playwright && python -m playwright install chromium`) to drive and screenshot a
  local web UI.
- Save generated images into a per-repo `assets/` folder and embed with plain Markdown
  `![alt](assets/foo.png)` — these get committed (unlike slides/pptx). Check the repo's
  `.gitignore` for a blanket `*.png`/`*.csv`/etc. rule before assuming the file will actually be
  tracked (Weeks 3 and 5 both had one) — add a `!assets/*.png` exception line rather than removing
  the blanket rule.
- `uv venv --python 3.11 .venv-run` + `uv pip install ...` is more reliable here than raw
  `python3 -m venv` — this environment's default `uv`-managed Python 3.11 has a broken relocatable
  build that crashes `ensurepip` with `Fatal Python error: init_fs_encoding`; `uv venv` sidesteps
  it entirely and is what every repo from Week 3 onward actually used.
- No live microphone/webcam in this environment. When a repo's demo needs one (Week 5's Streamlit
  audio recorder), macOS's built-in `say -o clip.aiff "text"` piped through `ffmpeg` produces
  genuine synthesized speech with a known ground-truth transcript — real audio through the real
  pipeline, just not literally captured live. Say so transparently in the README rather than
  presenting it as an actual recording.
- When a dataset needs a large git-lfs clone or gated access (Week 5's CREMA-D, ~7,442 clips) but a
  smaller sibling dataset downloads directly (EmoDB, ~40MB zip, no auth), download the smaller one
  for genuine artifacts and document what the large one would still need — don't fabricate data or
  skip the exercise because the "main" dataset is expensive.
- System `ffmpeg` (or similar Homebrew-installed binaries) can be present but broken — missing a
  dylib after a partial upgrade (hit with `snappy`, blocking `ffmpeg` in Week 5). Run
  `ffmpeg -version` before concluding it's "not installed"; usually `brew reinstall <missing-lib>`
  fixes it in seconds.
- Repos with no UI at all (Week 2, Week 6) don't need a screenshot substitute — a real,
  copy-pasted terminal block (a genuine `RuntimeError`, a before/after generation comparison) fills
  the "screenshots" role in the Results section just as well, sometimes better, since it shows the
  actual failure mode rather than a static image of it.

## Pass 2 — code hygiene

Don't accumulate repo-specific findings in this file. Instead: **each repo gets its own `TODO.md`
at its root, gitignored** (add `TODO.md` to that repo's `.gitignore` — see any repo for the
pattern). Repo-specific issues — dependency gaps, path mismatches, dead code, a trained model
never being persisted — live there, not here. Don't reference `TODO.md` or this file from the
public README itself; both are excluded from the repo, so a link to either is dead for anyone
reading it on GitHub. If a finding belongs in the README (e.g. an honest limitation worth
documenting for readers), write it there in its own words — a `TODO.md` entry and a README
limitations note are different things serving different audiences.

### Pass 2a — Make it run (current focus)

**Scope: get every repo running end to end, exactly as its own README describes. Pure bugfixing —
no refactors, no new features, no "while I'm here" cleanup, no scope creep.** This is deliberately
narrower than the full Pass 2b checklist below (no tests/CI/LICENSE work yet); the only goal is
that every command shown in that repo's README (`Usage`/`Running It` section) actually executes
and produces the behavior the README describes.

This is being run as six parallel sessions, one per repo, each starting from a cleared context and
reading only this file plus that one repo. That's fine — the six repos are fully separate
directories and git repos, no shared state to worry about. What each fresh session needs from this
file:

- **Where to start:** that repo's own gitignored `TODO.md` already lists concrete, specific bugs
  found during the README pass — most of Pass 2a for that repo is working through that list. It
  isn't necessarily exhaustive, though; still actually run every command in the README top to
  bottom and fix whatever else breaks.
- **"Recurring patterns across repos" below** is worth reading before starting on *any* repo, not
  just the one it was originally found in — several of these are library/environment version-drift
  issues (the Qwen3 dtype default, missing `requirements.txt` entries) likely to recur even where
  not yet noted.
- **The "Environment/tooling recipes" addendum inside "Getting real artifacts" above** (the `uv
  venv` workaround, the `ffmpeg`/Homebrew fix, etc.) — reuse those rather than rediscovering them.
- **Fix the code, not the README.** The READMEs just went through a full accuracy pass — don't
  rewrite prose to work around a bug instead of fixing it, and don't touch README wording as part
  of this pass except in the specific case below.
- **When a fix resolves a bug the README's own Results section describes as a known limitation**
  (e.g. Week 2's tower-never-persisted, Week 3's decoder-shift bug, Week 4's checkpoint/adapter
  mismatch, Week 6's untrained reward model) — that paragraph is now stale once the fix lands.
  Don't rewrite it inline while bugfixing (stay focused on the code); instead leave a note in that
  repo's `TODO.md` that the Results section needs a follow-up update, and treat reconciling the
  affected READMEs as one short pass after all six repos are fixed, not part of Pass 2a itself.
- **Definition of done for a repo:** every command in its README's Usage section runs successfully
  from a fresh clone (fresh venv, `pip install -r requirements.txt` or equivalent, nothing manually
  patched), producing output consistent with what the README says happens. Note any command that
  can't reasonably be verified in this environment (e.g. something needing a GPU, a paid API key,
  live hardware) in that repo's `TODO.md` rather than blocking on it.

This file *does* get cross-repo patterns — findings that showed up independently in two or more
repos stop being "that one repo's problem" and become useful context for working on any of them.
That's what the subsection right below is for.

### Recurring patterns across repos (learned during the Pass 1 run-it work)

Read this before starting Pass 2a — it's the fast way back into context on what's actually wrong
across the portfolio, without re-reading six `TODO.md` files cold. Each repo's own `TODO.md` still
has the full specific detail; this is just the pattern, so a recurrence in a repo not listed here
isn't a surprise.

- **Qwen3 loads in `bfloat16` by default now, and crashes against anything `float32`.** Hit in
  Weeks 4 and 6 (both load `Qwen/Qwen3-0.6B-Base`) — Week 5 is Whisper (OpenAI's `whisper`
  package) and never touches Qwen3, so it doesn't apply there. Loading via plain
  `AutoModelForCausalLM.from_pretrained(model_name)` with no `dtype` argument silently loads bf16
  weights under current `transformers`, which then crashes the instant they're combined with any
  float32 layer (an adapter, a value/reward head) — `RuntimeError: mat1 and mat2 must have the
  same dtype`. Fix at the call site with `dtype=torch.float32`; this isn't a version-pin problem
  (pinning `transformers` doesn't help, since the *checkpoint's own* dtype metadata is what
  changed how loading defaults behave), it's a missing explicit argument.
- **"Never actually trained/persisted" is the single most common headline bug class.** Week 2 (GRU
  tower trained fully in memory, never saved — downstream scripts silently reinitialize a random
  tower and nobody notices because nothing crashes), Week 3 (seq2seq decoder trained without
  shifting targets, so it learns to echo its input and collapses at real inference), Week 4 (a
  shipped checkpoint's state dict doesn't match the current adapter class — an extra layer was
  removed from the code after that checkpoint was saved), Week 6 (the reward model's head and the
  "SFT model" are both never actually trained, despite class names implying a real training stage
  happened). When a repo has a class or file named after a training stage (`RewardModel`,
  `SFTModel`, `trainer.py`), verify empirically that the stage ran and its output is actually used
  downstream — don't take the name's word for it. This is the finding worth spending the most
  verification effort on in every repo, because it's usually invisible until you actually run
  things end to end.
- **A script hardcoded to one dataset crashes the instant only a different one is present.** Week
  5's `create_manifest.py` and half of `explore_data.py` are hardcoded to CREMA-D-only paths with
  no existence check, while sibling functions in the *same file* correctly loop over all
  datasets. CREMA-D needed a multi-gigabyte git-lfs clone that was never actually done; only the
  much smaller EmoDB was available, and those hardcoded paths crashed immediately with
  `FileNotFoundError`. Worth grep-ing for hardcoded dataset names in any script that claims to
  support "all datasets."
- **Hardcoded/fabricated stats that don't reflect real local state.** Week 5's
  `print_dataset_summary()` prints fixed totals (7,442 / 535 files) regardless of what's actually
  on disk — confirmed 0 files present for one of them, function still claimed 7,442.
- **Batched autoregressive generation with the wrong padding side.** Week 6's tokenizer never sets
  `padding_side='left'`; a decoder-only model right-padded produces genuinely garbled output for
  whichever prompt in the batch is shorter after tokenization. Provable directly by toggling
  `padding_side` on the same batch and diffing the output — worth doing whenever a repo batches
  prompts of different lengths through a decoder-only HF model.
- **Project metadata that doesn't match the actual code.** Week 6's `pyproject.toml` lists Apple's
  `mlx`/`mlx.nn`/`mlx.optimizers` as dependencies (plus a fake org name, email, and GitHub URL)
  even though the entire codebase is plain PyTorch with no `import mlx` anywhere — confirmed the
  documented `uv sync` install command fails outright. Worth a quick sanity check of any
  `pyproject.toml`/`setup.py`/`package.json` metadata against the real imports before trusting the
  README's own install instructions.
- **The most-recently-named file isn't always the complete one.** Week 4's `final.ipynb` (name
  implies the polished version) has a broken single-example training loop; the real batched
  `DataLoader`-based training that actually produced one of the shipped checkpoints lives in
  `notebook.ipynb`, labeled the "earlier exploratory draft." Check which file's logic actually
  produced the artifacts that exist on disk, not which one sounds more finished.
- **Missing `requirements.txt` entries are the norm.** Every repo except Week 3 and Week 6 needed
  at least one manual `pip install` beyond what was listed: `fastapi`/`uvicorn`/`transformers`
  (Week 1), `datasets`/`redis`/`python-dotenv` version pins (Week 2), `torchvision`/`tqdm` (Weeks
  4 and 5), plus Week 5's `ffmpeg` listed as a pip package when `openai-whisper` actually needs the
  *system* `ffmpeg` binary (Homebrew/apt), which the pip package of the same name doesn't provide.
  Week 6 inverted this — its `pyproject.toml` over-specifies (see above) rather than
  under-specifies. Assume every repo's dependency list is wrong in some direction until a fresh
  venv proves otherwise.
- **Test suites, on the rare occasion they exist, are worth actually running.** Only Week 6 has
  one (`tests/test_ppo_components.py`) — real signal: 11/13 passed, and both failures were bugs in
  the *tests* (a wrong attribute path; an assertion that a PPO-clip loss must be positive, which
  isn't a real invariant with mixed-sign advantages), not in the code under test. Don't assume a
  red test means the implementation is wrong — check both directions before writing it up.

### Pass 2c — infrastructure standardization (done)

**Scope: a consistent project layout — Docker + Makefile + a walkthrough notebook — applied
independently to each of the six repos.** Structure only, no ML/architecture changes, no shared
code between repos (each stays fully standalone), and no Kubernetes work yet — Docker Compose today
is the future on-ramp to k8s (`kompose convert` or hand-written manifests, later), not something to
design for now.

Decided with the user going into this pass:
- **Docker containerizes both training and serving**, in every repo, even where a training
  container can't realistically be exercised in this sandbox (no GPU) — it should still build and
  be structurally correct for a future real GPU host.
- **READMEs get rewritten** so `make` targets become the primary documented commands — new scope
  compared to Pass 2a's "don't touch the README" rule, done deliberately here. The already-accurate
  raw commands from Pass 1/2a are kept as a short "what this target actually runs" reference under
  the make-target line, not deleted.
- A **walkthrough notebook** per repo narrates the happy path end to end — but per the Week 4
  lesson from Pass 2a (two notebooks silently diverging caused real bugs), it must import and call
  the real modules/scripts, never reimplement pipeline logic inline. Week 4 already has exactly
  this kind of notebook (verified working in Pass 2a) and doesn't get a second one.

Per-repo layout:
- One multi-stage `Dockerfile` at repo root (`base` → `train` / `serve` / `ui` stages sharing one
  dependency layer) — replacing Week 1's three separate Dockerfiles and Week 3's scattered
  per-folder ones. Week 6 installs via `uv` in the `base` stage, not raw pip, matching its
  already-fixed `pyproject.toml`/`uv.lock`. Where a repo has no real standalone "serve" step
  (Weeks 2, 4), the closest existing entrypoint fills that slot rather than inventing a new one.
- One `docker-compose.yml`, services named consistently (`train` profile-gated so a bare
  `docker-compose up` never trains by accident; `serve`; `ui` where applicable; plus any real
  dependency like Week 2's Redis, using the exact image/flags Pass 2a round 2 already validated —
  `redis/redis-stack-server:latest` with `--save "" --appendonly no`). GPU reservations present as
  a commented-out block (Week 3's existing pattern), inert here, ready for a real GPU host. Week
  3's specific disk-blowup bug (an unpinned CUDA-enabled `torch` pulled a multi-GB image on
  `linux/arm64` and nearly filled the host disk, found in Pass 2a) gets fixed here since its compose
  file is being rewritten anyway.
- One `Makefile` per repo, identical target names everywhere (`install`, `train`, `serve`, `test`,
  `docker-build`, `docker-up`, `docker-train`, `docker-down`, `clean`, `help`) — same verbs, guts
  differ per repo's real pipeline. No new capability gets built just to fill a slot a repo doesn't
  naturally have.
- A `walkthrough.ipynb` at repo root for Weeks 1, 2, 3, 5, 6 (Week 4 already has one) — markdown
  narration plus cells that call the real functions/scripts in sequence, mirroring the README's
  Usage stages.

Rollout: six parallel subagents, one per repo, same proven pattern as Pass 2a. Each agent smoke-
tests its own `serve`/`ui` stage locally and attempts a `train` image build for a structural check
without forcing a multi-GB CUDA pull on this sandboxed host if that would blow the disk (document
instead of forcing it through — the same judgment call Pass 2a round 2 already made successfully
with Week 3's Docker Compose). Watch disk space and Docker Desktop's daemon health across the
parallel run — its containerd store corrupted once already under concurrent load during Pass 2a
and needed a manual restart; that's an orchestrator-level watch, not something each agent should
try to self-diagnose.

**Operational lessons for running six parallel subagents in this environment** (learned the hard
way across Pass 2c/2d/2e, worth reading before launching another multi-repo round):
- **Subagents very often background a long Docker build/nbconvert run as a plain shell command
  (not via their own `run_in_background: true`), then end their turn waiting for a completion
  notification that will never arrive** — because only the orchestrator sees real OS-level process
  state, and a plain backgrounded shell doesn't notify anyone. When an agent reports "waiting for a
  background X to finish," check the actual process directly (`ps aux | grep ...`, `docker ps`,
  tail its log file) rather than trusting the agent's belief that it's tracked — then resume it
  with the concrete result (or tell it to poll `ps`/logs directly, or just re-run in the
  foreground) rather than waiting passively yourself.
- **Fresh `general-purpose` agents in this environment can spontaneously enter plan mode** (not by
  deliberately calling a plan-mode tool — the session context restricts them to read-only) and then
  have no way to exit it themselves, wasting the entire run. If this happens, don't keep resuming
  the same stuck agent — read whatever plan it *did* manage to write, then launch a **fresh**
  `general-purpose` agent (not a `fork` — this user prefers to avoid forking for token-cost reasons)
  with that plan embedded directly in the prompt so it can execute without redoing the research.
  This stopped recurring once the user disabled plan mode at the environment level.
- **Session/API rate limits are account-wide, not per-agent** — when one subagent fails with "hit
  your session limit," every other concurrently-running subagent will fail the same way around the
  same time. Don't relaunch anything until checking the actual current time against the reported
  reset time; relaunching before the reset just repeats the failure. Uncommitted work sitting in a
  repo's working tree is safe to leave alone across an arbitrarily long wait — nothing is lost by
  waiting.
- When resuming an interrupted agent (session limit or otherwise), **check `git status`/`git diff
  --stat` yourself first** and tell the resumed (or freshly-relaunched) agent exactly what's already
  done — agents that re-discover their own prior progress from scratch waste real time re-deriving
  what a two-line summary would have told them.

**Environment gotcha found during Pass 2c (Week 3):** a plain `uv venv --python 3.11 .venv-run`
(no `--seed`) creates a venv with no `pip` binary at all — a Makefile's `install` target that
shells out to bare `pip install` will silently fall through to a global/Homebrew `pip` and install
into the wrong site-packages, leaving the venv's own Python with nothing (`ModuleNotFoundError` on
the first real import). Use `uv venv --seed` when a Makefile target needs a real `pip` inside the
venv, or use `uv pip install` throughout instead of bare `pip`.

**Recurring pattern found across Pass 2c + a follow-up MPS/container round (Pass 2d):**
- **Unpinned `torch` pulls a full CUDA build on `linux/arm64` and bloats images 5-10x.** Found and
  fixed in Weeks 1, 2, 3, 5 (Week 3 first, disk-blowup bug; Week 2 went 8.54GB→1.81GB; Week 5 went
  10GB→3.33GB). Fix: in the Dockerfile's `base` stage, `pip install --index-url
  https://download.pytorch.org/whl/cpu torch torchvision torchaudio` *before* the general
  `pip install -r requirements.txt`, so the unpinned line in requirements resolves as
  already-satisfied instead of re-resolving a CUDA wheel. Week 6 (`uv`-managed) never had this bug.
  Check any repo's image size after a rebuild before assuming it's fine.
- **MPS device selection is worth auditing per repo, but benchmark before assuming it helps.**
  Fixed real gaps in Weeks 1 (CBOW), 4 (clip.py + both notebooks — was dead/missing code), 5
  (Whisper — the `openai-whisper` package only checks CUDA, never MPS), 6 (PPOTrainer, no LoRA/PEFT
  incompatibility materialized). Week 2's CBOW got a genuine ~3.7x speedup; its GRU dual-encoder was
  *correctly left CPU-only* after benchmarking showed MPS was ~2.2x *slower* there (per-op dispatch
  overhead on tiny unbatched tensors) — don't add MPS just because it's available, measure first.
  Week 3's classifier trainer is still honestly CPU-only (a pre-existing hardcoded device line, not
  a new bug) while its seq2seq trainer and serving API correctly use MPS — a repo can be
  inconsistent internally and that's fine to document rather than "fix" uniformly.
- **MPS-trained checkpoints can fail to load in a non-MPS environment.** Week 1 hit this directly:
  after adding MPS training, checkpoint files carried `mps:0` tensor metadata that failed to
  deserialize inside a (CPU-only, Linux) Docker container. Fix: `map_location='cpu'` on every
  `torch.load()` that might load an MPS-trained checkpoint outside of MPS.
- **Docker Desktop's VM has a real, fixed memory ceiling (~7.65GB on this host).** Week 6's full
  4-model PPO pipeline (SFT+Policy+Reward+Value, all float32) reliably OOMs in-container (exit
  137) — reproduced on demand, confirmed environment-side not code-side. Don't chase this; verify
  with scoped single/two-model checks instead (confirm dtype/padding/device directly via a targeted
  `docker run ... python -c "..."` that doesn't load everything at once).
- **A missing `.dockerignore` can silently leak a local `.venv` into the build context.** Found in
  Week 4 — nearly doubled the `serve` image size. Any repo getting Docker for the first time needs
  one from the start, not as an afterthought.

### Pass 2e — deep repository structure standardization (done)

**Scope: physically reorganize each repo's internal directory layout onto one common convention,
applied independently per repo.** This is explicitly higher-risk than Pass 2c — it moves existing
code and updates every path reference that touches it (imports, Dockerfile `COPY`s, `docker-
compose.yml` volumes, Makefile targets, README commands, notebook cells), rather than only adding
new files alongside what's already there. The user was told this overlaps with what was originally
reserved for the later single-repo Pass 3 and explicitly chose to do it now anyway, across all six.
Because of the risk, each repo's Pass 2c (+ MPS) work was committed as a clean checkpoint
immediately before this pass started — if a restructuring goes wrong, revert to that commit rather
than losing the verified container/MPS work underneath it.

**The common convention** (apply per repo, adapting to what that repo actually has — don't force a
folder that has nothing to hold):
- `src/` — all importable Python code/packages (whatever subpackages already exist move under here,
  keeping their names, e.g. `etl/` → `src/etl/`).
- `scripts/` — standalone CLI entrypoints that aren't meant to be imported as a library (only where
  this distinction is genuinely clean — don't force it if a repo's scripts are already reasonably
  imported from the walkthrough notebook).
- `notebooks/` — every `.ipynb` file (`walkthrough.ipynb`, and whatever else already exists —
  `final.ipynb`, `rosh.ipynb`, etc.; rename/consolidate stray or personally-named notebooks after
  actually reading their content, don't delete blind).
- `checkpoints/` — all trained model artifacts (`.pth` files, tokenizer dirs, anything a training
  stage produces), consolidating any redundant same-purpose dirs a repo already has (e.g. Week 3's
  `model_data/` + `model_outputs/`).
- `data/` — raw/processed datasets, consolidating any redundant same-purpose dirs (e.g. Week 3's
  `data/` + `datasets/` — check what each actually holds before merging, don't assume).
- `ui/` — the interactive frontend (Streamlit app), renamed consistently from whatever a repo
  currently calls it (`streamlit-app/`, `streamlit/`, etc.) — every repo's UI in this portfolio
  happens to be Streamlit, `ui/` keeps the name tool-agnostic.
- `assets/` — README screenshots/plots (already this name in every repo since Pass 1 — no change
  needed).
- Root stays: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `Makefile`, `README.md`,
  `TODO.md`, `requirements.txt`/`pyproject.toml`, `.gitignore` — these were just freshly
  standardized in Pass 2c and are low-value/higher-risk to relocate further; leave them at root.

**Non-negotiable per repo:** after moving anything, update every reference that pointed at the old
location — imports, `Dockerfile` `COPY` lines, `docker-compose.yml` volumes, Makefile recipes,
README's Usage commands, notebook cells, any hardcoded path string in a script — then re-run the
full verification sequence Pass 2c already established (`make install && make test`, `make serve`/
local training smoke test, `make docker-build`, `docker compose up`) to *confirm* nothing broke,
not assert it. Leave everything uncommitted for user review, same as every other pass.

**Result, per repo** (full detail in each repo's `TODO.md`, "Pass 2e" section):
- **Week 1**: `src/{etl,models,prediction,serving}/`, `ui/`, `checkpoints/`, `notebooks/`. Found
  moving `walkthrough.ipynb` into `notebooks/` broke a `Path.cwd()` assumption (nbconvert defaults
  kernel cwd to the notebook's own directory) — caught by actually running it, fixed. Also found
  and deleted a stale, dimensionally-incompatible (32-dim vs. real 128-dim) duplicate predictor
  checkpoint under the old `prediction/` that had been dead weight independent of this pass.
- **Week 2**: `src/{cbow,data_prep,models,retrieval}/`, `scripts/`, `config/`, `checkpoints/`,
  `data/`, `notebooks/`. Found and fixed a stale tokenizer path in `create_embeddings.py` that
  actually predated this restructuring (confirmed via `git show` against the pre-move commit) —
  same bug class as everything else in this pass, fixed for consistency even though not on the
  critical path.
- **Week 3**: `src/{models,services,transformer,utils}/`, `ui/` (was `streamlit/`), `checkpoints/`
  (consolidated `model_data/` + `model_outputs/`), `notebooks/`. `data/` (raw MNIST cache) and
  `src/datasets/` (Dataset classes) correctly kept separate — different things despite similar
  names. Hit a real hang during verification: a classifier training smoke test blocked
  indefinitely: root cause was matplotlib's interactive `macosx` backend blocking on `plt.show()`
  when run as a plain script (works fine under Jupyter, which uses `Agg`) — fixed with
  `matplotlib.use("Agg")` before `pyplot` import. Also found stale path references in three
  package-level `README.md` files the Pass 2c session never touched.
- **Week 4**: `src/clip.py`, `checkpoints/`, `data/images/`, `notebooks/` (no new walkthrough
  notebook — already had `notebook.ipynb`/`final.ipynb`). Found a genuinely new bug: no
  `.dockerignore` had ever existed, letting a stray `.venv/` leak into the build context — fixed,
  rebuilt. Verified by running both real notebooks to completion with SHA256-verified checkpoint
  integrity before/after.
- **Week 5**: `src/{...,data_prep/}`, `ui/app.py`, `checkpoints/`, `data/` kept deliberately
  separate from `datasets/` (generated outputs vs. raw third-party corpora — different things).
  Clean pass, nothing found stale.
- **Week 6**: `src/mlx_week6_rlhf_ppo/` (proper src-layout, matching its existing installable
  `pyproject.toml`). Real gotcha: `[tool.setuptools.packages.find] where = ["src"]` alone isn't
  enough for src-layout — also needs `[tool.setuptools] package-dir = {"" = "src"}`, or `uv sync`
  silently builds an empty/wrong wheel. `rosh.ipynb` looked like stray scratch content but the
  README's own file tree documents it by name as the original exploratory notebook the package was
  refactored from — kept, moved to `notebooks/`, not deleted (read content before deciding, don't
  delete blind).

## README cross-reference audit (done, 2026-07-30)

The user asked to cross-reference Pass 1's README work against the actual source material:
`../Presentations/MLX-weekN_*.pdf` (real team-authored decks, sibling to `../Slides/`, not
previously covered by the README methodology above — only weeks 1, 2, 4, 5, 6 have one; no deck
exists for Week 3) and `../ai-residency-teams.txt` (the real team roster per week). Ran as six parallel
audit agents, then six parallel fix agents.

**Headline finding: every week 1–6 was a team project, but Weeks 2–6's READMEs all said "Developed
individually."** Only Week 1 credited a team correctly. Fixed all five to name the real team
(`ai-residency-teams.txt` roster, cross-confirmed against a roster slide inside the actual deck/course
slides for each week) in the same phrasing convention Week 1 already used. Week 6 needed a
nuance kept in the fix: its own presentation is solo-narrated by Rosh in first person — real
evidence the from-scratch PPO implementation was his own extra effort, not a team build — so the
credit line names the team but says so explicitly, rather than overclaiming collaborative authorship
of the code.

**Other real findings, not just the credit line:**
- Week 1: figure citations were *under*-credited, not over — only 4 of at least 10 lifted figures
  were listed in Acknowledgments. Also found one factual mislabel: Figure 7's tokenizer example used
  embedding values that really belong to the word "orphism" on the actual slide, relabeled
  "bardiwac" in the README — fixed to match the real source (don't confuse this with the separate,
  correctly-unrelated "bardiwac" wine-glass example used elsewhere in the same doc for the
  distributional hypothesis — that one wasn't part of the mislabel and was deliberately left alone).
- Week 2: the Results section claimed "no team presentation with concrete numbers to draw on" —
  false; the deck has a real retrieval screenshot with real similarity scores that independently
  corroborate the tower-never-wired-into-retrieval bug the README's own re-run demonstrates.
- Week 4: the team's own deck has an uncredited original architecture diagram and real sample
  captions from the team's actual trained model; Figure 3 was generically attributed to "course
  slides" but actually comes from a second, different course deck specifically.
- Week 5: the team's deck has a genuinely better artifact than what shipped — a real Streamlit
  screenshot with actual recorded audio ("Rush"→"Rosh"), vs. the README's own idle-state screenshot
  (no live mic in this environment) — noted as a nice-to-have swap, not done (would need PDF image
  extraction, out of scope for a text-fix pass).
- Week 3 and Week 6: figure/background attributions checked out as accurate on inspection — no
  fix needed there beyond the credit line.

**Lesson for future work in this environment:** the user's voice input garbles badly and
unpredictably (this session opened with "presentations? directory, besides directory" for
"Presentations directory, beside this directory," and a later message needed a full second
clarification round for "when I move this to a Kubernetes repository... a separate service as the
frontend" which came through as unrelated word salad the first time). When a message has a clear
core ask plus a garbled tail, confirm the clear part and explicitly ask for the garbled part to be
retyped rather than guessing — especially before starting multi-repo work based on the guess.

## Dependency modernization: uv migration + version upgrades (done, 2026-07-30)

Two user requests handled back to back: "migrate away from pip and managing venvs to using uv" (full
depth: `pyproject.toml` + committed `uv.lock`, chosen explicitly over the lighter "keep
requirements.txt, just swap venv commands" option), then "these were pinned at the time of
creation... can we use newer versions."

**uv migration, all six repos:** consolidated each repo's `requirements*.txt` files (Week 1 had
`requirements.txt` + `requirements_db.txt` + a dead, never-installed `ui/requirements.txt`, since
deleted; Week 3 had three) into one `pyproject.toml`, `[tool.uv] package = false` on every repo
except Week 6 (already a proper installable package), `uv lock` + `uv sync` replacing manual
`python3 -m venv`/`pip install`, every Makefile target now `uv run ...`, every Dockerfile `base`
stage now `pip install uv` → `COPY pyproject.toml uv.lock` → `uv sync --frozen --no-install-project`
→ `COPY . .` → `uv sync --frozen` (mirrors Week 6's pre-existing pattern, which is why it was used
as the reference template).

**New standard fix for the CUDA-torch-bloat bug, now that repos are uv-managed:** the old fix
(`pip install --index-url https://download.pytorch.org/whl/cpu torch ...` before the general
install, see Pass 2c/2d notes above) doesn't apply the same way under `uv sync`. The uv-native
equivalent, confirmed working:
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]   # if present
torchaudio = [{ index = "pytorch-cpu" }]    # if present

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```
**Important, easy to miss:** this bug can reappear even in an already-`uv`-managed repo. Week 6's
`pyproject.toml` (already existed, already uv-based, previously confirmed CUDA-free per Pass 2c/2d
notes above) got its `uv.lock` regenerated this session (adding `streamlit` for its new UI) and
silently picked up ~20 `nvidia-*`/`cuda-*` packages worth several GB, because a newer `torch`
release than the one previously locked resolves those as separate platform-conditional dependencies
unless pinned. The old "Week 6 never had this bug" note in Pass 2c/2d above was true *at the time*,
not permanently — re-check `grep -c "nvidia-cu\|cuda" uv.lock` after *any* `uv lock`/`uv sync
--upgrade` on any repo, not just the first time a repo goes uv-native.

**Version upgrades:** most repos (2–6, mostly) were already floor-pinned (`>=`) or unpinned, so they
auto-resolved to current versions the moment `uv lock` ran — nothing further to do there. Real work
was concentrated in two repos with exact (`==`) pins:
- **Week 1** — heavily exact-pinned, with a *documented, deliberate* coupling: `torch==2.2.0` +
  `transformers==4.44.2` were pinned together because newer `transformers` needs `torch>=2.4` and
  silently turns the CBOW model's `PreTrainedModel` usage into a dummy stub otherwise. Bumped both
  together (→ torch 2.13.0, transformers 5.14.1) and *actually re-verified* the specific failure
  mode doesn't reproduce (instantiation, forward/backward pass, save/load round-trip on the real
  `CBOWNS` class) rather than trusting a clean install — this is exactly the kind of upgrade that
  can break silently (see the Qwen3 bf16-default story in "Recurring patterns" above) so it needs
  real re-exercise of the risky code path, not just `pip check`. Also dropped `setuptools<81`
  (a `wandb==0.16.3`-era `pkg_resources` workaround) after confirming a current `wandb` no longer
  needs it. Verified further with a live served prediction (before/after identical: `27.56`), a
  ~60s real CBOW training smoke test with decreasing loss, and a full Docker build/run/curl.
- **Week 2** — three exact pins (`datasets`, `huggingface_hub`, `redis`) bumped to current. Caught a
  real breaking change: `redis-py` 8.x restructured `FT.SEARCH`'s response shape (flat array →
  nested dict) — fixed the actual parsing code in `src/retrieval/query_engine.py` to branch on both
  shapes, re-verified identical real results came back afterward. Worth remembering as a class of
  bug: a dependency major-version bump can change a *response format*, not just an API signature —
  re-run real integration calls after any client-library bump, don't just check `import` succeeds.

## UI + backend-API build-out (done, 2026-07-30)

User asked for full directory-structure uniformity across all six repos, specifically calling out
that Weeks 2, 4, 6 needed their own Streamlit UI like Weeks 1/3/5 already had (context: eventual
plan is each week becomes its own Kubernetes microservice with a separate frontend service later —
see "ai-residency-hub" below for where that's actually headed).

**New Streamlit UIs, Weeks 2/4/6** — each reuses real existing code, doesn't reimplement inference:
- Week 2: refactored `query_redis_ann.py`'s CLI logic into `src/retrieval/query_engine.py`
  (`run_query()` + loaders), both the CLI and the new `ui/app.py` call the same function.
- Week 4: extracted the real captioning pipeline from `notebooks/notebook.ipynb` (the one with the
  actual batched training loop, per Pass 2a notes above — not `final.ipynb`) into `src/caption.py`,
  which `ui/app.py` calls. Surfaces the already-known checkpoint/adapter mismatch bug as a graceful
  on-page error rather than hiding or "fixing" it.
- Week 6: since no trained PPO checkpoint exists anywhere in the repo (confirmed by checking disk,
  not assumed), the UI is a single honest generation demo using only `PolicyModel`, with an explicit
  on-page disclosure — deliberately *not* a fake base-vs-PPO comparison, which would have implied a
  meaningful trained difference that doesn't exist.

All three wired into that repo's existing Docker `ui` stage / compose `ui` service / Makefile `ui`
target pattern (matching Weeks 1/3/5 exactly), each verified by actually running it (not just
writing it) — real predictions/captions/generations confirmed, including the Docker build for Week
4. Also fixed incidentally: a dead-code `NameError` in Week 2's CLI found during the refactor, a
`.dockerignore` glob gap in Week 4 (`.venv/` didn't match a differently-named scratch venv
`.venv-run`, leaked into a build context, doubled image size), Week 6's stale `docs/` folder (three
ASCII diagrams predating the README's Mermaid rewrite, unreferenced anywhere, deleted).

**Backend FastAPI APIs, Weeks 2/4/5/6** — added afterward, once building `ai-residency-hub` (below)
made it clear the frontend there needed a real HTTP contract for every week, not just the two
(Weeks 1, 3) that already had one. Each is a thin wrapper around the *same* function the week's own
`ui/app.py` already calls — `POST /query` (Week 2), `POST /caption` (Week 4), `POST /transcribe`
(Week 5), `POST /generate` (Week 6) — new `api` Docker stage/compose service/Makefile target
alongside each repo's existing `train`/`serve`/`ui` ones (never repurposed `serve`, which already
means something specific and different in several of these repos — Week 4's `serve` is the Part-1
zero-shot demo, Week 6's `serve` runs pytest). Portfolio convention for binary input over JSON:
base64-encode in a pydantic field (`image_data`, `audio_data`), not multipart upload — matches how
Weeks 1 and 3's pre-existing APIs already did it.

## ai-residency-hub (new 7th repo, done, 2026-07-30)

A new repo, `/Users/rosh/mlx/ai-residency-hub`, sibling to the six weeks. User's ask: a Streamlit
frontend repo using the six weeks as submodules, serving all six in Docker containers plus one
frontend container — explicitly the local-compose on-ramp to the eventual Kubernetes picture (each
week as its own microservice, one separate frontend service), not something to build k8s manifests
for yet.

**Design decisions, confirmed with the user before building:**
- Submodules use **local relative paths** (`../ai-residency-weekN-...`), not GitHub URLs — none of the six
  repos have been pushed anywhere. This means `ai-residency-hub` **will not clone correctly for
  anyone else** until the six repos are pushed to their real GitHub remotes and `.gitmodules` is
  repointed — documented prominently in the hub's own README, don't let a future session forget it.
  Modern git blocks the `file://` transport by default; use `git -c protocol.file.allow=always
  submodule add/update ...` (a one-off flag, not a persisted config change) for any local-path
  submodule operation.
- Frontend is a **unified single-page app** (one Streamlit app, one page per week, each a thin HTTP
  client calling that week's own backend directly), not a launcher/iframe hub — the more expensive
  option, chosen deliberately over the cheaper one.
- All six repos' current uncommitted session work was committed locally (not pushed) *before*
  adding submodules, since a submodule always pins a specific commit — pointing at a stale
  pre-session commit would have silently dropped all the README/uv/UI work above.

**Real problems found during the build, not anticipated going in:**
- **Gitignored trained artifacts are physically absent from a fresh submodule checkout.** Git only
  clones what's tracked; large/binary checkpoints (and in Week 2's case, `data/tokenized_triples.json`
  needed only for one-time Redis indexing) are correctly gitignored in every repo except Week 4
  (whose checkpoints are small enough to be intentionally committed). Fix: mount those specific
  paths in the hub's `docker-compose.yml` from the **sibling original repo** on the host
  (`../ai-residency-weekN-.../checkpoints/...`), not from the submodule's own copy — documented as a known,
  host-specific workaround in the hub's README; a real multi-machine deployment would need actual
  artifact hosting (git-lfs, release assets, DVC) instead. This also broke Week 5's Docker *build*
  itself, not just a runtime mount — its original `api` stage bakes checkpoints in via `COPY . .`
  at build time, which only works when building from a directory that actually has them.
- **Found and fixed a real, separate bug while investigating this:** Week 3's own `.gitignore` was
  accidentally excluding `.dockerignore` itself, so that file had never been committed in the first
  place — invisible until a fresh submodule clone of Week 3 conspicuously had no `.dockerignore` at
  all. Fixed directly in the Week 3 repo (not just worked around in the hub), a genuinely portable
  fix independent of the hub's existence.
- **Docker Desktop's memory ceiling, already documented for Week 6 alone in Pass 2c/2d notes above,
  reproduces at the hub level under sustained multi-container load** — confirmed twice: Week 4 was
  OOM-killed (exit 137) after running ~25 hours alongside the other five, and restarting Week 6
  immediately pushed Week 4 (or whichever container) back over the edge again. Not a hub bug —
  `docker compose up -d weekN` restarts just the killed one; documented in the hub's own README
  rather than treated as something to "solve" by shrinking the models.
- Each backend independently verified with a real request through the hub (not just a health
  check): Week 1 real score prediction, Week 2 real ranked retrieval (after a one-time
  `docker compose exec week2 uv run python src/retrieval/save_doc_embeddings_to_redis.py` to
  populate the hub's own, separate, starts-empty Redis — different data store from the original
  Week 2 repo's own persistent one, documented in the hub README as a one-time setup step), Week 3
  real classifier+seq2seq output, Week 4 real caption, Week 5 real base-vs-fine-tuned transcription
  (exact match to the already-known real output), Week 6 real generation with its honest
  no-trained-checkpoint disclosure.

Layout: `week1/` … `week6/` (submodules), `frontend/` (the Streamlit app — lightweight, only
`streamlit`+`requests`, no ML dependencies at all, since every model actually runs in its own
week's backend container), `docker-compose.yml`, `Makefile`, `README.md`. Single commit `2f165e0`,
nothing pushed.

## Kubernetes (Helm) deployment for ai-residency-hub (done, 2026-07-30)

User installed a single-node Kubernetes cluster on their Mac Studio via Docker Desktop and asked for
a Helm chart to deploy the portfolio hub there — the concrete first step toward the "eventual
Kubernetes picture" the hub's own README already gestured at. Built `helm/ai-residency-hub/` at the
hub repo root: one data-driven chart (`Deployment`+`Service` generated per entry in `values.yaml`'s
`backends` map, via a single `range` template) rather than seven hand-written template pairs — the
six backends are close enough to identical that copy-pasted per-service templates would've just been
duplication. Reuses the hub's existing locally-built images (`imagePullPolicy: Never` — Docker
Desktop's Kubernetes node shares the exact same `dockerd` as `docker build`, so no registry
push/pull is needed at all). Frontend is `LoadBalancer` (Docker Desktop auto-binds these to
localhost, so `localhost:8500` works with zero extra steps); the six backends are deliberately
`ClusterIP`-only with documented `kubectl port-forward` commands for direct debugging — a real user
decision point, resolved by asking rather than assuming (this matches a real production shape:
external traffic only ever hits the frontend). Every `docker-compose.yml` volume mount (the
gitignored-checkpoints-mounted-from-sibling-repo pattern documented under "ai-residency-hub" above)
was translated 1:1 into a `hostPath` volume, rendered via Helm's `tpl` function against one
`hostRoot` value so the whole mount set can be repointed by changing a single line.

**Real infrastructure problem hit on first install, not anticipated going in — worth remembering for
any future from-scratch k8s work in this environment:** with no memory requests set, all 8 pods
(6 backends + frontend + week2-redis) starting simultaneously produced an infinite crash loop across
the *entire* namespace, not just the heaviest container. Root cause, confirmed by direct comparison:
docker-compose's memory-ceiling problem (already documented under "ai-residency-hub" above — Week 4
OOM-killed after ~25 hours) is *survivable* under compose because a killed container just stays
stopped. Under Kubernetes, a Deployment's default `restartPolicy: Always` immediately recreates an
OOM-killed pod, which immediately re-competes for the same scarce memory and gets OOM-killed again —
strictly worse, and fast (minutes, not hours). The thrash got bad enough that the cluster's own API
server started timing out (`kubectl get pods` itself hanging), which in turn made `docker stats` and
even plain `docker ps` sluggish — a memory-starved node degrades observability precisely when you
need it most, so diagnosis had to lean on whatever *did* respond rather than on live monitoring.

Fix applied: every container in the chart now has `resources.requests.memory` (`memoryRequest` per
entry in `values.yaml`, rough estimates — no metrics-server on this cluster to measure real RSS),
deliberately sized so all 8 requests sum to *more* than the node's default allocatable memory. This
makes the scheduler leave whatever doesn't fit cleanly `Pending` instead of the whole set thrashing —
turns an active failure mode into a passive, self-explaining one. On Docker Desktop's stock memory
allocation (~7.55GiB allocatable), 6 of 8 fit; the user then bumped Docker Desktop's own Kubernetes
memory limit to 24GiB (of the Mac Studio's 32GiB), comfortably above the ~11.5GiB the 8 requests sum
to, and all 8 pods came up and stayed `Running` with zero restarts — confirmed with a real prediction
through week1 and real `/docs` responses from week4 and week6, all reached through their in-cluster
Services rather than assumed. One more gotcha on the way there: Docker Desktop restarts its whole VM
to apply a memory-limit change, which bounces every pod (normal reschedule, not a bug) — pods that
were `Pending` before the resize didn't get automatically re-evaluated against the new, larger
allocatable memory afterward and needed a `kubectl delete pod` nudge to trigger a fresh scheduling
attempt.

**A second, unrelated discovery during the same debugging session, also worth remembering:** the
hub's own leftover `docker compose up` stack from earlier in the session, *and* a completely
separate stray standalone `docker compose up` stack for `ai-residency-week3-vision-transformers` specifically (left
running from independently verifying that repo earlier), were both still running and silently
competing with the new Kubernetes pods for the same host memory — a real contributor to the crash
loop above, not just the missing resource requests alone. `docker ps` distinguishes the two:
Kubernetes-managed containers carry a `k8s_<container>_<pod>_<namespace>_...` name prefix; anything
without that prefix predates or is unrelated to the cluster. Both stray stacks were torn down with a
plain `docker compose down` from their own repo directory. General lesson for this portfolio: a
`docker compose up` run in *any* individual week repo (done routinely throughout Pass 2a–2e for
per-repo verification) can outlive the session that started it and silently compete with a later,
unrelated deployment on the same host — worth an explicit `docker ps` check before diagnosing a
fresh memory problem as solely the new thing's fault.

Chart also gained a `make helm-install`/`helm-upgrade`/`helm-uninstall`/`helm-status` set of Makefile
targets (same verb convention as the compose targets) and a full "Kubernetes (Helm)" README section
covering all of the above. **Commit state:** all of this — `helm/ai-residency-hub/`, the Makefile
targets, the README section — is uncommitted in the hub repo as of this writing (routine for this
portfolio's work sessions, see the top-level commit-state note above); the six week repos and their
submodule pins are untouched by this pass.

### Pass 2b — broader hygiene checklist (SKIPPED 2026-08-02 — see the Pass 3 section)

Applies to all six regardless of what's in their individual `TODO.md`:

- [ ] `requirements.txt` completeness *and* correctness — verify with a fresh venv, install, then
      actually run every entry point script. See the recurring-patterns list above for exactly
      what's been wrong so far; check each repo fresh anyway, don't assume it's fine because an
      earlier repo's fix works here too.
- [ ] No CI (GitHub Actions), in any of the six.
- [ ] No LICENSE file, in any of the six.
- [ ] Empty GitHub repo descriptions/topics, for all six.
- [ ] No live demo link, for any repo.
- [ ] Dead/commented-out code and style consistency generally — the `simplify` skill fits well
      here.

## README diagram screenshot pass (done, 2026-07-30)

User request: instead of recreating source diagrams in Mermaid (the original Pass 1 approach — see
"Diagrams" under README methodology above, now amended to reflect this), extract and embed the
**real original diagram** wherever one genuinely exists — a course slide, a team presentation deck,
or a paper cited in that repo's own References section — followed by prose on how the repo
implements what the diagram shows. Diagrams with no real source (each repo's own invented System
Architecture data-flow diagram, UML sequence diagrams describing that repo's own code) stay Mermaid.
A follow-up user instruction mid-pass ("download all the papers and look for relevant diagrams")
broadened this from "swap diagrams already suspected to have a source" to "actually fetch and
inspect every cited paper's real figures against every remaining Mermaid diagram, not just the
obvious candidates" — run as an explicit round 2 per repo after round 1 landed.

Run as six parallel round-1 subagents (one per repo) followed by six parallel round-2 subagents
(paper sweep). Round-2 agents were fresh, not resumed — attempting to `SendMessage` a round-1
agent by its raw `agentId` after it had already completed and reported failed ("No transcript
found"); background subagents in this environment don't stay resumable purely by ID once done, so
round 2 had to be full fresh `general-purpose` agents, each hand-briefed in the prompt with exactly
what round 1 already changed so they didn't rediscover it.

**Result: 30 diagrams across the six repos swapped from Mermaid recreations to real extracted
originals**, all left uncommitted for review (confirmed via `git status` after — only `README.md`
modified plus new untracked files under each repo's `assets/`, and Week 2's `.gitignore` for the
blanket-`*.png` fix below):

- **Week 1** (11 swapped: Figures 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12) — mostly course-slide sourced;
  Figure 8 (CBOW/Skip-gram) and Figure 11 (country–capital analogy PCA plot) came from the real
  word2vec papers directly (Mikolov et al. 2013a and 2013b) since the slides turned out to be
  redraws of those exact figures. Figure 4 (feature-type taxonomy, inspiration scattered across 3
  slides, no single lift), Figure 13 (pipeline architecture), Figure 14 (sequence diagram) stayed
  Mermaid. Also fixed a stale Acknowledgments paragraph that still credited the now-image Figures
  13/14 as slide-adapted and never listed Figure 10 at all.
- **Week 2** (7 of 8 swapped: Figures 1–7) — course-slide sourced, plus Figure 7 (System
  Architecture) swapped in round 2 for a real composite of two team-presentation-deck slides after
  confirming (not assuming) it was the same technical data-flow, not just thematically similar.
  Figure 8 (sequence diagram) stayed Mermaid — the deck's own diagram is a flowchart, not UML
  sequence notation, so nothing to swap it for. Found and fixed a real, separate bug along the way:
  a blanket `*.png` `.gitignore` rule that would have silently excluded the new assets (same bug
  class as Weeks 3/5 in Pass 1's "Getting real artifacts" section above) — added the
  `!assets/*.png` exception. Also found Figure 3 had a real slide source but had never been
  credited in Acknowledgments at all (not just under-credited) — fixed.
- **Week 3** (4 swapped: Figures 1, 2, 3, 4) — Figure 3 (single encoder block) came from round 2's
  paper check: Vaswani et al.'s own Figure 1 (encoder half), chosen deliberately over ViT's diagram
  because this repo's `encoder.py` uses post-norm like the original Transformer paper, not ViT's
  pre-norm. Figure 2 (ViT) uses the real Dosovitskiy et al. Figure 1 directly rather than the
  slide's redraw of it. Figures 5 (System Architecture) and 6 (usage flow) stayed Mermaid — no
  team presentation deck exists for Week 3 at all, and neither figure recreates any cited paper.
- **Week 4** (4 swapped: Figures 1–4) — Figure 1 (CLIP contrastive pretraining) from the real CLIP
  paper (Radford et al. 2021) directly; Figures 2–3 from the two separate course-slide decks
  (confirming a prior audit's correction that Figure 3 specifically comes from the second deck,
  `Week 4 Colbert Thinking.pdf`, not "course slides" generically); Figure 4 from the team's own
  presentation deck. Round 2 checked Vaswani, PaliGemma, and LLaVA papers directly against the
  three remaining diagrams and found a close conceptual cousin (PaliGemma's own prefix-LM masking
  figure) but judged it a different diagram scope, not a genuine substitute — correctly left alone.
- **Week 5** (2 swapped: Figures 2, 3) — both from the real Whisper paper (Radford et al. 2022)
  Figure 1's two panels ("sequence-to-sequence learning" and "multitask training format"), which
  the team's own slides already reproduce full-page. Figure 1 (waveform→FFT→mel-spectrogram
  flowchart) stayed Mermaid after both rounds — checked page-by-page against both slide decks and,
  in round 2, against Attention Is All You Need, EmoDB, and CREMA-D (via real figure captions,
  since the CREMA-D PDF itself is paywalled/gated with no accessible mirror) — no single diagram
  anywhere depicts that specific transform as a process flow.
- **Week 6** (2 swapped: Figures 1, 2) — the three-stage RLHF diagram from Stiennon et al. (2020)
  Figure 2 (matching the README's own citation) and the LoRA weight-update diagram from Hu et al.
  (2021) Figure 1, both extracted from course-slide screenshots that already faithfully reproduced
  those exact paper figures. Round 2 directly checked all remaining cited papers (Schulman et al.
  PPO, Christiano et al., Ouyang et al. InstructGPT) against Figures 3–5 and confirmed none has a
  diagram detailed enough to match this repo's own PPO-step/package-layout/sequence diagrams.

**Confirmed pattern across every repo:** an "invented" diagram (documenting this repo's own code,
file layout, or call sequence) never has a real external match, and every round-2 agent correctly
declined to force one — a genuine paper or slide existing somewhere is not sufficient grounds for a
swap; the diagram has to be faithfully describing that specific external figure, not just thematically
adjacent to it (Week 2's Figure 2 vs. Bromley et al.'s Siamese-network diagram, and Week 4's Figures
5–7 vs. PaliGemma/LLaVA, are the clearest documented examples of this judgment call going the right
way — noted in each repo's own `TODO.md` in more detail).

**Commit state:** all six repos' changes from this pass are uncommitted, same as every other pass in
this portfolio — left for user review.

## README voice, rigor, and structure pass (done, all six repos, 2026-07-30/31)

Four rounds of direct, iterative user feedback on Week 1's README specifically, done inline (not
via parallel subagents) since it needed real back-and-forth calibration before being worth rolling
out. Once calibrated, the same four-part treatment was applied to Weeks 2–6 as five parallel
`general-purpose` subagents (first attempt failed — all 5 hit an account-wide session limit before
making any changes, confirmed via `git status` showing zero diff beyond the pre-existing diagram
pass; relaunched fresh after the reset, same prompts, all 5 succeeded). Each agent was briefed with
Week 1's README as a worked example plus repo-specific hints on which figures were already known to
be paper-sourced vs. course-slide-sourced from the earlier diagram-screenshot pass.

**1. Cut AI-generated tells, but learn to distinguish that from cutting technical reasoning.** The
user's first instinct read Week 1's rewritten README as "too AI" — rhetorical-question openers,
templated transitions, "exactly" as a verbal tic (9 uses), heavy em-dash overuse (64 in 613
lines). A first trim pass over-corrected and cut some real ML reasoning along with the actual
filler (the late-fusion/early-fusion tradeoff paragraph, a line about validating embeddings before
training the predictor) — caught only because the user pushed back explicitly: *"there's a fine
line between not explaining stuff and just over-explaining everything."* Restored both. The
working distinction, now in the "Tone" section above: cut self-referential commentary and
redundant restatement; never cut the actual *why* behind a technique or a real tradeoff, since
that reasoning is the whole point of a portfolio meant to prove ML understanding to a technical
hiring audience.

**2. Add real rigor, not just adjust prose, when a reader would ask "compared to what?"** Prompted
by "imagine you work at DeepMind/Anthropic/OpenAI and you're reading this" — the answer to what
would build trust wasn't more polished writing, it was real numbers a rigorous reader looks for
first: scale (vocab size, corpus size, and critically, that the HN regression stage only has 782
labeled examples — never stated anywhere before), and a baseline. Went further than a prose fix:
`src/prediction/train.py` had no fixed random seed, so the README's loss numbers were never
actually reproducible (already flagged as a known gap in Week 1's own `TODO.md`, from an earlier
pass, sitting unaddressed). Fixed it for real — added `torch.manual_seed(42)` plus a real
train-mean baseline computed on the same seeded split, re-ran training, and reported what actually
came out: the model doesn't clearly beat the trivial baseline on raw MSE, and shows a textbook
overfitting signature past epoch 5. That's a more honest, more interesting result than the old
unreproducible number, and it's the kind of finding this audience trusts *more*, not less. General
lesson: when a README's Results section reports a number that isn't actually reproducible, that's
a code bug worth fixing (seed it, add a baseline), not just a caveat to add in prose.

**3. Drop course-slide/team-deck crediting entirely; keep paper citations.** User clarification
mid-pass: the course's own terms allow reusing slide/deck material freely, without needing
attribution — the earlier "used with permission" framing (established back in Pass 1, see
Acknowledgments guidance) had been overclaiming a permission requirement that never existed. Fixed
per repo, starting with Week 1: stripped every "from the course's assignment brief" / "from the
same workshop deck" caption phrase and the Acknowledgments paragraph crediting course curriculum,
while keeping paper citations (Figures 8→10 and 11→13 after the section-reorder renumbering below
still cite Mikolov et al.'s own Figure 1/Figure 2 directly) — those still need real citation,
course material doesn't. Full reasoning now lives in "Diagrams" and "Source material" above; this
is a real asymmetry, not an oversight — keep it that way in Weeks 2–6, don't over-correct and
strip the paper citations too.

**4. Move System Architecture ahead of Method.** The original academic-paper order (Method, then
Architecture) meant a reader had to get through the entire technical deep-dive — feature
representation, the distributional hypothesis, CBOW/negative-sampling math, evaluation methodology
— before ever seeing the simple "here's the 4-stage pipeline" orientation. Reordered so Architecture
(the big picture) comes right after Introduction, Method (the deep-dive) follows once the reader
already knows the shape of the system — closer to how someone would actually walk a colleague
through the repo in person. Confirmed working end to end on Week 1: every one of the 17 figures
renumbered to match the new document order (Architecture's two diagrams became Figures 4–5, every
figure after shifted by +2 until Implementation's section, where the original numbering picked back
up), the Implementation section's opening line rewritten to point back to "Section 2" explicitly
since Architecture is no longer the immediately preceding section, and the Architecture section's
own opening line rewritten since it used to assume Method had already been covered ("Method's
covered — here's how it's wired together" → "Before the method — here's how it's wired together at
a glance"). All ToC anchors and cross-references re-verified by grep after the reorder, not assumed
correct. This is now the standard structure (see "Structure" above) — apply it to Weeks 2–6 too,
including the same figure-renumbering discipline.

**Result, Weeks 2–6** (full detail in each repo's own `TODO.md`): every repo got the same structural
reorder + figure renumbering, voice rewrite (em-dash counts roughly halved across the board — e.g.
Week 4 66→31, Week 6 65→21), and course-crediting strip (paper citations kept and renumbered where
they exist: Week 3 keeps ViT/Vaswani, Week 4 keeps CLIP, Week 5 keeps Whisper, Week 6 keeps
Stiennon/Hu; Week 2 has no paper-sourced diagrams at all, confirmed in an earlier round, so nothing
to preserve there). **The rigor pass turned up real, previously-unknown bugs in four of the five
repos** — this ended up being the most valuable part of the whole exercise, not the prose:
- **Week 2**: `simple_dual_encoder_rnn.py` had no seed and no held-out evaluation at all (only
  training-loss logging). Added both — held-out pairwise ranking accuracy comes back at 52.0%,
  barely above an untrained baseline's 50.2%. Training loss looked great; the towers are
  memorizing, not generalizing, at this training scale. Now stated plainly in Results.
- **Week 3**: found and fixed a real correctness bug, not just a reproducibility gap —
  `Seq2SeqVisionTransformer.generate()` terminated the *entire batch* the instant any one sequence
  emitted EOS, silently truncating every other still-generating sequence in every batched eval,
  including the trainer's own val/test loop. The previously-reported ~30–33% exact-match figure
  understated real model quality as a result. Fixed with a per-sequence `finished` mask; verified
  with a controlled before/after on identical data (24.00% buggy → 71.50% fixed), then a fresh
  500-example run: **68.00% exact-match** is the new headline number, against a 0.10% random-chance
  baseline.
- **Week 4**: neither training notebook set a seed — proved it mattered (adapter-init checksums and
  losses differed run to run), added `torch.manual_seed(42)` to both `notebook.ipynb` and
  `final.ipynb`, reran, byte-identical results. Also found and fixed a citation numbering bug
  (LLaVA cited but pointing at PaliGemma's reference number) and an undocumented FastAPI service
  (`src/api_service.py` existed, ran, and was never mentioned in Architecture/Implementation/Usage
  — fixed).
- **Week 5**: investigated and found nothing to fix — confirmed reproducible by actually rerunning
  the demo twice (byte-identical output) and confirming the real seeded train/val split, then added
  that verified evidence to Results instead of just asserting reproducibility.
- **Week 6**: `PPOConfig` had no seed anywhere. Adding one immediately exposed a second, real bug:
  the very first reproducible run produced `Value loss: inf` / `Policy loss: -inf`. Traced to
  `SequenceGenerator` computing the policy's own logprob from **post-top-p-filtered** logits — top-p
  had zeroed the probability of a token the model had just actually sampled, producing `-inf` in a
  log. Fixed by recomputing the policy logprob via an unwarped forward pass (matching how
  `ref_logprob` was already computed), which also fixes a real PPO-ratio inconsistency that predates
  this pass. After the fix: `Value loss: 57.19`, `Policy loss: -7.26`, byte-identical across reruns,
  13/13 tests still passing.

**Lesson for future work in this environment:** "add a random seed for reproducibility" sounds like
a cosmetic ask, but in this portfolio it was a real bug-finder four times out of five — an unseeded
training/eval loop tends to be hiding either a silently-broken evaluation (Week 3), a
memorization-not-generalization result nobody had measured (Week 2), or a numerically-unstable code
path that only surfaces once results stop varying run to run and an actual number can be inspected
(Week 6). Treat "no fixed seed" as worth investigating properly in any future repo, not just
patching mechanically.

**Commit state:** all six repos' `README.md` files are uncommitted, same as the rest of this
portfolio's work-in-progress convention — left for user review. Each repo's real code fix is also
uncommitted: Week 1 `src/prediction/train.py`; Week 2 `src/models/simple_dual_encoder_rnn.py`; Week
3 `src/models/seq2seq/seq2seq.py`; Week 4 `notebooks/notebook.ipynb` + `notebooks/final.ipynb`;
Week 6 `src/mlx_week6_rlhf_ppo/config/ppo_config.py` + `ppo_trainer/generator.py` +
`ppo_trainer/ppo_trainer.py`. Week 5 needed no code change.

## Code standardization pass (done, 2026-07-31/08-01)

Before this pass the six repos shared a Dockerfile, a Makefile and a directory shell, but nothing
underneath: six different device-selection implementations, 12 `sys.path` hacks, 15 `PYTHONPATH=`
references, no shared config or logging, and lint/type tooling in exactly one of six. The user's
framing was that it looked like "six different codebases that have just been dockerized." This pass
gives them one common internal shape.

**Scope constraint, quoted, and it governs every decision here:** this is a single-node learning
portfolio, *not* a production/HA system. "We're not trying to optimize things, we're not trying to
fix all these edge cases, all we're doing is building out the happy path and ensuring that works."
Do not add retry logic, graceful degradation, or edge-case handling that was not already there.

### The standard

**`ai-residency-common`** (`/Users/rosh/mlx/ai-residency-common`, private repo at `github.com/RoshBeed/ai-residency-common`,
consumed by **git URL pinned to a tag**) holds everything that must be identical across repos:

| Helper | Replaces |
|---|---|
| `get_device()` | six separate cuda/mps/cpu implementations |
| `get_logger(__name__)` | ~495 `print()` calls, no logging config anywhere |
| `save_checkpoint` / `load_checkpoint` | 15 `torch.save` sites in inconsistent formats |
| `BaseConfig` + `set_seed()` | per-repo ad-hoc config; four repos had no seed at all |
| `push_checkpoint` / `pull_checkpoint` | weights hosted on HF Hub instead of only on this Mac |
| `init_tracking` / `log_metrics` / `finish_tracking` | W&B, previously in only 2 of 6 repos |

Pin by tag (currently `@v0.6.0`), never a branch — six repos depend on it and an unpinned change breaks all
six at once. `load_checkpoint` defaults to `map_location="cpu"` on purpose: MPS-trained weights
carry `mps:0` metadata and fail to deserialise in the CPU-only containers. Hub and W&B both degrade
rather than fail when credentials are absent, so a fresh clone can still run `make train`.

**`ai-residency-repo-template/`** is the target shape, and it is a *working* repo rather than a skeleton —
`make train` trains, `make serve` answers, `make check` passes. A template nobody has run is a
template that does not work.

```
config/__init__.py       Config(BaseConfig) — every hyperparameter and path, one dataclass
<topic>/data.py           build_datasets(config) -> (train, val)
<topic>/models.py         model classes
<topic>/training.py       train(config), plus main()
serving/api.py           FastAPI app exposing `app`
tests/
```

(where `<topic>` is the repo's topic package — `word_embeddings/`, `multimodal_captioning/`, etc.)

**The training shape**, worth copying exactly:
`set_seed` → build loaders → `compute_baseline` → `init_tracking` → epoch loop with `log_metrics`
→ on new best `save_checkpoint` then `push_checkpoint` → `finish_tracking`. Seeding comes first
because unseeded runs in this portfolio were hiding real correctness bugs, not just producing
wobbly numbers. A baseline is computed because a loss with nothing to compare against says very
little — in Week 1 the trained model turns out not to beat predicting the mean.

**Naming spec:** `get_device()`, `train(config)`, `config` (not `cfg`), `embeddings` (not `emb`),
`learning_rate`/`batch_size`/`num_epochs` spelled out, `main()` + `if __name__ == "__main__":`,
API at `src/<pkg>/serving/api.py`, UI at `ui/app.py`.

**Packaging:** every repo is a real installable package. `pyproject.toml` needs **both**
`[tool.setuptools] package-dir = {"" = "src"}` **and** `[tool.setuptools.packages.find]
where = ["src"]` — the `find` block alone silently builds an empty wheel. Python floor `>=3.11`
with `python:3.11-slim` everywhere. Zero `sys.path` hacks, zero `PYTHONPATH=`; commands are
`python -m <package>.<module>`.

**Tooling:** ruff (format + lint) and mypy **strict, with zero suppressions** — no `type: ignore`
on our own code, no blanket `noqa`. Third-party `ignore_missing_imports`/`follow_imports` entries
in `[[tool.mypy.overrides]]` are fine and expected; those are stub decisions about libraries we do
not control, which is a different thing from silencing our own code. The `[tool.ruff]`/`[tool.mypy]`
blocks live canonically in `ai-residency-common/pyproject.toml` and are copied verbatim, so configuration
cannot drift.

**Private-dependency builds.** `ai-residency-common` is private, so `docker build` cannot inherit host git
credentials — a naive build fails at `uv sync` (verified, not assumed). Use a BuildKit secret:

```dockerfile
RUN --mount=type=secret,id=gh_token \
    git config --global credential.helper 'store --file=/tmp/gc' && \
    printf 'https://x-access-token:%s@github.com\n' "$(cat /run/secrets/gh_token)" > /tmp/gc && \
    uv sync --frozen --no-install-project && \
    rm -f /tmp/gc && git config --global --unset credential.helper
```

Built with `docker build --secret id=gh_token,env=GITHUB_TOKEN`; the Makefile fills `GITHUB_TOKEN`
from `gh auth token`, and `git` must be installed in the base stage. Leak-checked: no credential
file survives, no global helper is left set, and the token appears in no image layer.

### What the pass keeps finding

Strict typing and ruff are earning their keep as bug-finders, not just style enforcement. In Week 1
alone, with nothing else changed:
- **A ~180-entry character-replacement table was dead code.** Both ETL processors ran
  `text.encode("ascii", "ignore")` *before* applying it, deleting every character it matches.
  `'cost: EUR20 - GBP15'` was coming out as `'cost: 20  15'`, silently degrading the corpus.
  Surfaced by chasing a ruff F601 duplicate-key warning.
- **`cbow/predict.py` was broken at runtime.** It called `CBOWNS(vocab_size, embed_size)` when the
  constructor takes a single `CBOWConfig`. Confirmed live: `TypeError: takes 2 positional
  arguments but 3 were given`. mypy found it; nothing else ever had.

Treat what the tooling surfaces as a lead to investigate and confirm empirically, not as noise to
silence.

### Reusable decisions (do not rediscover these)

- `N812` (`import torch.nn.functional as F`) is waived in the **canonical** ruff block — universal
  PyTorch idiom, otherwise needs a per-file dodge in all six repos.
- `RUF001`/`RUF003` need per-file ignores wherever a module *is* a unicode→ASCII table; the rule is
  misreading intent there.
- `transformers.*` needs `follow_imports = "skip"`; its partial annotations otherwise leak
  `no-untyped-call` out of `PreTrainedModel` internals. Any module subclassing it then needs
  `disallow_subclassing_any = false`.
- Add `types-tqdm` and `types-PyYAML` dev deps; waive stubs for scipy, nltk, psycopg2, sklearn,
  matplotlib, seaborn, wandb.
- PEP 695 `type X = ...` is **not** available at the 3.11 floor — use plain aliases.
- `TensorDataset` is typed `tuple[Tensor, ...]`, not a fixed pair; `DataLoader.dataset` has no
  guaranteed `__len__`, so count examples in the loop.
- Trained artifacts must not live inside `src/` — Week 1 had 1.2GB of weights in its source tree.
  They belong in the top-level `checkpoints/`.
- **Never regex-rewrite the character-replacement tables.** They contain smart-quote characters
  that render like ASCII quotes; a naive line-pattern edit produced an unterminated triple-quote.
  Edit by line number and verify with `ast.parse` immediately.

### Running this as parallel agents

Five parallel conversion agents all died at the same account-wide session limit ~40% in, each
leaving its repo **half-migrated** — files moved, `pyproject.toml` not yet flipped, imports pointing
at paths that no longer exist. That state is strictly worse than untouched. **Run two repos at a
time**, and instruct each agent to re-verify importability after every structural step so an
interruption leaves something coherent.

Recovery when an agent dies mid-flight: check whether `pyproject.toml` has `package-dir` yet; if
the conversion is incomplete, `git reset --hard` to the repo's pre-pass commit, then remove the
leftover **untracked** `src/mlx_week*/` directory, which `reset --hard` will not touch. **Never
`git clean -fdx`** — `data/` and `checkpoints/` are gitignored and hold hundreds of MB of real
training artifacts.

### Result

All six repos converted, plus `ai-residency-common`, `ai-residency-repo-template` and the hub. Everything committed
locally; **nothing pushed except `ai-residency-common`**. Per-repo detail lives in each repo's own
`TODO.md`; the phase-by-phase account is in `/Users/rosh/mlx/HANDOFF.md`.

| | Before | After |
|---|---|---|
| Device-selection implementations | 6 | 1 |
| `sys.path` / `PYTHONPATH` hacks | 27 | 0 |
| Repos with lint + type checking | 1 of 6 | 6 of 6, strict, zero suppressions |
| `print()` in packages | ~495 | 0 |
| Repos with tests | 1 | 6 (89 tests) |
| Repos with a config layer | 2 | 6 |
| Repos hosting weights off this machine | 1 | 6 |
| Backends exposing `/health` | 1 | 6 |

Verified by building all six backends through the hub, starting them, and making a real request
against each: score prediction `12.13`, ranked retrieval over 1,498 indexed documents, digit
classification at 0.937, captioning, `"My name is Roosh."` → `"My name is Rosh"`, and generation.
Week 2's and Week 4's outputs are poor, which is the correct result — both READMEs document that
honestly, so matching them is the check passing.

### Bugs the tooling found, which is the part worth remembering

Strict typing and lint earned their place as bug-finders, not style enforcement. None of these were
found by reading:

- **Week 1** — a ~180-entry character-replacement table was dead code, because
  `encode("ascii", "ignore")` ran before it and deleted every character it matched. Surfaced by
  chasing a ruff duplicate-key warning. Separately, `CBOWNS(vocab_size, embed_size)` raised
  `TypeError` on every call; the constructor takes a config object. mypy caught it.
- **Week 2** — four separate broken consumers of trained artifacts, including a module that raised
  `NameError` unconditionally and one that opened a Redis connection at import time against a
  placeholder host.
- **Week 4** — the checkpoint/adapter mismatch was never about the extra `LayerNorm`: nothing
  recorded which architecture produced which checkpoint, so the loader guessed and was wrong for
  one file in three. Also, MPS training is not reproducible and occasionally wrong — across 10
  seeded runs, 8 matched, 1 diverged, and **1 reported `0.0000` loss from epoch 2 and saved it as
  best**. Training now defaults to CPU; inference still uses `get_device()`.
- **Week 5** — **train/val leakage**. Re-running the split layered a new assignment over the old
  one, leaving 170 files in both splits (705 files from a 535-file corpus).
- **Week 6** — `requires-python = ">=3.10"` contradicting a `python:3.13-slim` base.
- **The hub** — no `secrets:` block at all, so every backend build would have failed on the private
  dependency; a build target naming a stage that no longer existed; a stale `PYTHONPATH`.

A second, subtler class showed up in the documentation: **two READMEs described their repo as more
broken than it was**, listing already-fixed bugs as open. Nobody audits docs for being unfairly
harsh, but on a portfolio meant to demonstrate competence it costs credibility just as much as
overclaiming. Check "known issues" lists in both directions.

### Still open

- Nothing has been pushed except `ai-residency-common`. The six week repos' older history is authored with
  `rosh.beed@icloud.com`, which GitHub's email-privacy setting rejects — the global config change
  only affects new commits, so pushing them needs that resolved first.
- `ai-residency-common` is private at the user's request; they intend to flip it public themselves.
- ~~The hub still mounts checkpoints from sibling repos on this host.~~ **Closed** by the three-host
  verification pass below: serving now falls back to `pull_checkpoint`, and every repo's artifacts
  are on the Hub. The `hostPath` mounts remain as the fast local path, not as the only one.

## Three-host verification (done, 2026-08-01/02)

The user added `HF_TOKEN` and `WANDB_API_KEY` to `/Users/rosh/mlx/.env` and asked to verify the
whole portfolio end to end on **macOS, docker-compose, and Kubernetes** before renting a RunPod GPU
box for training and a cheap Hetzner VM for inference — deliberately finding the problems on
hardware that costs nothing.

It turned into a build rather than a test run, because two things the cloud split depends on did not
exist: **serving always assumed its checkpoint was already on local disk** (true only on the machine
that trained it), and **nothing read a `.env`**, so the tokens reached nothing.

### What changed

- **`ai-residency-common` v0.2.0 → v0.5.1** during this pass (v0.6.0 came later, with dataset-repo
  support), each release forced by a real failure:
  `load_env()` (layers every `.env` from cwd upwards, nearest first), `MLX_<FIELD>` overrides on
  every `BaseConfig` field, `resolve_checkpoint()` (local file, else Hub, reporting which), and
  `save_checkpoint(..., state_dict=)` for adapters on frozen backbones.
- **Serving pulls from the Hub** in the Template and Weeks 1–4, and every backend reports
  `checkpoint_loaded` / `checkpoint_source` from `/health`. Not Week 5 (its API deliberately loads
  a fresh Whisper per request) and not Week 6 (nothing meaningfully trained to serve).
- **Weeks 5 and 6 brought to parity** — both now push and track; Week 6 gained the `save_checkpoint`
  it never had.
- **The hub can train**, on both runtimes, without new images: profile-gated compose services and
  opt-in Helm `Job`s that reuse the `api`-stage image with `command:` overridden. Every Dockerfile
  here is one base layer plus stages differing only in `CMD`, so that image already has everything.
- **Helm gained a `Secret`** filled at install time from the shell, so no committed file holds a
  token.

### Verified by running it, not by reading

Train → checkpoint → **HF push** → **online W&B run** on all six plus the template. Then the actual
point of the exercise, the *Hetzner rehearsal*: move the local checkpoint aside and confirm the
service pulls from the Hub and returns identical output. Proven four ways — Template (SHA256-
identical file, same prediction), Week 3 (both models, `source: hub`), Week 1 (805MB CBOW **and**
predictor, `10.38` either way), and a container with no checkpoint mount at all.

Kubernetes: all 8 pods on freshly built images, a training `Job` completing and pushing, and
**88.78% / 89.06% identical between the compose run and the k8s run** — same seed, same CPU, two
container runtimes.

### The bugs, which are the part worth remembering

1. **Every HF push would have 403'd on the cloud box.** All six repos said
   `hf_repo_id = "RoshBeed/..."`; the Hub creates repos only under the token's exact canonical
   namespace, `roshbeed`. Lookups of *existing* repos are case-insensitive, so a first probe
   reported success and was wrong — the real test is creating a name that does not exist yet.
2. **A rejected push killed the finished run.** `push_checkpoint` raised on HTTP errors while
   documenting itself as degrading. On a rented GPU that throws away a completed run at the last
   step. It now logs and returns `False`; the checkpoint is already on disk by then.
3. **A LoRA checkpoint was 2.4GB.** `model.state_dict()` on a PEFT-wrapped Qwen3 includes the frozen
   base, re-uploaded every run. With an explicit `state_dict` it is **18.4MB** and the losses are
   unchanged. Log the file size on save — that is the only reason this was visible.
4. **A partial `.env` silently shadowed the shared one.** Week 2 is the only repo with its own
   `.env` (Redis vars only), and stopping at the first file found meant it trained with no token
   while the other five pushed. Load every `.env` up the tree, nearest first.
5. **Three repos would have committed a live-token `.env`**; two others ignored `.env.example` via
   an `.env.*` pattern, so the template could never be committed. Check both directions.
6. **The hub's Makefile did not parse at all** — a `GITHUB_TOKEN` block inserted mid-`.PHONY`
   continuation. Every hub `make` target had been dead since that commit.
7. **Week 1's compose `env_file` was inert** — the file used shell `export KEY=value`, which compose
   reads literally, producing a variable named `export WANDB_API_KEY`.
8. **Week 6 was `OOMKilled` on Kubernetes** at 3072Mi. A float32 Qwen3-0.6B is ~2.4GB of weights
   before activations. The `values.yaml` memory numbers were guesses; these two are now measured.
9. **Serving artifacts were missing from the Hub**, so a bare host could not have started at all:
   Week 1's 805MB CBOW weights, Week 2's vocabulary (under `data/`, not `checkpoints/`), Week 4's
   committed adapters.
10. **Week 5 carried a phantom submodule gitlink** — `datasets/crema_d`, mode 160000, no
    `.gitmodules`, pointing at a commit the repo does not contain, and empty on disk. Nothing here
    runs `git submodule update`, so a clone on a rented box would have been the first thing to trip.
11. **Week 4 answered 404 on `/`**, alone among the six, so `make status` reported it down while it
    was serving fine.

### Things to know before working on this again

- **`AI_RESIDENCY_<FIELD_NAME>` overrides any config field** — `AI_RESIDENCY_NUM_EPOCHS=1 make train` for a smoke run,
  `AI_RESIDENCY_NUM_EPOCHS=50` on a real GPU. It walks the dataclass fields, so repo-specific ones work too.
  **Week 3's classifier is the exception**: it uses `classifier_epochs`, so it needs
  `AI_RESIDENCY_CLASSIFIER_EPOCHS`, and `AI_RESIDENCY_NUM_EPOCHS` alone will not shorten it.
- **Training rewrites tracked files.** Week 3's smoke run overwrote a committed 95%-accuracy
  checkpoint with its 88.86% one and rewrote `training_config.json`; Week 4's default
  `checkpoint_name` is a committed adapter. Back up, or point `AI_RESIDENCY_CHECKPOINT_NAME` elsewhere.
- **The hub can only ever build committed code** — submodules pin commits, so any repo change must
  be committed and the submodule re-pinned (with a `git fetch` inside the submodule first) before
  the hub can test it.
- **Compose and Helm must not both run.** Scale the release to zero first.
- Node allocatable is **23.3GiB / 10 CPU** after the user's Docker Desktop bump; the "~7.55GiB
  ceiling" comment in `values.yaml` predates that.

### Still open after this pass

Everything here was closed later the same day; kept for the trail rather than as live work.

- ~~Week 5 tracks ~145MB of EmoDB audio.~~ **Wrong and resolved** — no `.wav` was ever tracked. See
  "No data in the repos" at the end of this file.
- ~~The frontend on `:8500` was never confirmed answering HTTP.~~ **Confirmed** — `HTTP 200` over
  the Kubernetes LoadBalancer, and Streamlit's own `/_stcore/health` returns `ok`. The earlier `000`
  under compose was a startup race inside a 20s wait, not a fault.
- Nothing pushed except `ai-residency-common`; the email-privacy blocker above still applies. **This is the
  one that is genuinely still open.**

## Pass 3 — architecture pass, Week 1 (done, 2026-08-02)

The last numbered item in the plan, and the only one that was real architectural change rather than
documentation, structure, tooling or verification. Week 1 was chosen as the flagship — the most
likely thing a reviewer opens first.

**Pass 2b was skipped, deliberately, at the user's direction.** Its checklist had largely
evaporated: `requirements.txt` completeness is superseded by uv and committed locks, dead code was
already down to seven lines across six repos, and GitHub descriptions / topics / a demo link are all
blocked on the repos not being pushed. What remained actionable was a LICENSE (added — **MIT**, all
nine repos) and CI (not done; note that a workflow needs either a PAT secret per repo or
`ai-residency-common` made public, since it is a private git dependency).

### The diagnosis

Week 1 had standardised its outside but not its middle. It was the first repo converted, before the
canonical training shape had settled in `ai-residency-repo-template`, so it was the one repo never
retrofitted — and every specific problem followed from that one fact:

- **Two config systems.** `Config(BaseConfig)` was the standard, but the CBOW stage read
  `cbow_ns.yml` (SHOUTY_CASE keys) back out of **`wandb.config`**. The stage therefore could not
  start unless a tracking run had been opened first: a tracking tool was load-bearing for
  configuration. It also meant the `MLX_*` overrides reached the predictor stage and silently did
  nothing for CBOW.
- **A 165-line `train_run()`** doing config, vocabulary, data, the epoch loop, evaluation,
  checkpointing, promotion and the sweep.
- **Bespoke everything downstream of that**: four separate output files, a best-loss text file,
  promotion by `shutil.copy`, and a `push_to_hub` that created **a different HF repo per sweep run**
  — which is where the stray `roshbeed/cbow-model-<run name>` repos came from.
- **Dead code presented as real.** `skipgram/` had model, train and predict; nothing imported it,
  yet the README described it as half the system.
- **Stage 1 could not run at all.** `db_processor.py` hardcoded a Postgres mirror, with no timeout,
  that is unreachable from at least one network here — so `make train` had never completed.

### What it looks like now

`config.py` holds every hyperparameter for all three stages. The CBOW stage is `data.py` (vocabulary
and corpus), `evaluation.py` (returns metrics; no `wandb.log` inside model code) and a
`train(config)` following the canonical shape. Checkpointing is `save_checkpoint` / `push_checkpoint`
like everywhere else, with the vocabulary stored *inside* the checkpoint — embeddings indexed by a
mapping in a separate file are one stale file away from being silently wrong. Sweeps are W&B's, via
`sweep.yaml`, driven through the same `MLX_*` variables so there is no sweep-specific plumbing in the
trainer.

### Worth remembering

- **The 800MB CBOW checkpoint was migrated in place**, old bespoke layout to `save_checkpoint`'s,
  weights verified byte-identical and all 258,196 vocabulary words preserved. The copy on the Hub was
  replaced too — anything still expecting the old top-level `word_to_lemma_index` key will not load
  it. `serving/api.py` and `visualise_embeddings.py` both read `extra` now.
- **`evaluate_embeddings` skips checks the vocabulary cannot support** rather than scoring 0.0. The
  analogy check needs "king"/"queen"/"man"/"woman", which a real corpus is not obliged to contain,
  and a silent zero there reads as a training failure.
- **`make train-cbow-sample`** trains on a tiny built-in corpus in seconds, to its own checkpoint
  path, and never pushes. The old `--dummy` flag guarded only the promotion step, by which point the
  real file had already been overwritten.
- Verified by running it: 18 tests (was 10), ruff and strict mypy clean, a real sample training run,
  the ETL fallback exercised against the genuinely unreachable host, and **the same `10.38`
  prediction as before the change, from both the host and the container**. A pure architecture change
  should be invisible from outside.

### Still open across the portfolio

- **Nothing pushed except `ai-residency-common`.** The six week repos' older history is authored with
  `rosh.beed@icloud.com`, which GitHub's email-privacy setting rejects; pushing needs a history
  rewrite first. This also blocks CI, repo descriptions, topics and any demo link.
- ~~Week 5 tracks EmoDB audio.~~ **Resolved 2026-08-02** — see "Do not commit downloadable data"
  below. The earlier note here was also wrong on the facts: no `.wav` was ever tracked.
- No CI in any repo (see the Pass 2b note above for the private-dependency wrinkle).


## No data in the repos (done, 2026-08-02)

Standing instruction from the user, in two steps: first "don't commit data that can be downloaded",
then, after seeing the result, **"these repos shouldn't have any data committed to them"**. The
second is the rule now. All nine repos track **zero** files under `data/` or `datasets/`.

### What the audit found, including a correction to this file

Week 5 was recorded above as tracking "~145MB of EmoDB audio". **It never did** — the audio was
always gitignored. What was tracked were 1,572 EmoDB *annotation* files (`.lablaut`, `.silb`,
`.labsilb`) totalling 1.1MB. The wrong number came from running `du` on the working tree instead of
`git ls-files`, which measures a completely different thing. Worth remembering as a method point:
**to find out what a repo commits, ask git, not the filesystem.**

The real offender was **Week 1 at 216MB**, which had never been flagged at all.

### Untracked, with `git rm --cached` (never plain `rm` — the working copies stay)

| Repo | Removed | Why safe |
|---|---|---|
| Week 1 | 215MB, 10 files | `text8` downloads; the rest is `make train-corpus` output |
| Week 5 | 1.1MB, 1,572 files | ships in the `EMODB_URL` zip `make download-data` already fetches |
| Week 5 | 13 `wandb/` run files | build output. This was the only repo missing a `wandb/` ignore rule, and they were committed by an over-broad `git add -A` earlier the same session |
| Week 4 | 7.3MB, 10 images | see below — argued for keeping these, and was wrong |

### Hosted on the Hub instead, in dataset repos

Three things were **not** downloadable and existed only on this Mac. Committing them was the only
reason they were reachable at all — which is an argument for hosting them, not for committing them:

| Dataset repo | Holds | What it replaced |
|---|---|---|
| `roshbeed/ai-residency-week1-hn-posts` | the Hacker News extract | the last committed file in Week 1; it came from a Postgres mirror that is not reachable from every network |
| `roshbeed/ai-residency-week2-retrieval-data` | tokenizer vocabulary, CBOW corpus, tokenized triples | **the portfolio hub's `data/` bind-mount** — the last host-specific mount in the stack |
| `roshbeed/ai-residency-week4-sample-images` | 10 demo images | fixtures for the zero-shot demo and the UI's sample picker |

`ai-residency-common` **v0.6.0** made this possible: `repo_type` on `push_checkpoint` / `pull_checkpoint` /
`resolve_checkpoint` (default `"model"`, so nothing existing changed), plus `hf_dataset_repo_id` on
`BaseConfig`. Every reader resolves data exactly as serving already resolves weights — local file if
present, Hub otherwise.

### The line on what gets hosted

**Only our own extracts and derived artifacts.** `text8`, MS MARCO, MNIST and EmoDB all download
from their own homes, and re-hosting someone else's corpus under this account is a licensing
question rather than a technical one. Also skipped: Week 2's `triples_small.json` (107MB), an
intermediate `make train-triples` regenerates from MS MARCO — hosting a copy would only be another
artifact to keep in step with its source.

### Worth remembering

- **Untracking Week 1's data only became defensible because Pass 3 made the corpus stage runnable.**
  Before that its outputs were irreproducible in practice, whatever their provenance said. Check
  that data really is reproducible *on this machine* before removing it, not just in principle.
- **A test started passing for the wrong reason.** Week 2 has a test asserting that a missing
  vocabulary raises an error naming the command that builds it. Once the loader could reach the Hub,
  the file was no longer missing and the test passed without exercising anything. The fixture already
  set `hf_repo_id=None` to keep tests offline; `hf_dataset_repo_id` needed the same. **Any new
  Hub-reaching field has to be disabled in the test fixtures too**, or the suite quietly becomes
  network-dependent.
- **Week 4's demo images were kept at first and should not have been.** The argument was that
  fixtures are not a dataset. They are still data, and the rule is simpler without the exception.
- Verified the same way in every case: hide the local copy, confirm the file returns
  **byte-identical** by SHA256 and the code still works. Week 1 served the same `10.38` from a
  container with no `data/` directory at all; Week 2's vocabulary came back with all 7,692 words;
  all ten Week 4 images matched.

### The caveat that matters

**None of this shrinks the repos.** The blobs are still in history — Week 1's `.git` is ~256MB on a
fresh clone. Only `git filter-repo` removes them, and that is worth doing in the same pass as the
`rosh.beed@icloud.com` author-email rewrite the six repos already need before they can be pushed:
same operation, same history, far better done once.

## Container-source / documentation-repo split (done, 2026-08-06)

The user's own framing, since the end goal is Kubernetes: separate duties more cleanly. Strip the
six week repos down to just what a container build needs — no UI, no `assets/`, no long README —
and move all of that into the Hub, which becomes the entry point. Each week gets **two** Dockerfiles
now instead of one multi-stage file, because inference and training have genuinely different
targets: Kubernetes runs inference, a rented single GPU host (RunPod, or now AWS) runs training. And
`ai-residency-repo-template` stops being just a copy-paste pattern and becomes the literal base image every
other Dockerfile builds `FROM`.

Confirmed via three clarifying questions before starting: (1) the Template becomes a **real
published Docker base image** (`roshbeed/ai-residency-base`), not just a reference layout; (2) existing README
content moves into the Hub **1:1**, not trimmed; (3) plan the full six-repo rollout up front rather
than piloting on Week 1 first, the one time in this portfolio's history that pattern was skipped.

### Phase 1 — `roshbeed/ai-residency-base`

Built from a new `Dockerfile.base` in `ai-residency-repo-template`: `python:3.11-slim` + `uv` + `ai-residency-common`
installed once, using the BuildKit-secret dance every repo's Dockerfile used to repeat. **Confirmed
empirically, the load-bearing fact the rest of this pass depends on:** a downstream repo's own
`uv sync --frozen`, run against its own completely different `pyproject.toml`/`uv.lock`, needs **zero**
git credentials once `ai-residency-common` is already installed in the base image — `uv` recognizes the same
pinned tag/commit as already-satisfied and never re-fetches it. Every week's `Dockerfile.train`/
`Dockerfile.serve` therefore builds `FROM roshbeed/ai-residency-base` with no secret mount of its own; only
`Dockerfile.base` ever touches `GITHUB_TOKEN`, in one repo instead of six. Hit the same CUDA-bloat
bug documented earlier in this file on the very first build (ai-residency-common's own unpinned `torch>=2.0`
resolving the CUDA wheel) — fixed by installing the CPU wheel explicitly before ai-residency-common, same
family of fix as always, just newly relevant because this was the first time a *bare* `uv pip
install` (no project `pyproject.toml`, no `[tool.uv.sources]` override to inherit) needed it.

### Phase 2 — the pattern, applied to Template + all six weeks

Two Dockerfiles per repo: `Dockerfile.serve` (built from what used to be the `api` stage — the
FastAPI inference container Kubernetes runs) and `Dockerfile.train` (RunPod/AWS). The old `ui` stage
is gone outright; the Hub's own `frontend/app.py` was already the one real UI. Naming ambiguity
resolved per repo: Week 2's old `serve` stage (an interactive query CLI, not a real server) becomes
a local-only `make query` target; Week 6's old `serve` stage (ran `pytest`) becomes just `make test`,
with `Dockerfile.serve` built from what used to be its `api` stage instead. `docker-compose.yml`
trimmed to `serve` (default) + `train` (profile-gated); Makefiles gained `docker-push-train`/
`docker-push-serve` targets. Each week's long README moved 1:1 into the Hub as `docs/weekN.md`
(images into `docs/assets/weekN/`), replaced by a ~100–150 line technical README pointing back at it.

Run as three batches of two parallel agents (Weeks 1–2, 3–4, 5–6) — the same two-at-a-time discipline
this portfolio settled on after earlier passes found that more concurrent agents produces
half-migrated, worse-than-untouched repos under an account-wide session limit. **This time the limit
was a *weekly* one, not the per-session kind documented earlier in this file** — both Week 3's and
Week 4's agents died mid-verification with "hit your weekly limit, resets Aug 6 3pm Europe/London," a
four-day wait. Their actual file edits were already complete and correct at the point of failure
(confirmed directly: `git status`/`git diff`, then real Docker builds and served requests) — only the
verification step was interrupted, so the fix was finishing that verification directly rather than
re-running the agents once the reset passed. Worth remembering: a "your weekly limit" message is a
different, much longer wait than the session limits this portfolio has hit before — check the actual
reported reset time before assuming a short retry will work.

### Phase 3/4 — Hub wrap-up

Hub gained a "Documentation" section linking all six `docs/weekN.md`, and its own "Related" section
(which had been claiming each week repo still had its own full README) got fixed. `docker-compose.yml`
and the Helm chart's `values.yaml`/`train-jobs.yaml` updated for the new image shape — each
`train-weekN` service/Job now builds and runs its own real `-train` image instead of reusing the
serving image with `command:` overridden, since `Dockerfile.train` and `Dockerfile.serve` are
genuinely distinct files now. Verified structurally (`docker compose config`, `helm template` —
confirmed the rendered train Job actually references `ai-residency-hub-week3-train:latest`), **not**
with a live `docker compose up` through the Hub: its git submodules are still pinned to
pre-restructuring commits (the Hub only ever builds committed code, a pre-existing constraint), and
committing nine repos wasn't something to do unattended without being asked.

### Phase 5 — Week 3's AWS Terraform, generalized

`ai-residency-week3-vision-transformers/infra/` moved to `ai-residency-hub/infra/aws-training/` and stopped being
Week-3-specific: instead of cloning a git repo and hand-building a Python environment on the instance,
`user_data.sh` now installs Docker + the NVIDIA Container Toolkit and `docker run`s whichever week's
already-pushed `roshbeed/ai-residency-weekN-train` image `train_image` names — the same container Phase 2
produces, just on a rented GPU instead of this Mac.

**Found while moving it — this Terraform module could never actually have been applied, three
separate ways:** `user_data.sh` referenced three brace-style bash variables
(`training_type`, `max_training_time`, and one used only inside a bash function) that were never
passed to the `templatefile()` call in `main.tf` — `templatefile()` fails outright on any `${...}`
reference it can't resolve. `main.tf` and `outputs.tf` also both separately defined the same four
outputs, which Terraform rejects as a duplicate definition. And an output computing the maximum
total cost called a function, `parsefloat`, that doesn't exist in Terraform (the real one is
`tonumber()`) — silently wrong rather than a hard error, since the `$${...}` syntax around it was
Terraform's own escape for a *literal* `${`, so the expression was never actually evaluated, just
printed back out as text. All three fixed; confirmed with a real `terraform init` + `terraform
validate` + `terraform plan` (placeholder `train_image`/`hf_token`) — plan got as far as computing
`max_total_cost` correctly (`"$12.00 (0.50/hr * 24h)"`) before stopping at the expected point, no AWS
credentials configured in this environment. `terraform apply` was never run — real, billable AWS
spend needs the user's explicit go-ahead, not a plan-approval blanket authorization.

### Verified by running it, not by reading

Every week's `Dockerfile.train` and `Dockerfile.serve` built and the serving container answered a
real request: Week 1 `/predict` (`3.8`, `2.48`), Week 2 real ranked retrieval over the indexed
corpus, Week 3 both `/predict/vit` and a full training run started cleanly, Week 4 `/caption` (a
poor-quality caption, matching its own documented undertrained-adapter limitation, not a bug), Week
5 `/transcribe` — **byte-identical to the historically documented result** (`"My name is Roosh."` →
`"My name is Rosh"`), Week 6 `/generate` with its honest no-trained-checkpoint disclosure intact.
`ai-residency-repo-template` itself: real training run beating its baseline (`5.2824` vs. `12.9774`, matching
its own README), real served `/health`.

### Still open, deliberately

Two things gated behind the user's explicit decision, not oversights:
1. A live `docker compose up` *through the Hub* (as opposed to each week verified standalone, which
   is done) needs all nine repos committed and the Hub's submodules re-pinned first.
2. Nothing was pushed to Docker Hub (`roshbeed/ai-residency-base` or any week's `-train`/`-serve` image), and
   `terraform apply` was never run.

**Commit state.** All eight touched repos (everything except `ai-residency-common`) are left uncommitted for
review, per this portfolio's established convention. Full plan: `/Users/rosh/.claude/plans/
resilient-stargazing-stream.md`. `HANDOFF.md` tracked this pass live and now reads COMPLETE.

## Rename pass: MLX → ai-residency (done, 2026-08-17)

The user wanted the "MLX" naming gone: it collides with Apple's MLX framework and, more importantly,
it isn't even the programme's real name — the [Machine Learning Institute](https://ml.institute/)
site brands it "**AI Residency — Advanced training in deep learning**", and "MLX" only ever appears
as a colloquialism in reviewer quotes ("I joined MLX", "MLX7 was..."). The whole portfolio was
renamed to `ai-residency-*`.

The name was settled after some back-and-forth. Early options (`mli`, `machine-learning-institute`,
`air`, `ml-residency`, bare descriptive topic names) were rejected; the user's own facts tipped it —
it is a real, selective, UK-government-funded residency, so `ai-residency` is accurate, not an
overclaim. Keeping the week number was a deliberate choice: several weeks contain multiple
sub-projects (Week 1 is word embeddings + HN score prediction; Week 3 has two model families), and
dropping the number would either bury those or invite splitting one week into fake "double the
projects" repos. Future splits/additions keep the number (`ai-residency-week1-content-scoring`
was the first such split, 2026-08-20; an object-detection repo from unfinished curriculum work is
next) rather than inventing `weekN-x`/`weekN+1` schemes.

**Naming map** (repo dir, GitHub remote, pyproject name, Docker images, README cross-links all
updated):

| Old | New |
|---|---|
| `MLX-Week1-Word-Embeddings` | `ai-residency-week1-word-embeddings` |
| `MLX-Week2-Dual-Encoder-Document-Retrieval` | `ai-residency-week2-vector-search-retrieval` |
| `MLX-Week3-Transformers` | `ai-residency-week3-vision-transformers` |
| `MLX-Week4-Vision-Transformer-Image-Captioning` | `ai-residency-week4-multimodal-captioning` |
| `MLX-Week5-Whisper-Fine-Tuning` | `ai-residency-week5-audio-fine-tuning` |
| `MLX-Week6-RLHF-PPO` | `ai-residency-week6-rlhf-ppo` |
| `mlx-common` | `ai-residency-common` |
| `MLX-Portfolio-Frontend` | `ai-residency-frontend` |
| `MLX-Portfolio-Infrastructure` | `ai-residency-infrastructure` |
| `MLX-Repo-Template` | `ai-residency-repo-template` |

Identifiers: `mlx_common` → `ai_residency_common` (71 import sites), `MLX_` → `AI_RESIDENCY_`
env prefix (in `ai-residency-common`'s `config.py`/`env.py` and everywhere it's documented), Docker
`roshbeed/mlx-*` → `roshbeed/ai-residency-*` (base, `-weekN-serve`/`-train`, frontend, template),
Helm chart dir + name `mlx-portfolio-hub` → `ai-residency-hub`, Terraform AWS resource names
(`mlx-training-*` → `ai-residency-training-*`), and the week-2/3/4/5 *topic* slugs changed too
(`dual-encoder-document-retrieval` → `vector-search-retrieval`, `transformers` → `vision-transformers`,
`vision-transformer-image-captioning` → `multimodal-captioning`, `whisper-fine-tuning` →
`audio-fine-tuning`) to read as the in-demand skills FDE/SA hiring managers screen for.

Deliberately **not** renamed: `ardMLX/text8` (a third-party HF dataset namespace), and the Apple
framework references in `ai-residency-frontend/docs/rlhf-ppo.md` (they document a real historical bug
— the repo's `pyproject.toml` used to list Apple's `mlx`/`mlx.nn`/`mlx.optimizers` as deps).

Mechanics: a Python script did the ordered replacement (longest string first) across 10 repos, 168
files. One trap worth remembering: a placeholder like `«APPLES_MLX»` still contains the substring
`MLX`, so a subsequent blanket `MLX` → `AI Residency` replacement corrupts it — protect with a
placeholder that contains no target substring, or do protected text last. The six week repos'
`origin` remotes were repointed to `RoshBeed/ai-residency-weekN-*` (nothing pushed, so no GitHub
rename redirect involved). `MLX-teams.txt` → `ai-residency-teams.txt`.

**Still open, deliberately:** the HF Hub model/dataset repos (`roshbeed/mlx-weekN-*`,
`mlx-week1-hn-posts`, `mlx-week2-retrieval-data`, `mlx-week4-sample-images`) still carry the old
ids — HF has no rename, so the user chose create-new + re-upload + delete-old, to be done as a
separate step (Week 1's CBOW is ~805MB). And the pre-existing `ai-residency-common` v0.7.2
push + re-lock blocker is untouched underneath this pass. Nothing committed or pushed.
