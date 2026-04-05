# Notebook Design Skill

Blueprint for building interactive lecture notebooks. Derived from the
01_foundations notebook and iterated with the instructor.

---

## 1. Overall Philosophy

- **Single notebook per topic**, not separate student/solution versions.
  A `_solutions` variant exists only as instructor reference.
- Students implement from-scratch; guidance comes from opening questions,
  collapsible thoughts, and auto-checker feedback. No explicit code hints
  in the notebook itself.
- The dataset is introduced **first** and used throughout. Abstract toy
  examples (A=(3,1), B=(6,8)) are avoided. Every exercise operates on
  real data so students see consequences immediately.
- Notebooks are **self-contained**: each one loads its own data and
  provides any functions it inherits from earlier notebooks as reference
  implementations.

---

## 2. Notebook Structure

### Title cell (markdown)

```
# Chapter N: Title

This notebook accompanies **Chapter N** of the lecture notes.

**Agenda**

☕ · 📏 · 🚕 · ...

**Next steps (take it from here):** 🌍 · 🎯 · ...

> **Tip:** Run cells top to bottom. Later cells depend on earlier ones.
```

- Agenda is a **single line of emojis separated by ` · `**. No table, no
  labels. Each emoji is used exactly once and maps to a `### EMOJI Title`
  section later in the notebook.
- "Next steps" emojis correspond to optional extension exercises at the
  end.

### Imports cell (code)

- First code cell. Contains all imports and shared helpers (e.g.
  `tufte_axis`).
- Checker functions live in a separate `checks.py` file, imported here.
- No output from this cell.

### Motivational intro (markdown)

- Starts with `## EMOJI Section Title`.
- Short narrative framing (2-3 sentences) that gives students a role or
  scenario. Example: "Imagine you work in a coffee quality lab..."
- Ends with a `>` blockquote opening question and a collapsible Thought.

### Data loading (code)

- Load CSV, print the table, extract features/labels/colors.
- Separate cells for loading vs. extracting so students can inspect the
  raw DataFrame before it becomes arrays.

### Scatter plot (code)

- First visualization. Always follows data loading.
- Uses Tufte style (see section 5).

### Exercise sections

Repeat this pattern for each exercise:

1. **Intro cell** (markdown) with `### EMOJI Title`
2. **Exercise cell** (code) with scaffolded function + checker call

### Recap cell (markdown)

- `### 🏁 Recap`
- Bullet list of what was done, using the section emojis.
- 2-3 key takeaways.
- Pointer to the next notebook if applicable.

### Optional extensions

- Section header: `## Take It from Here — Next Steps`
- Disclaimer: "optional ... work through them at your own pace after the
  session."
- Same exercise pattern as above but with more provided code.

---

## 3. Exercise Cell Pattern

### Intro cell (markdown)

```markdown
### EMOJI Title

> Opening question or observation (1-3 sentences, in a blockquote).
  Connects to a specific lecture concept and grounds it in the data.

<details><summary>Thought</summary>

Answer (3-5 sentences). Concrete, references the data.
</details>

Brief task description in plain language. No LaTeX math.
Tell students to "revisit the lecture notes for the formula."

Useful operations: `np.func1()`, `np.func2()`.
```

**Opening question rules:**
- Tied to the specific exercise and the dataset, not generic.
- Poses a genuine conceptual tension (e.g. "cosine says they're close,
  Manhattan says far — who's right?").
- NOT a leading question that gives away the implementation.
- NOT childish or trivial. Should require thinking about the lecture
  material.
- Never use `--` or `---` anywhere. Use `—` (em dash) or rephrase.

**Thought block rules:**
- Always wrapped in `<details><summary>Thought</summary>...</details>`.
- Answers the opening question concretely, referencing actual data values.
- Not a code hint. It's a conceptual answer.

**Task description rules:**
- Plain language, no LaTeX / math notation.
- Point students to lecture notes for formulas.
- List useful NumPy operations at the end.

### Code cell

```python
def function_name(params):
    """One-line docstring."""
    # YOUR CODE HERE
    pass


check_function(function_name, args)
```

- Function stub with docstring and `# YOUR CODE HERE` marker.
- `pass` as placeholder (returns `None`).
- Checker call immediately after, separated by two blank lines.
- No hints, no commented-out solution fragments.

---

## 4. Auto-Checker System (`checks.py`)

Each exercise gets a dedicated checker function in `checks.py`.

### Design principles

- Checkers compute the expected value **internally** from the inputs.
  They never hardcode expected values.
- This means they work with any data, any dimensionality.
- They are **generic**: `check_euclidean(fn, x, y)` works whether x, y
  are 2D or 3D.

### Three-state output

| State | Symbol | Meaning |
|---|---|---|
| Not attempted | ⬜ | Function returned `None` (student hasn't started) |
| Correct | ✅ | Output matches expected |
| Wrong | ❌ | Output is wrong; targeted hint follows |

### Checker structure

```python
def check_something(fn, *args):
    got = fn(*args)
    expected = <compute from args>

    # 1. Not implemented
    if got is None:
        print(f"  {_NONE} Title: not implemented yet (expected {expected:.4f})")
        return

    # 2. Correct
    if _close(got, expected):
        print(f"  {_OK} Title = {got:.4f}")
        return

    # 3. Diagnose common mistakes
    mistake_1 = <compute variant>
    if _close(got, mistake_1):
        print(f"  {_FAIL} Title = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: <targeted explanation of what went wrong>")
    elif ...
    else:
        print(f"  {_FAIL} Title = {got:.4f}  (expected {expected:.4f})")
        print(f"       Hint: <generic formula reminder>")
```

### Common-mistake detection

Each checker anticipates 2-4 typical student errors and gives a specific
diagnosis. Examples from 01_foundations:

| Exercise | Mistake | Detection |
|---|---|---|
| Euclidean | Forgot sqrt | Compare to sum of squares |
| Euclidean | Used abs instead of squaring | Compare to Manhattan result |
| Manhattan | Forgot abs | Compare to raw sum |
| Manhattan | Squared instead of abs | Compare to sum of squares |
| Cosine sim | Returned raw dot product | Compare to unnormalized dot |
| Cosine sim | Returned distance not similarity | Compare to 1-sim |
| Mahalanobis | Used cov instead of inv(cov) | Compute sqrt(delta @ cov @ delta) |
| Mahalanobis | Forgot sqrt | Compare to squared form |
| Z-score | Subtracted mean but didn't divide | Compare to centered-only |
| Z-score | Divided by variance not std | Compare to var-scaled |
| Min-max | Forgot to subtract min | Compare to data/(max-min) |
| Pairwise | All zeros | Check np.allclose(D, 0) |
| Pairwise | Not symmetric | Check D == D.T |

### Output formatting

- Every failure line includes `(expected X.XXXX)` so students always
  have a reference value.
- The ⬜ (not-implemented) state also shows the expected value.
- Indentation: 2 spaces, emoji, space, text. Hint lines indented to
  align.
- Tolerance: `_TOL = 0.01` via `abs(a - b) < tol`.

---

## 5. Plot Style (Tufte)

All plots follow the minimalist Tufte style from the lecture notes'
TikZ figures.

### Lecture TikZ reference

```latex
axis line style={draw=none},
tick style={black, thick},
tick align=outside,
tick pos=left,
only marks, mark=*, mark size=2pt, black
```

### Matplotlib translation

#### Shared helper (defined in imports cell)

```python
def tufte_axis(ax):
    """Remove spines, keep only outward ticks on left and bottom."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', direction='out',
                   length=5, width=1.2, colors='black',
                   top=False, right=False)
```

#### Scatter plots

```python
ax.scatter(x, y, c=colors, s=20, linewidths=0)
tufte_axis(ax)
```

- `s=20` — small markers (matches `mark size=2pt`).
- `linewidths=0` — no marker edge.
- No titles on individual axes (section headings serve that role).
- Legends: `frameon=False, fontsize=9`.

#### Heatmaps

```python
im = ax.imshow(D, cmap='magma_r', aspect='equal')
ax.set_title(title, fontsize=11, pad=8)
ax.set_xticklabels(ids, rotation=90, fontsize=5)
ax.set_yticklabels(ids, fontsize=5)
ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
```

- Tick length 0 (labels only, no tick marks).
- White separator lines at variety/group boundaries.
- `cmap='magma_r'` — dark=close, bright=far.

#### Contour plots

```python
ax.contour(xx, yy, Z, levels=[0.5, 1.0, 1.5],
           colors='black', linewidths=0.8)
ax.set_aspect('equal')
ax.plot(0, 0, 'k+', markersize=8)
tufte_axis(ax)
```

#### General rules

- No grid lines. Ever.
- No box around the plot (all spines removed).
- Subplot titles use `fontsize=10` or `fontsize=11`, never bold.
- `plt.tight_layout()` before `plt.show()`.
- `figsize` — wide: `(12, 4.5)` for 1x2, `(14, 4)` for 1x3,
  `(12, 9)` for 2x2.
- Color palette: `tab:blue`, `tab:orange`, `tab:red`, `tab:green` for
  the four coffee varieties (or equivalent categorical palette).

---

## 6. Text & Formatting Rules

- **No LaTeX math** in notebooks. Use plain language and NumPy notation.
  Students should "revisit the lecture notes" for formal definitions.
- **No `--` or `---`** anywhere. Use `—` (em dash) or rephrase.
- **No numbered section headings.** Use `### EMOJI Title` for exercises.
- **No code hints in the notebook.** The checker system replaces them.
- **Em dashes** for parenthetical asides: `— not a shortcut —`.
- Unicode ranges: `1–10` (en dash), not `1-10`.
- Observation blocks after plot cells use **bold** lead-in:
  `**Observe:**` followed by a bullet list.

---

## 7. Data & Sample Selection

- Use a single, themed dataset per notebook (e.g. coffee_samples.csv).
- The dataset should have:
  - Features on **deliberately different scales** to motivate
    normalization.
  - Natural groupings (varieties/classes) to reveal through analysis.
  - 15-30 samples — small enough to print, large enough for heatmaps.
- Reference samples for exercises should be chosen to **maximize
  disagreement between metrics**. For 01_foundations:
  - S01 (cold brew) vs S12 (espresso): identical brew strength (23.0),
    opposite taste profiles. Cosine says nearly identical, Euclidean
    sees a large gap. The story writes itself.

---

## 8. Notebook Continuity

When a notebook depends on functions from a previous one:

- Provide reference implementations in a markdown + code cell pair:
  ```markdown
  The distance functions below are reference implementations from
  notebook 01. If you completed 01, feel free to paste your own
  versions instead.
  ```
  Followed by a code cell with the implementations.
- Re-load the dataset (don't assume prior notebook was run).
- Re-define `tufte_axis` and any shared helpers.
- Import the relevant checkers from `checks.py`.

---

## 9. Solutions Notebook

- Named `NN_topic_solutions.ipynb`.
- Identical structure to the student notebook.
- Exercise cells have the solution filled in instead of `pass`.
- All opening questions, thoughts, and observations remain.
- Exists as instructor quick-reference, not distributed to students.

---

## 10. File Organization

```
notebooks/
  01_foundations.ipynb            # student notebook
  01_foundations_solutions.ipynb  # instructor reference
  02_centroid_clustering.ipynb    # next topic
  checks.py                      # all auto-checker functions
  coffee_samples.csv             # shared dataset
  notebookskill.md               # this file
```

- One `checks.py` per chapter or shared across chapters.
- CSV data files live alongside notebooks.
- No helper scripts left behind after generation.
