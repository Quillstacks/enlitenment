# Contributing

Thanks for considering a contribution. This repo is a living set of lecture notebooks, and improvements from other instructors, students, and practitioners are welcome.

## Scope of contributions

Good candidates for a PR:

- A bug fix in a notebook, checker, or dataset
- A clarification in a markdown cell (wording, typo, broken link)
- A new exercise or observation within an existing lecture
- A new lecture inside an existing course (e.g. a new chapter of unsupervised deep learning)
- Infrastructure: build tweaks, CI improvements, deployment fixes

Please open an issue first for:

- A new course (not just a new lecture)
- A change in notebook conventions (see `notebookskill.md`)
- A change in plot style (see `content/plot_style.py`)

This keeps bigger changes discussable before anyone invests time.

## Before you start

1. Read [`notebookskill.md`](notebookskill.md). It is the style bible for every notebook in the repo, covering structure, exercise cells, the checker pattern, Tufte plot style, and text formatting rules. Contributions that diverge from it will be asked to realign.
2. Install dependencies and run the build once to confirm it works on your machine (see `README.md`).
3. Open the notebook you want to edit in Jupyter Lab or VS Code. Run it end to end to confirm a clean baseline before editing.

## Notebook conventions at a glance

Full rules live in `notebookskill.md`. The essentials:

- **Three-state checker pattern.** Each exercise has a checker in the lecture's `checks.py`. Not attempted shows `⬜`, correct `✅`, wrong `❌` with a targeted hint. Checkers compute expected values from the input, never hardcoded.
- **Exercise cell template.** Markdown intro with a blockquote opening question and a `<details><summary>Thought</summary>` block, then a code cell with `# YOUR CODE HERE` and `pass`, then the checker call.
- **Plot style.** Always `from plot_style import *` and `tufte_axis(ax)`. Dark background, no grid lines, `frameon=False` on legends.
- **Text rules.** No LaTeX in notebooks, no em or en dashes in markdown cells, no numbered headings, observation blocks use `**Observe:**`.

## Adding a new lecture

1. Create `content/<course-name>/lecture-NN/`.
2. Add `NN_topic.ipynb` (student) and `NN_topic_solutions.ipynb` (instructor reference). Both notebooks should be cell-aligned so the only differences are the exercise cells.
3. Add `checks.py` with one checker per exercise. Each checker anticipates 2 to 4 common student mistakes and gives a targeted hint.
4. Add any dataset files (CSV preferred). Datasets larger than a few MB should be generated or downloaded at notebook run time rather than committed.
5. Solution notebooks are automatically excluded from the deployed build by `content/.jupyterliteignore`. Double-check the ignore pattern still matches your file name.

## Running the notebook validation

The CI pipeline executes every student notebook top to bottom. You can run the same check locally:

```bash
pip install -r requirements-notebooks.txt
jupyter nbconvert --to notebook --execute content/**/lecture-*/NN_*.ipynb --output-dir /tmp
```

If your notebook depends on an exercise the student has not filled in, make sure a reference implementation cell (e.g. `X_c = X - X.mean(axis=0)`) appears right after the exercise so downstream cells still run.

## Pull request process

1. Fork the repo and create a feature branch.
2. Make your changes. Keep PRs focused, one lecture or one theme per PR.
3. Run the notebook end to end. Run the CI check locally if you touched notebook code.
4. Open a PR against `main`. Fill out the PR template.
5. A maintainer will review. Expect feedback on notebookskill.md compliance and checker quality.

## Code of conduct

Be kind, direct, and assume good faith. This is a teaching repo; every change should make the student experience better. Personal attacks, harassment, or dismissive behaviour toward learners will get you removed.

## Questions

Open an issue with the `question` label, or reach out to the maintainers on the PR or issue thread.
