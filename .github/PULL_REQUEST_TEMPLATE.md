# Summary

<!-- What does this PR do and why? One or two sentences. -->

# Type of change

- [ ] Bug fix in an existing notebook, checker, or dataset
- [ ] New exercise inside an existing lecture
- [ ] New lecture
- [ ] Wording, typo, or markdown fix
- [ ] Infrastructure (build, CI, deploy)
- [ ] Other (describe below)

# Checklist

- [ ] I have read `notebookskill.md` and my changes follow the conventions (structure, exercise pattern, checker system, plot style, text rules)
- [ ] The student notebook runs top to bottom without errors
- [ ] The solutions notebook runs top to bottom and all checkers return `✅`
- [ ] The student and solutions notebooks are cell-aligned (only the exercise cells differ)
- [ ] New exercises have a corresponding checker in `checks.py` that anticipates common mistakes
- [ ] No em or en dashes in markdown cells, no LaTeX math in notebooks
- [ ] No large binaries committed (datasets over a few MB should be generated or fetched at runtime)

# Screenshots or notes

<!-- If visual output changed, paste a screenshot. If you changed the checker logic, describe the mistakes it now catches. -->
