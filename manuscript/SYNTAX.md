# Manuscript Syntax for Citations, Labels, and Numbering

Use Pandoc citations such as `[@heins2022pymdp]`, section references such as
`[@sec:methodology]`, equation labels such as `{#eq:belief-update}`, figure
labels such as `{#fig:posterior-entropy}`, and table labels such as
`{#tbl:candidate-scores}`.

Do not write hardcoded prose like `Figure N`, `Equation N`, or `Table N`.
Reference labels with `[@fig:...]`, `[@eq:...]`, and `[@tbl:...]` so the
template/pandoc-crossref lane can number them.
