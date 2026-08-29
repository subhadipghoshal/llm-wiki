# Maintaining This Wiki

Start each page by deciding what breaks without the thing you are about to explain. If you cannot name the failure, you do not understand the component well enough to write the page yet.

## Page workflow

1. Write the concept in under five sentences.
2. Write the smallest implementation that runs, in `src/`.
3. Run it. Then break it deliberately and run it again.
4. Record the ablation result, including the numbers you actually saw.
5. Label every claim Fact, Inference, or Question.
6. Add any paper or doc you leaned on to `references/sources.md`.

## Writing rules

- Show the failure mode, not just the mechanism.
- Keep implementations under roughly 100 lines. Length is a sign the page is doing too much.
- Prefer a measured number over an adjective. "Loss went from 2.51 to 4.17" beats "much worse".
- Never copy a benchmark number from a paper into prose without labeling it Fact and linking it.
- If you did not run it, say you did not run it.
- Dependencies stay at torch and numpy. A new dependency needs a page-level justification.

## Review checklist

- Does the page contain a real ablation with real numbers?
- Is every quantitative claim labeled and sourced?
- Does the code run from a clean checkout with `uv run`?
- Does it still train in under a few minutes on a laptop?
- Would a reader who stopped here still have something that works?

## Frontmatter

Every page under `wiki/` uses the schema in `wiki/_template.md`. It matches the
schema in my publish vault, extended with `type: wiki`, so pages can move into a
site build later without a second contract.
