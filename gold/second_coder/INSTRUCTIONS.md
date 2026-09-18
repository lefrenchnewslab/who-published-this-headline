# Same-event coding — second rater

You will see 50 pairs of French news headlines. For each pair, decide whether the
two headlines report **the same specific real-world occurrence** (the same event),
not merely the same topic or theme.

- Label **1** if both headlines are about the same specific event
  (e.g. both about the same vote, the same match, the same incident on the same day).
- Label **0** otherwise — including when they share a topic but describe different
  events (e.g. two different matches, two different votes, general-vs-specific).

Judge the event, not the wording or the outlet. Fill the `same_event` column with
1 or 0 for every row; leave none blank. You are blind to the cosine similarity and
to the first coder's labels by design — code each pair on its own.
