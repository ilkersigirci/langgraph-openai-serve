# Code Style

## Comments

Write comments for the reader, not as a transcript of the code.

- Explain information that the code cannot express on its own, especially why
  an action is necessary, why its location or order matters, or why an
  apparently more natural alternative was rejected. Capture relevant
  constraints, invariants, tradeoffs, and regression risks.
- Avoid narrating clear code line by line.
- Use comments when they materially lower cognitive load. A concise description
  of state, stages, or intent can be worthwhile even when a determined reader
  could reconstruct it from the code.
- Keep comments close to the behavior they explain, and update or remove them
  when that behavior changes.

Inspired by [Writing system software: code comments](https://antirez.com/news/124).
