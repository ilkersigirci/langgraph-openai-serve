# Code Style

## Priorities

Prioritize correctness and security. Prefer simple, readable, maintainable code;
optimize performance when requirements or evidence justify it.

- Do not copy an existing pattern merely because it exists. Question code that
  is unclear, fragile, or unnecessarily complex, and improve it when doing so
  remains within the task's scope.
- Keep changes focused. A small change in legacy code does not justify an
  unrelated refactor.
- This project is in early development. Prefer a clean current design over
  compatibility shims or deprecation layers. Project APIs may break, but
  preserve required external contracts such as OpenAI compatibility. Update
  affected callers, tests, and documentation in the same change.
- Resolve consequential ambiguity before coding. State assumptions and surface
  meaningful tradeoffs instead of silently choosing an interpretation.

## Simplicity

Write the minimum code that fully solves the stated problem.

- Do not add speculative features, configuration, abstractions for a single
  use, or handling for scenarios excluded by validated contracts.
- Prefer early returns and small, cohesive units over deep nesting, monster
  functions, and god objects.
- Keep file and package structure deliberate. Avoid both unrelated
  responsibilities in one module and needless file sprawl.

## Design and Types

- Prefer composition and dependency injection over inheritance and hidden
  dependencies, especially when they make behavior easier to test.
- Keep boundaries fully typed. Validate untyped external data at the boundary,
  using Pydantic where appropriate, then pass precise types through the core.
  Avoid `Any` and coarse containers such as bare `dict` when a real type is
  available.
- Prefer immutable values and avoid unnecessary reassignment, but do not force
  awkward immutable designs where mutation is the clearest local solution.
- Before duplicating parsing or translation across API surfaces, look for an
  existing shared helper. Add one at the common boundary when the behavior is
  genuinely shared.
- Prefer standard protocols, official SDKs, and established libraries over
  hand-rolled equivalents.

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
- Every lint or type suppression must name the exact rule and include a concise
  reason. Do not use a broad suppression to hide an avoidable violation.

Inspired by [Writing system software: code comments](https://antirez.com/news/124).

## Tests

- Add focused tests for behavior changes. A useful test fails before the
  feature exists or when the behavior regresses; coverage without such a signal
  is not a goal.
- For a bug fix, add a regression test that reproduces the specific failure.
- Assert observable behavior at the narrowest stable public boundary. A
  behavior-preserving refactor should not normally require test changes.
- Avoid change-detector tests that mirror production logic, mock every
  collaborator, or assert incidental call order and internal steps. Test an
  interaction only when that interaction is itself a required contract.
- Prefer realistic inputs and independently derived expected results. If a test
  cannot distinguish correct from incorrect behavior, rewrite or delete it.
- Prefer one focused test per behavior. Use parametrization for equivalent
  cases, and do not split tests merely to increase test count or coverage.
  Extend the closest existing test module when practical.
- Test through public behavior where possible. Prefer injected dependencies
  over monkeypatching class internals.

See Google's
[change-detector test guidance](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html)
for the rationale.
