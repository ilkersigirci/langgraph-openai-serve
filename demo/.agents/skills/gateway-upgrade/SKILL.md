---
name: gateway-upgrade
description: Upgrade the demo LiteLLM or Bifrost gateway and check whether upstream fixes allow existing routing configuration or UI workarounds to be simplified.
---

# Demo Gateway Upgrade

Use the requested gateway release, verify the demo's OpenAI contract, and
remove only workarounds that the new release demonstrably replaces. An
assessment-only request does not authorize changing the running gateway.
Paths below are relative to the repository root.

LiteLLM defaults to the public `ghcr.io/ilkersigirci/homeserver-litellm` image;
`DEMO_LITELLM_IMAGE` in `demo/.env.example` owns its default pin; users can
change the value in `demo/.env`. Upgrade that image in place unless the user
requests a different distribution. Bifrost uses its official image. Keep image
builds and patch maintenance outside this repository.

## Establish the Baseline

Read `demo/AGENTS.md`, `.agents/CODE_STYLE.md`, and `tests/README.md`.
Use `docs/how-to-guides/openai-proxies.md` for the routing contract and
`docs/explanation/openai-compatibility.md` before changing API behavior.

Inspect the affected gateway's:

- LiteLLM image tag and digest in `demo/.env.example`, and Bifrost's pin in
  `demo/docker/apps/bifrost.yml`;
- configuration under `demo/docker/configs/{litellm,bifrost}/`;
- focused suite in `demo/api/tests/integration/test_{litellm,bifrost}_proxy.py`
  and shared `test_direct_responses.py`;
- endpoint selection and callers under `demo/ui/chainlit_ui/src/lgos_chainlit/`
  and `demo/ui/openwebui/src/lgos_openwebui/` when considering UI simplification.

Check the running demo image and Compose overlays. When available, run the
existing gateway suite before upgrading to distinguish regressions from known
failures. Read strict `xfail` reasons; do not assume they still describe the
requested release.

## Compare Upstream Fixes With Our Actual Paths

Read official release notes, relevant pull requests, and implementation at the
requested tag. A fix mentioning Responses does not necessarily affect the
route or provider used here. In particular:

- Distinguish native Responses from Responses-to-Chat bridging. LGOS uses
  `store: false`, owns interrupt checkpoints, and leaves ordinary history to
  clients. Bridge history fixes do not replace that contract.
- For LiteLLM streaming, inspect concrete-model capability lookup as well as
  wildcard deployment metadata. A wildcard capability flag alone does not
  prove upstream commentary events survive.
- Distinguish model-bound uploads and encoded file IDs from the demo's shared
  Files provider. Preserve one file namespace across both graph APIs.
- Verify model list and detail separately from inference. Catalog extensions
  can still require pass-through even when native Responses works.
- Check response-ID handling and continuation after streaming as well as
  non-streaming creates. Clients must return opaque IDs unchanged.
- Check the selected image's startup, authentication, migrations, and route
  availability; preserve the demo's custom LiteLLM image and streaming opt-in.

Prefer upstream configuration over custom adapters. Remove an exact model
entry, pass-through, or UI helper only when the replacement preserves its
observable contract. A version bump with no safe simplification is valid.

## Update and Verify

Resolve the requested image's registry digest and update its pin at the location
above. These gateways run as external images; Python dependencies and lockfiles
normally need no change. For an authorized local upgrade, preserve the running
Compose overlays and recreate only the affected gateway with `--no-deps`.
Wait for health before testing; do not reset its database or restart unrelated
services to make a test pass.

From the repository root, use
`just demo/test-litellm --editable` or
`just demo/test-bifrost --editable`.
LiteLLM checks native model info and managed routing against direct LGOS
streaming; Bifrost also runs the shared pass-through contract. Validate the
selected Compose profile with
`OPENAI_GATEWAY_TYPE=litellm just demo/compose-config` (or
`bifrost`).

Cover both graph providers, text, commentary and `phase`, function-output
continuation, Files lifecycle and input IDs, catalog metadata, and OpenAI
errors. Extend the closest behavior test for relevant gaps. Keep remaining
expected failures strict; remove an `xfail` only after verifying the restored
behavior, and investigate an unexpected pass. Inspect gateway logs too:
successful client responses do not prove background usage logging works.
If UI behavior changes, run its focused checks and follow
`demo/.agents/ui_checks.md` for browser verification. Report unavailable live
checks explicitly instead of presenting source inspection as a passing test.

## Keep Documentation Stable

Keep exact release pins at the locations above and version-specific failure evidence near
the integration tests or configuration that needs it. Published docs describe
the bundled setup, user-facing behavior, and current operational limitations;
update them when those facts change. Do not scatter version numbers through
prose and diagrams or add upgrade assessments, PR-by-PR analyses, test-run
counts, or release histories under `docs/`. Keep reusable upgrade guidance in
this skill and report release-specific findings, upstream links, validation,
and any retained workarounds in the task handoff.
