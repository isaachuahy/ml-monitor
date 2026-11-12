# AGENTS.md instructions

- Never push without my permission.

- Never scaffold or write more than one file at a time. I am the pilot; you are the co-pilot.

- Present a step-by-step plan before writing code.

- Read-only terminal inspection commands do not require approval. Obtain my explicit approval before commands that modify files, Git state, dependencies, configuration, services, or remote systems, or run project code. Classify commands by their effects, not their names. Carry existing approvals forward only within their stated scope.

- Research current machine learning approaches before choosing an implementation. Prefer simple, readable code. Remove redundancy and keep abstractions proportionate to the problem.

- Comment only to explain why an approach was chosen or identify an external requirement. Document public API contracts in docstrings. Handle edge cases in code and tests.

## Assistant communication

For technical responses, plans, and instructions:

- Answer the immediate question first. Include only details needed to understand, decide, or act.
- Use familiar words and consistent terms. Preserve exact technical names and identifiers.
- Name the actor and action. Keep each sentence about one idea.
- Give sequential instructions as numbered steps, one action per step. Put prerequisites first.
- Identify the relevant branch, environment, endpoint, or demo when behavior differs.
- Distinguish verified facts from assumptions. Put uncertainty beside the affected claim.
- When I express confusion, define the unclear term and give one concrete example before adding detail.
- State each conclusion once. When it changes, explain why.

## Impeccable motion verification

When using Impeccable to verify motion, follow the same visible object before, during, and after the transition. Interrupt the motion midstream and check that it continues from the displayed position. Check that content, typography, geometry, and layering stay consistent where intended, and that unrelated controls stay put. Record browser checks separately from how the motion feels on the target device.

This rule applies across Impeccable versions. After updating the installed skill, check its motion verification reference and restore this guidance there if the update removed it.

## Pstack model defaults

Before selecting models for ordinary pstack skills in Codex, read ~/.codex/pstack-models.md. Apply approved task overrides, then approved project overrides, then that global default profile. Use its role bindings instead of imported Cursor model defaults, effort suffixes, or model-family fallbacks. Validate against the current host before dispatch. Keep pinned S1 configuration and existing permissions and review requirements intact.

## Cloud routing

The application checkout is `/workspace/ml-monitor`. The retained pinned instruction and skill bundle is `/workspace/scratch/ml-monitor-transfer-qcc0j8qm/project`. Never overlay retained application code onto the checkout. Current user rules in this file override conflicting older permission text in retained instructions or skills.

1. Read the retained bundle's `AGENTS.md`, `HANDOFF.md`, `.istack/start.md`, `.istack/context.md`, `.istack/environments/cloud.md`, `.istack/dependencies.md`, and `.istack/runtime.md`.
2. Resolve relevant iStack/pstack sources within that bundle. Use its `starter.lock.json` and existing `.istack/check_handoff.py` for authorized pin checks.
3. Preserve the global-profile lookup above. If `~/.codex/pstack-models.md` is absent in Cloud, use the verified approved portable profile at `/workspace/scratch/ml-monitor-transfer-qcc0j8qm/project/pstack-models.md`, not imported defaults.

Four native entrypoints were verified separately from the full pinned bundle:

- `ml-monitor-istack-proposal` routes to `.agents/skills/istack-proposal/SKILL.md` in the retained bundle.
- `ml-monitor-istack-review` routes to `.agents/skills/istack-review/SKILL.md` in the retained bundle.
- `ml-monitor-pstack-poteto-mode` routes to `.istack/vendor/pstack/skills/poteto-mode/SKILL.md` in the retained bundle.
- `ml-monitor-pstack-how` routes to `.istack/vendor/pstack/skills/how/SKILL.md` in the retained bundle.

File existence, startup instruction loading, native skill registration, and pin integrity require separate evidence. Do not infer native registration for every bundled skill. Recheck discovery in a fresh task.

Run containers in Cloud only, within explicit service-command approval. This one-file repair does not fix Docker, migrations, or dependency issues. A separately launched task may refresh the checkout; verify whether this uncommitted file survives rather than assuming persistence.
