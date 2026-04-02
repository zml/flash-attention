# AGENTS

This branch is being used to isolate an FA3 forward-pass correctness bug that shows up in a custom Llama path while FA2 remains correct.

Primary working notes live in `NOTES.md`.

Working rules for this branch:
- Update `NOTES.md` after each meaningful experiment.
- Commit local checkpoints often.
- Push after each checkpoint commit because the machine is preemptible.
- Prefer small, reversible repro-focused changes over broad refactors.

