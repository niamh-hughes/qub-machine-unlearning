# EchoForge Codex starter pack

Copy the contents of this folder into the existing `Research Proj copy 2/code/`
folder. `AGENTS.md` must end up directly inside `code/`, beside the existing
`README.md`, `requirements.txt`, `notebooks/`, `results/` and `src/` entries.

Do not copy this pack over the whole research project and do not remove the
existing Adult Income work. The Codex prompts deliberately create a separate
EchoForge path beneath the existing code structure.

## How to use it

1. Copy this pack into the `code/` directory.
2. Put `EchoForge_Synthetic_Dataset_CSV_Package.zip` in `code/incoming/`.
   `US-00` creates this directory if it does not exist.
3. Open the `code/` directory in Codex.
4. Give Codex the contents of `codex_stories/US-00-repository-audit-and-scaffold.md`.
5. Review the result and run the listed verification commands.
6. Continue with one user story at a time, in numerical order.

Do not paste all stories into Codex at once. Each story has explicit
preconditions and acceptance criteria so errors are caught before later models
depend on them.

