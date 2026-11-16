# Native finalist acceptance

Run this after staging the candidate pack and its projector. Start from the repository root and give each run a new report filename.

```bash
python -m scripts.mind.accept_finalist /path/to/new-report.json \
  --native-data /path/to/staged-data \
  --server /path/to/verified/llama-server
```

The driver runs one scenario at a time. Pass `--case repo` or `--case history` to isolate a failure. The full suite also checks stale-path recovery, the dashboard-to-HTML workflow, and the existing 3D hatch fixture. Keep the server build, reasoning setting, and deadlines fixed across candidates. Chrome is required for the two artifact checks.

- **Repository** first asks for a read-only diagnosis. It then asks for a repair in the same chat, using the earlier finding. The driver checks the exact file set changed, the original Git HEAD and index, and all four fixture tests.
- **History** records an early constraint in a chat, seeds about 18,000 estimated tokens of later synthetic notes, and asks for the earlier directory and checkpoint. It checks exact recovery, the compaction boundary, and the untouched workspace. Use `--history-tokens` to repeat the check at another size.
- **Artifacts** run the app renderer and independent browser checks. The report includes hashes and sizes of files that were actually produced.

The report includes setup, execution and teardown outcomes, the staged artifact and runtime digests, elapsed time, timeouts, and sampled RAM, swap and GPU memory. A pass requires every selected scenario to finish and pass. The driver uses temporary app state and an isolated home and reports fixture metrics without chat text, environment values or absolute paths. `--scripted` exercises the runner and postcondition checks without model inference.

Use `--turn-timeout` and `--suite-timeout` to set bounds in seconds. A timed-out turn stops accepting tool approvals. The suite deadline terminates the isolated test process and its children.
