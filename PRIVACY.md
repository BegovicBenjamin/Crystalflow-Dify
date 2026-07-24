# CrystalFlow privacy notice

CrystalFlow does not contact external services and does not request credentials.

The plugin stores crystal names, descriptions, deterministic JSON programs, test vectors, version
metadata, and aggregate execution counters in Dify's plugin key-value storage. Dify scopes that
storage to the workspace and plugin identity; records can persist across plugin upgrades or
reinstallation. CrystalFlow does not automatically capture conversation history.

Inputs and outputs supplied to `execute_crystal` are processed in memory and are not stored.
Inputs and expected outputs supplied as crystallization test vectors are stored as part of the
version record so the version can be audited. Do not include secrets or personal data in a crystal
or its test vectors unless your organization's Dify retention policy permits it.

Retiring a crystal disables it but intentionally retains its immutable versions for rollback and
audit. CrystalFlow 0.1.0 has no bulk-purge tool, and uninstalling the plugin does not promise to
erase its persisted KV records. Permanent removal requires a Dify administrator to delete the
plugin's managed persistence/storage according to the deployment's retention procedures.

CrystalFlow performs no analytics, advertising, tracking, or cross-workspace data sharing.
