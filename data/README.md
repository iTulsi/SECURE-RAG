# Dataset location

The latest pinned setup and split-generation commands are in the root
[README](../README.md). Reference pilot/holdout manifests and corpus source
hashes are committed under `data/manifests`. Generated benchmark examples
and external snapshots remain local; their sources retain their own licenses.

The project intentionally does not redistribute the upstream benchmark files.
Clone the official repository and pass its `Dataset` directory to the CLI:

```bash
git clone --depth 1 https://github.com/aiforsec/SECURE.git ../SECURE
secure-rag validate --data-dir ../SECURE/Dataset
```

The validator reads the files without modifying them and records their SHA-256
hashes in each pilot manifest.
