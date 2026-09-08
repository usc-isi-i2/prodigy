# Private artifact inventory

This analysis has a private archive of replay tensors and model outputs that is
intentionally excluded from Git. Canonical aggregate tables, validation records,
figures, and findings remain tracked in this analysis directory.

## Target-mechanism replay archive

- Archive: `target_mechanisms_private_20260907.tar.gz`
- Contents: 145 files, including 122 PyTorch `.pt` artifacts
- Uncompressed file bytes: 75,779,055
- Archive size: approximately 41 MiB
- SHA-256: `756de7bc4550bcc85f4cf3a520a2d0c2ac64261724cb6c42ed7d1b6ab8e5e1ef`
- Source branch: `codex/target-performance-mechanisms`
- Source revision: `3a6f0a4c17ae46a02715fabdd36ba0543e9af82f`
- Original relative path: `log/target_mechanisms/`

Verified private copies as of 2026-09-07:

- Tucker: `/dataMeR1/phil/gfm/private-artifacts/target-performance-mechanisms/target_mechanisms_private_20260907.tar.gz`
- Laptop: `/Users/philipp/projects/gfm/prodigy-private-artifacts/target-performance-mechanisms/target_mechanisms_private_20260907.tar.gz`

Both copies matched the SHA-256 above after transfer. The archive contains
private feature/model tensors and must not be published or added to ordinary
Git history. Restore it only into a private directory and verify the checksum
before loading any tensors.
