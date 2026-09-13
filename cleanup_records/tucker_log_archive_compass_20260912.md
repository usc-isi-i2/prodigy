# Tucker top-level log archive compass — 2026-09-12

This record locates raw launch logs that were removed from the top level of
`/dataMeR1/phil/gfm` after preservation in a separate Git repository. The logs
are intentionally not part of the PRODIGY repository or its history.

## Archive

- Tucker Git repository: `/dataMeR1/phil/gfm/log-archive`
- Branch: `main`
- Archive commit: `e94198b9e710344b5b67ed2846a0557a50e0b78c`
- Log tree: `logs/top-level-20260912/`
- Integrity manifest: `SHA256SUMS` at the archive repository root
- Contents: 25 `.log` files, 5,940,484 bytes before Git compression

The archive commit contains exact copies of every top-level `.log` file found at
inspection. Each source was compared byte-for-byte with its committed archive
copy before the source was removed. No top-level `.log` files remained afterward.
No tmux session, matching process, or open file handle was present at deletion.

This is recoverable Git history on Tucker, not an independent off-machine backup.
Loss of the Tucker filesystem could therefore remove both the archive repository
and the original experiment outputs.

## Verify or restore

Verify the archive object database and file checksums from Tucker:

```bash
git -C /dataMeR1/phil/gfm/log-archive fsck --full
cd /dataMeR1/phil/gfm/log-archive
sha256sum --check SHA256SUMS
```

Restore a particular log without changing the archive:

```bash
git -C /dataMeR1/phil/gfm/log-archive show \
  e94198b9e710344b5b67ed2846a0557a50e0b78c:logs/top-level-20260912/<name>.log \
  > /dataMeR1/phil/gfm/<name>.log
```

## Scope

Only files directly under `/dataMeR1/phil/gfm` whose names ended in `.log` were
archived and removed. Logs inside experiment worktrees, run directories, runtime
archives, and other subdirectories were not changed.
