from pathlib import Path

def replace(path, old, new):
    p=Path(path)
    s=p.read_text()
    if old not in s:
        raise RuntimeError(f"Expected pattern not found in {path}: {old[:80]!r}")
    p.write_text(s.replace(old,new))

# Baseline result schema.
replace("qpeo/baseline_common.py",
'    best_scalar_history: np.ndarray\n    archive_size_history: np.ndarray\n',
'    best_scalar_history: np.ndarray\n    guide_objective_history: np.ndarray\n    archive_size_history: np.ndarray\n')

p=Path("qpeo/baselines.py")
s=p.read_text()
s=s.replace('    best_history: list[float],\n    archive_history: list[int],\n',
            '    best_history: list[float],\n    guide_history: list[np.ndarray],\n    archive_history: list[int],\n')
s=s.replace('        best_scalar_history=np.asarray(best_history, dtype=float),\n        archive_size_history=np.asarray(archive_history, dtype=int),\n',
            '        best_scalar_history=np.asarray(best_history, dtype=float),\n        guide_objective_history=np.asarray(guide_history, dtype=float),\n        archive_size_history=np.asarray(archive_history, dtype=int),\n')
s=s.replace('    best_history: list[float] = []\n    archive_history: list[int] = [len(archive.entries)]\n',
            '    best_history: list[float] = []\n    guide_history: list[np.ndarray] = [objectives[guide_index(objectives, scalar_weights)].as_array().copy()]\n    archive_history: list[int] = [len(archive.entries)]\n')
s=s.replace('        archive_history.append(len(archive.entries))\n',
            '        guide_history.append(objectives[guide_index(objectives, scalar_weights)].as_array().copy())\n        archive_history.append(len(archive.entries))\n')
s=s.replace('        best_history, archive_history, seed,\n',
            '        best_history, guide_history, archive_history, seed,\n')
s=s.replace('archive=_initialize_archive(pop,assignments,objectives); bh=[]; ah=[len(archive.entries)]',
            'archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]')
s=s.replace('; ah.append(len(archive.entries))',
            '; gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy()); ah.append(len(archive.entries))')
s=s.replace('        ah.append(len(archive.entries))',
            '        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())\n        ah.append(len(archive.entries))')
s=s.replace(',archive,bh,ah,seed)', ',archive,bh,gh,ah,seed)')
p.write_text(s)

# QPEO raw guide-objective history.
p=Path("qpeo/optimizer.py")
s=p.read_text()
s=s.replace('    entropy_history: np.ndarray\n    archive_size_history: np.ndarray\n',
            '    entropy_history: np.ndarray\n    guide_objective_history: np.ndarray\n    archive_size_history: np.ndarray\n')
s=s.replace('    entropy_history: list[float] = []\n    archive_sizes: list[int] = [len(archive.entries)]\n',
            '    entropy_history: list[float] = []\n    guide_objective_history: list[np.ndarray] = [objectives[guide].as_array().copy()]\n    archive_sizes: list[int] = [len(archive.entries)]\n')
s=s.replace('        guide = _guide_index(objectives, settings.scalar_weights)\n        archive_sizes.append(len(archive.entries))\n',
            '        guide = _guide_index(objectives, settings.scalar_weights)\n        guide_objective_history.append(objectives[guide].as_array().copy())\n        archive_sizes.append(len(archive.entries))\n')
s=s.replace('        entropy_history=np.asarray(entropy_history, dtype=float),\n        archive_size_history=np.asarray(archive_sizes, dtype=int),\n',
            '        entropy_history=np.asarray(entropy_history, dtype=float),\n        guide_objective_history=np.asarray(guide_objective_history, dtype=float),\n        archive_size_history=np.asarray(archive_sizes, dtype=int),\n')
p.write_text(s)

# Explicit tests.
replace("tests/test_optimizer.py",
'    assert result.entropy_history.shape == (4,)\n',
'    assert result.entropy_history.shape == (4,)\n    assert result.guide_objective_history.shape == (5,6)\n')
replace("tests/test_baselines.py",
'    assert len(result.best_scalar_history) == 4\n',
'    assert len(result.best_scalar_history) == 4\n    assert result.guide_objective_history.shape == (5,6)\n')
replace("tests/test_original_baselines.py",
'    assert len(res.best_scalar_history)==3\n',
'    assert len(res.best_scalar_history)==3\n    assert res.guide_objective_history.shape==(4,6)\n')
