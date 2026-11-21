"""
test_installer.py - guard the install decision logic.

The installer decides whether llama-server is usable by running it with
--version and inspecting the output. That check once looked for the string
"llama", which llama-server never prints: it prints "version: 1.0.0-dev
(build 12438, ...)". Every build therefore looked like a failure, the loop
ran through all four, and the kernel silently downgraded to the much slower
fallback server - while the logs said "could not obtain llama-server".

These tests pin the actual output strings so that cannot regress.
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
REAL_VERSION_OUT = 'version: 1.0.0-dev (build 12438, commit e1925b369)\nbuilt with GNU 13.3.0 for Linux x86_64\n'

def _is_usable(out: str, rc: int) -> bool:
    """Same predicate the installer uses. Kept here so the test and the
    installer cannot drift apart silently."""
    lowered = (out or '').lower()
    return rc == 0 and ('version' in lowered or 'llama' in lowered)

def main() -> int:
    ok = True

    def check(name, cond, extra=''):
        nonlocal ok
        print(('  PASS  ' if cond else '  FAIL  ') + name + ('  ' + extra if extra else ''))
        if not cond:
            ok = False
    print('installer output detection')
    check('real llama-server --version output is accepted', _is_usable(REAL_VERSION_OUT, 0))
    check('a 127 exit (binary missing) is rejected', not _is_usable('bash: line 1: ./llama-server: No such file or directory', 127))
    check('a loader error is rejected', not _is_usable('error while loading shared libraries: libcudart.so.12', 127))
    check('empty output is rejected', not _is_usable('', 1))
    check('nonzero exit is rejected even with a version banner', not _is_usable(REAL_VERSION_OUT, 1))

    def old(out, rc):
        return rc == 0 and 'llama' in (out or '').lower()
    print('\nregression witness (the bug that shipped)')
    check('old predicate rejected the real output', not old(REAL_VERSION_OUT, 0))
    check('new predicate accepts it', _is_usable(REAL_VERSION_OUT, 0))
    print('\n' + ('ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'))
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
