# Call-site argument-count mismatch is silently accepted (no arity check) — binds garbage / shifts args

**Filed:** 2026-06-23 (by a tentib consumer — tentib 0.4.0 M3c, the matmul-free kernel
gained a bias param and three call sites went stale)
**Severity:** P2 — silent-miscompile footgun. Compiles clean (`OK`), wrong runtime
result, **no diagnostic**. The worst failure mode in a no-bounds-checking language.
**Component:** parser / call resolution — function-call argument-count validation.
**Toolchain:** observed on cyrius 6.2.37 (host x86-64).
**Repro:** [`repros/2026-06-23-call-arity-no-check.cyr`](repros/2026-06-23-call-arity-no-check.cyr)

## Observation

Cyrius does not check that a call's positional-argument count matches the callee's
declared parameter count. A call with **too few** args leaves the trailing params
**unbound** (they read whatever is in the arg register/slot — 0 or garbage); a call
with **too many** silently ignores / shifts. Either way `cyrius build` reports `OK`.

Minimal repro (full file in `repros/`):

```cyrius
fn add3(a, b, c): i64 { return a + b + c; }
fn main(): i64 {
    var r = add3(10, 20);   # MISSING the 3rd arg
    return r;               # if checked: compile error. else: 30 (c bound to 0/garbage)
}
```

```
$ cyrius build 2026-06-23-call-arity-no-check.cyr arity && ./arity ; echo $?
compile 2026-06-23-call-arity-no-check.cyr -> arity [x86_64] OK
30          # no error; `c` was silently 0
```

## How it bit a real consumer (tentib M3c)

tentib's matmul-free inference kernel grew a bias parameter:

```cyrius
# was: fn ternary_matmul_free(qx, sx, Wq, gamma,    y, M, K, N)   # 8 params
# now: fn ternary_matmul_free(qx, sx, Wq, gamma, b, y, M, K, N)   # 9 params
```

Three call sites (`tests/tentib.tcyr`, `src/main.cyr` ×2) still passed the old **8**
arguments. The build said `OK`. At runtime the 8 args bound as
`(qx, sx, Wq, gamma, b=y_ptr, y=M, M=K, K=N, N=unbound)` — so the kernel wrote through
a mis-bound output pointer and looped on a garbage `N`. The **only** signal was a
failed logit-parity test (81/82). A subsystem without that test would have shipped
silent garbage — exactly the "compiling ≠ working" trap.

## Why this is always a bug (cheap to catch)

Cyrius has **no overloading and no default arguments**, so positional arity is
unambiguous: an argument-count mismatch is *never* intentional. This is the canonical
class of error a compiler catches for free, and the one most costly to debug at
runtime in a language with no bounds/type safety net to backstop it.

## Proposed fix

At call resolution, compare the call's positional-arg count to the callee's declared
param count; emit a **hard error** on mismatch (or, if a softer landing is wanted for
a transition window, a warning that an opt-in flag promotes to an error):

```
error: `ternary_matmul_free` expects 9 arguments, got 8
  --> src/main.cyr:131:5
```

**Carve-outs to consider:** truly variadic builtins (`syscall`, `print`, any
`asm`-backed varargs) need an exemption or an explicit variadic marker on the
declaration. Everything user-declared with a fixed param list should be checked.

## Workaround (consumer side, until fixed)

None at compile time. The only defense today is a runtime test that exercises the
changed call path — which is what caught it here. Treat any signature change as
requiring a full test run, never a clean `build` as sufficient
(per the ecosystem's "QEMU-test / compiling ≠ working" discipline).

---
*Mirrored consumer-side at `tentib/docs/development/issues/2026-06-23-cyrius-call-arity-no-check.md`.*
