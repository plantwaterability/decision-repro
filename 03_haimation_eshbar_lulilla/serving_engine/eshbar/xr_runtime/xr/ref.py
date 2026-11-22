# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.

from xr.ops import logadd_table, vrand_constants

def wrap(v: int, N: int) -> int:
    v &= (1 << N) - 1
    return v - (1 << N) if v >= 1 << N - 1 else v

def unsigned(v: int, N: int) -> int:
    return v & (1 << N) - 1

def max_(b: int, c: int) -> int:
    return max(b, c)

def logadd(b: int, c: int, N: int, frac_bits: int) -> int:
    table = logadd_table(frac_bits)
    d = abs(b - c)
    return wrap(max(b, c) + (table[d] if d < len(table) else 0), N)

def xlat_address(b: int, c_col: int, s: int, N: int) -> int:
    k = s.bit_length() - 1
    return c_col | unsigned(b, N) & (1 << min(k, N)) - 1

def shift(b: int, c: int, N: int) -> int:
    if c >= N:
        return 0
    return wrap(b << c, N) if c >= 0 else b >> -c

def add_carry(b: int, c: int, N: int) -> int:
    return unsigned(b, N) + unsigned(c, N) >> N

def sub_carry(b: int, c: int, N: int) -> int:
    return int(unsigned(b, N) >= unsigned(c, N))

def adc(b: int, c: int, carry: int, N: int):
    total = unsigned(b, N) + unsigned(c, N) + carry
    return (wrap(total, N), total >> N)

def mulstep(hi: int, lo: int, mc: int, N: int):
    lo_u = unsigned(lo, N)
    total = unsigned(hi, N) + (unsigned(mc, N) if lo_u & 1 else 0)
    return (wrap(total >> 1, N), wrap((total & 1) << N - 1 | lo_u >> 1, N))

def divstep(rem: int, quo: int, d: int, N: int):
    quo_u = unsigned(quo, N)
    r = 2 * unsigned(rem, N) + (quo_u >> N - 1)
    q = quo_u << 1 & (1 << N) - 1
    if r >= unsigned(d, N):
        r -= unsigned(d, N)
        q |= 1
    return (wrap(r, N), wrap(q, N))

def lfsr(b: int, c: int, N: int) -> int:
    u = unsigned(b, N)
    return wrap(u >> 1 ^ (unsigned(c, N) if u & 1 else 0), N)

def clz(c: int, N: int) -> int:
    u = unsigned(c, N)
    return N - u.bit_length()

def abs_(c: int, N: int) -> int:
    return wrap(abs(c), N)

def argmax(values) -> int:
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best

def falign(e: int, m: int, target: int, N: int):
    d = target - e
    if d <= 0:
        return (e, m)
    return (target, m >> min(d, N - 1))

def fnorm(e: int, m: int, N: int):
    if m == 0:
        return (-(1 << N - 1), 0)
    u = unsigned(m, N)
    sign = u >> N - 1
    run = 0
    while run < N and u >> N - 1 - run & 1 == sign:
        run += 1
    k = run - 1
    return (wrap(e - k, N), wrap(m << k, N))

def vrand(seed: int, tag: int, N: int) -> int:
    mask = (1 << N) - 1
    c, K, h1, h2 = vrand_constants(N)
    u = unsigned(seed, N) + sum((cp for p, cp in enumerate(c) if tag >> p & 1)) & mask
    v = u ^ u >> h1
    w = K * v & mask
    return wrap(w ^ w >> h2, N)

def bitrev(i: int, t: int) -> int:
    return int(format(i, f'0{t}b')[::-1], 2) if t else 0

def transpose_index(i: int, t: int) -> int:
    if t % 2:
        return i
    a = t // 2
    return (i & (1 << a) - 1) << a | i >> a
