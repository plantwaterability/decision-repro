# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Changed file: modified from the original work (see NOTICE).

import math
from typing import List
from dataclasses import dataclass
from xr.base import SUBLEQConfig
OP_HALT = 0
OP_MOV = 1
OP_ADD = 2
OP_JMP = 3
OP_JZ = 4
OP_JNZ = 5
OP_INC = 6
OP_DEC = 7
OP_SHL = 8
OP_SHR = 9
OP_CMP = 10
OP_LOAD = 11
OP_AND = 12
OP_OR = 13
OP_XOR = 14
OP_SUB = 15
OP_FIND = 16
OP_SWAP = 17
OP_CMOV = 18
OP_MULACC = 19
OP_STORE = 20
OP_MAX = 21
OP_LOGADD = 22
OP_XLAT = 23
OP_SHIFT = 24
OP_ADC = 25
OP_MULSTEP = 26
OP_DIVSTEP = 27
OP_LFSR = 28
OP_CLZ = 29
OP_ABS = 30
OP_ARGMAX = 31
OP_JMPI = 64
OP_FALIGN = 65
OP_FNORM = 66
OP_VSETVL = 67
OP_VMOV = 68
OP_VADD = 69
OP_VSUB = 70
OP_VMAX = 71
OP_VLOGADD = 72
OP_VCADD = 73
OP_VFILL = 74
OP_VSHIFT = 75
OP_VXLAT = 76
OP_VREV = 77
OP_VTRANS = 78
OP_VRAND = 79
OP_VPERMX = 84
OP_VADDX = 85
OP_VBFLY = 86
OP_VMAXX = 87
OP_VLOGADDX = 88
VECTOR_OPCODES = (OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD, OP_VCADD, OP_VFILL, OP_VSHIFT, OP_VXLAT, OP_VREV, OP_VTRANS, OP_VRAND, OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX, OP_VLOGADDX)
PAGE2_OPCODES = (OP_JMPI, OP_FALIGN, OP_FNORM, OP_VSETVL) + VECTOR_OPCODES

def vrand_constants(N: int):
    import random as _random
    rng = _random.Random(0)
    mask = (1 << N) - 1
    c = [rng.randrange(1, 1 << N) | 1 for _ in range(N)]
    K = (round((1 << N) * 0.8242) | 1) & mask
    return (c, K, max(1, 3 * N // 8), N // 2)

def logadd_table(frac_bits: int) -> List[int]:
    scale = 2 ** frac_bits
    table = []
    while True:
        v = math.floor(scale * math.log2(1.0 + 2.0 ** (-len(table) / scale)) + 0.5)
        if v == 0:
            return table
        table.append(v)

@dataclass
class Config(SUBLEQConfig):
    log_frac_bits: int = 2
    vector_unit: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.nrows_buffer = 3 * self.N + self.N + self.logn
        self.nrows_tag = self.N
        self.idx_tag = self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad + self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer
        self.nrows_flags = 1
        self.idx_carry = self.idx_tag + self.nrows_tag
        self.page2 = self.logn >= 7
        self.vector = self.page2 and self.vector_unit
        if self.vector:
            self.idx_vcarry = self.idx_carry + 1
            self.idx_memcol = self.idx_carry + 2
            self.idx_vgate = self.idx_carry + 3
            self.idx_vmask = self.idx_carry + 4
            self.nrows_flags += 3 + self.logn
        self.d_model = self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad + self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer + self.nrows_tag + self.nrows_flags + self.nrows_indicator
