# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Changed file: modified from the original work (see NOTICE).

import numpy as np
from dataclasses import dataclass

@dataclass
class SUBLEQConfig:
    s: int = 8
    m: int = 8
    n: int = 32
    N: int = 8
    lam: float = 10.0

    def __post_init__(self):
        self.logn = int(np.log2(self.n))
        self.nrows_cmds = 3 * self.logn
        self.nrows_memory = self.N
        self.nrows_scratchpad = 3 * self.logn + 2 * self.N
        self.nrows_pc = self.logn
        self.nrows_pos_enc = self.logn
        self.nrows_buffer = max(3 * self.logn, 2 * self.N)
        self.nrows_indicator = 1
        self.d_model = self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad + self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer + self.nrows_indicator
        self.idx_memory = self.nrows_cmds
        self.idx_scratchpad = self.idx_memory + self.nrows_memory
        self.idx_pc = self.idx_scratchpad + self.nrows_scratchpad
        self.idx_pos_enc = self.idx_pc + self.nrows_pc
        self.idx_buffer = self.idx_pos_enc + self.nrows_pos_enc
        self.idx_scratch_cmd = self.idx_scratchpad + 2 * self.N
