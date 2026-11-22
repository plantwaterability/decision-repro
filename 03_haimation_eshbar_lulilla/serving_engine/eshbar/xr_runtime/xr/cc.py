# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Changed file: modified from the original work (see NOTICE).

from __future__ import annotations
import copy
import math
from fractions import Fraction
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union
from xr.ops import Config, OP_HALT, OP_MOV, OP_ADD, OP_SUB, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_SWAP, OP_CMOV, OP_MULACC, OP_STORE, OP_MAX, OP_LOGADD, OP_XLAT, OP_SHIFT, OP_ADC, OP_MULSTEP, OP_DIVSTEP, OP_LFSR, OP_CLZ, OP_ABS, OP_ARGMAX, OP_JMPI, OP_FALIGN, OP_FNORM, OP_VSETVL, OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD, OP_VCADD, OP_VFILL, OP_VSHIFT, OP_VXLAT, OP_VREV, OP_VTRANS, OP_VRAND, OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX, OP_VLOGADDX

class TT(Enum):
    INT = auto()
    FIX = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    VOID = auto()
    NOINLINE = auto()
    IDENT = auto()
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    BANG = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    SEMI = auto()
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    EOF = auto()

@dataclass
class Token:
    type: TT
    value: str
    line: int
KEYWORDS = {'int': TT.INT, 'fix': TT.FIX, 'if': TT.IF, 'else': TT.ELSE, 'while': TT.WHILE, 'for': TT.FOR, 'return': TT.RETURN, 'break': TT.BREAK, 'continue': TT.CONTINUE, 'void': TT.VOID, 'noinline': TT.NOINLINE}
OPERATORS = [('==', TT.EQ), ('!=', TT.NEQ), ('<=', TT.LE), ('>=', TT.GE), ('<<', TT.LSHIFT), ('>>', TT.RSHIFT), ('&&', TT.AND), ('||', TT.OR), ('+=', TT.PLUS_ASSIGN), ('-=', TT.MINUS_ASSIGN), ('+', TT.PLUS), ('-', TT.MINUS), ('*', TT.STAR), ('/', TT.SLASH), ('%', TT.PERCENT), ('&', TT.AMP), ('|', TT.PIPE), ('^', TT.CARET), ('~', TT.TILDE), ('!', TT.BANG), ('<', TT.LT), ('>', TT.GT), ('=', TT.ASSIGN), (';', TT.SEMI), (',', TT.COMMA), ('(', TT.LPAREN), (')', TT.RPAREN), ('{', TT.LBRACE), ('}', TT.RBRACE), ('[', TT.LBRACKET), (']', TT.RBRACKET)]
_NEG_CONTEXT = frozenset((TT.LPAREN, TT.COMMA, TT.ASSIGN, TT.PLUS_ASSIGN, TT.MINUS_ASSIGN, TT.SEMI, TT.LBRACKET, TT.PLUS, TT.MINUS, TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LE, TT.GE, TT.AND, TT.OR, TT.RETURN, TT.AMP, TT.PIPE, TT.CARET, TT.BANG, TT.TILDE))
_OP_TYPE = dict(OPERATORS)
_TOKEN_RE = re.compile('(?P<nl>\\n)|(?P<ws>[ \\t\\r]+)|(?P<lc>//[^\\n]*)|(?P<bc>/\\*(?:.|\\n)*?(?:\\*/|\\Z))|(?P<num>-?[0-9]+(?:\\.[0-9]+)?)|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>' + '|'.join((re.escape(op) for op, _ in OPERATORS)) + ')')

def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    line = 1
    pos, end = (0, len(source))
    match = _TOKEN_RE.match
    while pos < end:
        mt = match(source, pos)
        if mt is None:
            raise SyntaxError(f"Unexpected character '{source[pos]}' at line {line}")
        kind, text = (mt.lastgroup, mt.group())
        if kind == 'num' and text[0] == '-' and tokens and (tokens[-1].type not in _NEG_CONTEXT):
            tokens.append(Token(TT.MINUS, '-', line))
            pos += 1
            continue
        pos = mt.end()
        if kind == 'nl':
            line += 1
        elif kind == 'id':
            tokens.append(Token(KEYWORDS.get(text, TT.IDENT), text, line))
        elif kind == 'num':
            tokens.append(Token(TT.NUMBER, text, line))
        elif kind == 'op':
            tokens.append(Token(_OP_TYPE[text], text, line))
        elif kind == 'bc':
            line += text.count('\n')
    tokens.append(Token(TT.EOF, '', line))
    return tokens

@dataclass
class IntLiteral:
    value: int

@dataclass
class FixLiteral:
    value: float

@dataclass
class VarRef:
    name: str

@dataclass
class ArrayRef:
    name: str
    index: 'Expr'

@dataclass
class UnaryOp:
    op: str
    expr: 'Expr'

@dataclass
class BinaryOp:
    op: str
    left: 'Expr'
    right: 'Expr'

@dataclass
class FuncCall:
    name: str
    args: List['Expr']

@dataclass
class Slot:
    addr: int
Expr = Union[IntLiteral, FixLiteral, VarRef, ArrayRef, UnaryOp, BinaryOp, FuncCall, Slot]

@dataclass
class VarDecl:
    name: str
    size: Optional[int]
    init: Optional[Expr]
    line: int = 0
    type: str = 'int'
    values: Optional[List[int]] = None

@dataclass
class Assignment:
    target: Union[VarRef, ArrayRef]
    op: str
    value: Expr
    line: int = 0

@dataclass
class IfStmt:
    cond: Expr
    then_body: 'Block'
    else_body: Optional['Block']
    line: int = 0

@dataclass
class WhileStmt:
    cond: Expr
    body: 'Block'
    line: int = 0

@dataclass
class ForStmt:
    init: Optional['Stmt']
    cond: Optional[Expr]
    update: Optional['Stmt']
    body: 'Block'
    line: int = 0

@dataclass
class ReturnStmt:
    value: Optional[Expr]
    line: int = 0

@dataclass
class ExprStmt:
    expr: Expr
    line: int = 0

@dataclass
class JumpStmt:
    kind: str
    line: int = 0
Stmt = Union[VarDecl, Assignment, IfStmt, WhileStmt, ForStmt, ReturnStmt, ExprStmt, JumpStmt]

@dataclass
class Block:
    stmts: List[Stmt]

@dataclass
class FuncDecl:
    name: str
    params: List[str]
    body: Block
    ret_type: str = 'int'
    param_types: Optional[List[str]] = None
    noinline: bool = False

@dataclass
class Program:
    functions: List[FuncDecl]
    globals: List[VarDecl]

class Parser:

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt: TT) -> Token:
        t = self.advance()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt.name}, got '{t.value}' at line {t.line}")
        return t

    def match(self, tt: TT) -> Optional[Token]:
        if self.peek().type == tt:
            return self.advance()
        return None

    def parse_program(self) -> Program:
        functions = []
        globals_ = []
        while self.peek().type != TT.EOF:
            if self.match(TT.NOINLINE):
                func = self.parse_func_decl()
                func.noinline = True
                functions.append(func)
            elif self.peek().type in (TT.INT, TT.VOID, TT.FIX):
                saved = self.pos
                self.advance()
                name_tok = self.advance()
                if self.peek().type == TT.LPAREN:
                    self.pos = saved
                    functions.append(self.parse_func_decl())
                else:
                    self.pos = saved
                    globals_.append(self.parse_var_decl())
            else:
                raise SyntaxError(f"Unexpected token '{self.peek().value}' at line {self.peek().line}")
        return Program(functions=functions, globals=globals_)

    def parse_type(self) -> str:
        t = self.advance()
        if t.type not in (TT.INT, TT.FIX):
            raise SyntaxError(f"Expected a type, got '{t.value}' at line {t.line}")
        return t.value

    def parse_func_decl(self) -> FuncDecl:
        ret_type = self.advance().value
        name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        params = []
        param_types = []
        if self.peek().type != TT.RPAREN:
            param_types.append(self.parse_type())
            params.append(self.expect(TT.IDENT).value)
            while self.match(TT.COMMA):
                param_types.append(self.parse_type())
                params.append(self.expect(TT.IDENT).value)
        self.expect(TT.RPAREN)
        body = self.parse_block()
        return FuncDecl(name=name, params=params, body=body, ret_type=ret_type, param_types=param_types)

    def parse_block(self) -> Block:
        self.expect(TT.LBRACE)
        stmts = []
        while self.peek().type != TT.RBRACE:
            stmts.append(self.parse_stmt())
        self.expect(TT.RBRACE)
        return Block(stmts=stmts)

    def parse_stmt(self) -> Stmt:
        t = self.peek()
        if t.type in (TT.INT, TT.FIX):
            return self.parse_var_decl()
        if t.type == TT.IF:
            return self.parse_if()
        if t.type == TT.WHILE:
            return self.parse_while()
        if t.type == TT.FOR:
            return self.parse_for()
        if t.type == TT.RETURN:
            return self.parse_return()
        if t.type in (TT.BREAK, TT.CONTINUE):
            self.advance()
            self.expect(TT.SEMI)
            return JumpStmt('break' if t.type == TT.BREAK else 'continue', line=t.line)
        if t.type == TT.LBRACE:
            return self.parse_block()
        return self.parse_assign_or_expr()

    def parse_var_decl(self) -> VarDecl:
        ln = self.peek().line
        vtype = self.parse_type()
        name = self.expect(TT.IDENT).value
        size = None
        init = None
        values = None
        if self.match(TT.LBRACKET):
            size = int(self.expect(TT.NUMBER).value)
            self.expect(TT.RBRACKET)
        if self.match(TT.ASSIGN):
            if size is not None:
                self.expect(TT.LBRACE)
                values = []
                while self.peek().type != TT.RBRACE:
                    neg = self.match(TT.MINUS) is not None
                    values.append(int(self.expect(TT.NUMBER).value) * (-1 if neg else 1))
                    if not self.match(TT.COMMA):
                        break
                self.expect(TT.RBRACE)
                if len(values) > size:
                    raise SyntaxError(f"Too many initializers for '{name}' at line {ln}")
            else:
                init = self.parse_expr()
        self.expect(TT.SEMI)
        return VarDecl(name=name, size=size, init=init, line=ln, type=vtype, values=values)

    def parse_if(self) -> IfStmt:
        ln = self.peek().line
        self.expect(TT.IF)
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        then_body = self._parse_body()
        else_body = None
        if self.match(TT.ELSE):
            else_body = self._parse_body()
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body, line=ln)

    def parse_while(self) -> WhileStmt:
        ln = self.peek().line
        self.expect(TT.WHILE)
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        body = self._parse_body()
        return WhileStmt(cond=cond, body=body, line=ln)

    def parse_for(self) -> ForStmt:
        ln = self.peek().line
        self.expect(TT.FOR)
        self.expect(TT.LPAREN)
        init = None
        if self.peek().type != TT.SEMI:
            init = self.parse_stmt()
        else:
            self.expect(TT.SEMI)
        cond = None
        if self.peek().type != TT.SEMI:
            cond = self.parse_expr()
        self.expect(TT.SEMI)
        update = None
        if self.peek().type != TT.RPAREN:
            update = self.parse_assign_or_expr_no_semi()
        self.expect(TT.RPAREN)
        body = self._parse_body()
        return ForStmt(init=init, cond=cond, update=update, body=body, line=ln)

    def parse_return(self) -> ReturnStmt:
        ln = self.peek().line
        self.expect(TT.RETURN)
        value = None
        if self.peek().type != TT.SEMI:
            value = self.parse_expr()
        self.expect(TT.SEMI)
        return ReturnStmt(value=value, line=ln)

    def parse_assign_or_expr(self) -> Stmt:
        stmt = self.parse_assign_or_expr_no_semi()
        self.expect(TT.SEMI)
        return stmt

    def parse_assign_or_expr_no_semi(self) -> Stmt:
        ln = self.peek().line
        expr = self.parse_expr()
        if self.peek().type in (TT.ASSIGN, TT.PLUS_ASSIGN, TT.MINUS_ASSIGN):
            op = self.advance().value
            value = self.parse_expr()
            if isinstance(expr, (VarRef, ArrayRef)):
                return Assignment(target=expr, op=op, value=value, line=ln)
            raise SyntaxError(f'Invalid assignment target')
        return ExprStmt(expr=expr, line=ln)

    def _parse_body(self) -> Block:
        if self.peek().type == TT.LBRACE:
            return self.parse_block()
        return Block(stmts=[self.parse_stmt()])

    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.peek().type == TT.OR:
            self.advance()
            right = self.parse_and()
            left = BinaryOp('||', left, right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_bitor()
        while self.peek().type == TT.AND:
            self.advance()
            right = self.parse_bitor()
            left = BinaryOp('&&', left, right)
        return left

    def parse_bitor(self) -> Expr:
        left = self.parse_bitxor()
        while self.peek().type == TT.PIPE:
            self.advance()
            right = self.parse_bitxor()
            left = BinaryOp('|', left, right)
        return left

    def parse_bitxor(self) -> Expr:
        left = self.parse_bitand()
        while self.peek().type == TT.CARET:
            self.advance()
            right = self.parse_bitand()
            left = BinaryOp('^', left, right)
        return left

    def parse_bitand(self) -> Expr:
        left = self.parse_equality()
        while self.peek().type == TT.AMP:
            self.advance()
            right = self.parse_equality()
            left = BinaryOp('&', left, right)
        return left

    def parse_equality(self) -> Expr:
        left = self.parse_comparison()
        while self.peek().type in (TT.EQ, TT.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(op, left, right)
        return left

    def parse_comparison(self) -> Expr:
        left = self.parse_shift()
        while self.peek().type in (TT.LT, TT.GT, TT.LE, TT.GE):
            op = self.advance().value
            right = self.parse_shift()
            left = BinaryOp(op, left, right)
        return left

    def parse_shift(self) -> Expr:
        left = self.parse_additive()
        while self.peek().type in (TT.LSHIFT, TT.RSHIFT):
            op = self.advance().value
            right = self.parse_additive()
            left = BinaryOp(op, left, right)
        return left

    def parse_additive(self) -> Expr:
        left = self.parse_multiplicative()
        while self.peek().type in (TT.PLUS, TT.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(op, left, right)
        return left

    def parse_multiplicative(self) -> Expr:
        left = self.parse_unary()
        while self.peek().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self) -> Expr:
        if self.peek().type == TT.LPAREN and self.tokens[self.pos + 1].type in (TT.INT, TT.FIX) and (self.tokens[self.pos + 2].type == TT.RPAREN):
            self.advance()
            target = self.advance().value
            self.advance()
            return UnaryOp(f'({target})', self.parse_unary())
        if self.peek().type == TT.MINUS:
            self.advance()
            expr = self.parse_unary()
            if isinstance(expr, IntLiteral):
                return IntLiteral(-expr.value)
            if isinstance(expr, FixLiteral):
                return FixLiteral(-expr.value)
            return UnaryOp('-', expr)
        if self.peek().type == TT.TILDE:
            self.advance()
            return UnaryOp('~', self.parse_unary())
        if self.peek().type == TT.BANG:
            self.advance()
            return UnaryOp('!', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        t = self.peek()
        if t.type == TT.NUMBER:
            self.advance()
            if '.' in t.value:
                return FixLiteral(float(t.value))
            return IntLiteral(int(t.value))
        if t.type == TT.IDENT:
            self.advance()
            name = t.value
            if self.peek().type == TT.LPAREN:
                self.advance()
                args = []
                if self.peek().type != TT.RPAREN:
                    args.append(self.parse_expr())
                    while self.match(TT.COMMA):
                        args.append(self.parse_expr())
                self.expect(TT.RPAREN)
                return FuncCall(name=name, args=args)
            if self.peek().type == TT.LBRACKET:
                self.advance()
                index = self.parse_expr()
                self.expect(TT.RBRACKET)
                return ArrayRef(name=name, index=index)
            return VarRef(name=name)
        if t.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TT.RPAREN)
            return expr
        raise SyntaxError(f"Unexpected token '{t.value}' at line {t.line}")

@dataclass
class MemSlot:
    addr: int
    name: str
    size: int = 1

class MemoryAllocator:

    def __init__(self, m: int=64, N: int=8):
        self.m = m
        self.N = N
        self.slots: Dict[str, MemSlot] = {}
        self.constants: Dict[int, int] = {}
        self.initial: Dict[int, int] = {}
        self.next_addr = 0
        self.reserved: Optional[Tuple[int, int]] = None

    def _alloc(self, name: str, size: int=1, align: int=1) -> int:
        self.next_addr = -(-self.next_addr // align) * align
        if self.reserved and self.next_addr < self.reserved[1] and (self.next_addr + size > self.reserved[0]):
            self.next_addr = -(-self.reserved[1] // align) * align
        if self.next_addr + size > self.m:
            raise RuntimeError(f"Out of memory allocating '{name}' (need {size}, {self.next_addr}/{self.m} used)")
        addr = self.next_addr
        self.slots[name] = MemSlot(addr=addr, name=name, size=size)
        self.next_addr += size
        return addr

    def alloc_constant(self, value: int) -> int:
        mask = (1 << self.N) - 1
        half = 1 << self.N - 1
        value = value & mask
        if value >= half:
            value -= 1 << self.N
        if value in self.constants:
            return self.constants[value]
        name = f'__const_{value}'
        addr = self._alloc(name)
        self.constants[value] = addr
        return addr

    def alloc_temp(self, name: str) -> int:
        return self._alloc(name)

    def alloc_var(self, name: str, size: int=1, align: int=1) -> int:
        if name in self.slots:
            return self.slots[name].addr
        return self._alloc(name, size, align)

    def get_var(self, name: str) -> int:
        if name not in self.slots:
            raise NameError(f"Undefined variable '{name}'")
        return self.slots[name].addr

    def has_var(self, name: str) -> bool:
        return name in self.slots

    def init_memory(self) -> List[int]:
        mem = [0] * self.m
        for val, addr in self.constants.items():
            mem[addr] = val
        for addr, val in self.initial.items():
            mem[addr] = val
        return mem
INEXACT_ROOT, DOMAIN_ERROR, EXPONENT_LIMIT, NEG_INT_EXPONENT = (-1, -2, -3, -4)
DIV_MARKS = {INEXACT_ROOT: ('inexact_sqrt', 'root of a rational that is not a perfect power'), DOMAIN_ERROR: ('runtime_error', 'math_domain_error'), EXPONENT_LIMIT: ('step_limit', 'exponent beyond 4096'), NEG_INT_EXPONENT: ('runtime_error', 'negative_int_exponent')}

class ExecutionFault(Exception):

    def __init__(self, reason: str, detail: str):
        super().__init__(f'{reason}: {detail}')
        self.reason = reason
        self.detail = detail

def _wrap(value: int, N: int) -> int:
    value &= (1 << N) - 1
    return value - (1 << N) if value >= 1 << N - 1 else value
VECTOR_BUILTINS = {'vmov': (OP_VMOV, 'array'), 'vadd': (OP_VADD, 'array'), 'vsub': (OP_VSUB, 'array'), 'vmax': (OP_VMAX, 'array'), 'vlogadd': (OP_VLOGADD, 'array'), 'vcadd': (OP_VCADD, 'array'), 'vrev': (OP_VREV, 'array'), 'vtrans': (OP_VTRANS, 'array'), 'vxlat': (OP_VXLAT, 'table'), 'vfill': (OP_VFILL, 'scalar'), 'vshift': (OP_VSHIFT, 'scalar'), 'vrand': (OP_VRAND, 'scalar'), 'vpermx': (OP_VPERMX, 'scalar'), 'vaddx': (OP_VADDX, 'scalar'), 'vbfly': (OP_VBFLY, 'scalar'), 'vmaxx': (OP_VMAXX, 'scalar'), 'vlogaddx': (OP_VLOGADDX, 'scalar')}
VECTOR_REDUCE = {'vsum': OP_VADDX, 'vmaxall': OP_VMAXX, 'vlogsum': OP_VLOGADDX, 'wht': OP_VBFLY}
BUILTINS = {'printf', 'output', 'abs', 'min', 'max', 'mul', 'swap', 'logadd', 'xlat', 'lfsr', 'clz', 'argmax', 'umulhi', 'qdiv', 'mulmod', 'diff', 'falign', 'fnorm', 'fsqrt', 'froot', 'isqrt', 'fpowi', 'ipowi'} | set(VECTOR_BUILTINS) | set(VECTOR_REDUCE)

def _substitute(node, names: Dict[str, 'Expr']):
    if isinstance(node, VarRef) and node.name in names:
        return names[node.name]
    if not hasattr(node, '__dataclass_fields__'):
        return node
    fields = {}
    for k, v in node.__dict__.items():
        if isinstance(v, list):
            fields[k] = [_substitute(i, names) for i in v]
        else:
            fields[k] = _substitute(v, names)
    return type(node)(**fields)

def _walk(node):
    yield node
    for value in getattr(node, '__dict__', {}).values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if hasattr(item, '__dataclass_fields__'):
                yield from _walk(item)

class CodeGen:

    def __init__(self, cfg: Config, checked: bool=False, rational: bool=False):
        self.cfg = cfg
        self.checked = checked
        self.rational = rational
        self.checks: Dict[int, tuple] = {}
        self.fix_literal_rounding = 0
        self.mem = MemoryAllocator(cfg.m, cfg.N)
        self.cmds: List[List] = []
        self.labels: Dict[str, int] = {}
        self.label_counter = 0
        self.functions: Dict[str, FuncDecl] = {}
        self.source_map: List[Tuple[int, int]] = []
        self.CONST_ZERO = self.mem.alloc_constant(0)
        self.CONST_ONE = self.mem.alloc_constant(1)
        self.CONST_SIGN = self.mem.alloc_constant(-(1 << cfg.N - 1))
        self.T0 = self.mem.alloc_temp('__t0')
        self.T1 = self.mem.alloc_temp('__t1')
        self.T2 = self.mem.alloc_temp('__t2')
        self.T3 = self.mem.alloc_temp('__t3')
        self.RETVAL = self.mem.alloc_temp('__retval')
        self.OUTPUT = self.mem.alloc_temp('__output')
        self.var_types: Dict[str, str] = {}
        self.aligned_arrays: set = set()
        self.ret_types: List[str] = ['int']
        self._fix_level = 0
        self.cur_vl: Optional[int] = None
        self.ret_labels: List[Optional[str]] = [None]
        self.noinline_called: List[str] = []
        self.scratch = {self.T0, self.T1, self.T2, self.T3, self.RETVAL}
        self.spill_depth = 0
        self.spill_ns = ''
        self.spills: Dict[str, set] = {}
        self.prefix_bodies: set = set()
        self.loops: List[Tuple[str, str]] = []

    def clone(self) -> 'CodeGen':
        c = copy.copy(self)
        for name, value in vars(self).items():
            if name == 'cmds':
                value = [list(cmd) for cmd in value]
            elif name == 'spills':
                value = {k: set(v) for k, v in value.items()}
            elif name == 'mem':
                mem = copy.copy(value)
                mem.slots, mem.constants, mem.initial = (dict(value.slots), dict(value.constants), dict(value.initial))
                value = mem
            elif isinstance(value, (dict, list, set)):
                value = type(value)(value)
            setattr(c, name, value)
        return c

    def temp(self, name: str) -> int:
        addr = self.mem.alloc_var(f'__{name}')
        self.scratch.add(addr)
        return addr

    def temp_pair(self, name: str) -> int:
        addr = self.mem.alloc_var(f'__{name}', 2, align=2)
        self.scratch.update((addr, addr + 1))
        return addr
    _NO_WRITE = {OP_JMP, OP_JZ, OP_JNZ, OP_CMP, OP_HALT, OP_STORE, OP_JMPI}

    def _written(self, start: int) -> Optional[set]:
        s, out = (self.cfg.s, set())
        for a, b, c in self.cmds[start:]:
            if a == OP_JMP and isinstance(c, str) and c.startswith('__fn_'):
                return None
            if a in self._NO_WRITE:
                continue
            out.add(b - s)
            if a == OP_SWAP:
                out.add(c - s)
            elif a in (OP_MULSTEP, OP_DIVSTEP):
                out.add(b - s ^ 1)
        return out

    def _delete(self, pos: int) -> None:
        del self.cmds[pos]
        self.labels = {k: v - (v > pos) for k, v in self.labels.items()}
        self.checks = {k - (k > pos): v for k, v in self.checks.items()}
        self.source_map = [(k - (k > pos), line) for k, line in self.source_map]

    def _protect(self, held, compile_next, words=None):
        mine = self.spills.setdefault(self.spill_ns, set())
        cand = [w for w in dict.fromkeys(held) if (w in self.scratch or (words and w in words)) and w not in mine]
        start = len(self.cmds)
        for w in cand:
            self.emit(OP_MOV, None, self.addr(w))
        self.spill_depth += 1
        try:
            result = compile_next()
        finally:
            self.spill_depth -= 1
        written = self._written(start + len(cand)) if cand else set()
        remap, busy = ({}, set(held))
        for k in reversed(range(len(cand))):
            w = cand[k]
            if written is not None and w not in written:
                self._delete(start + k)
                continue
            j = 0
            while self.temp(f'spill[{self.spill_ns}]_{self.spill_depth}_{j}') in busy:
                j += 1
            spill = self.temp(f'spill[{self.spill_ns}]_{self.spill_depth}_{j}')
            mine.add(spill)
            busy.add(spill)
            self.cmds[start + k][1] = self.addr(spill)
            remap[w] = spill
        return (result, [remap.get(w, w) for w in held])

    def compile_operands(self, exprs, dests) -> List[int]:
        out = []
        for e, d in zip(exprs, dests):
            v, out = self._protect(out, lambda e=e, d=d: self.compile_expr(e, d))
            out.append(v)
        return out

    def const(self, value: int) -> int:
        return self.mem.alloc_constant(value)

    def check(self, kind: str, *slots) -> None:
        if self.rational and kind in ('fadd', 'fsub', 'fmul', 'fdiv', 'fdivi', 'fneg', 'fcast'):
            return
        self.checks[len(self.cmds)] = (kind,) + tuple(slots)

    def literal(self, value: int) -> int:
        lo, hi = (-(1 << self.cfg.N - 1), (1 << self.cfg.N - 1) - 1)
        if self.checked and (not lo <= value <= hi):
            raise ExecutionFault('overflow', f'literal {value} does not fit {self.cfg.N} bits')
        return self.mem.alloc_constant(value)

    def new_label(self, prefix: str='L') -> str:
        self.label_counter += 1
        return f'__{prefix}_{self.label_counter}'

    def addr(self, mem_idx: int) -> int:
        return self.cfg.s + mem_idx

    def emit(self, a: int, b: int, c) -> None:
        self.cmds.append([a, b, c])

    def emit_label(self, name: str) -> None:
        self.labels[name] = len(self.cmds)
        self.cur_vl = None

    def emit_mov(self, dest: int, src: int) -> None:
        self.emit(OP_MOV, self.addr(dest), self.addr(src))

    def emit_add(self, dest: int, src: int) -> None:
        self.emit(OP_ADD, self.addr(dest), self.addr(src))

    def emit_sub(self, dest: int, src: int) -> None:
        self.emit(OP_SUB, self.addr(dest), self.addr(src))

    def emit_and(self, dest: int, src: int) -> None:
        self.emit(OP_AND, self.addr(dest), self.addr(src))

    def emit_or(self, dest: int, src: int) -> None:
        self.emit(OP_OR, self.addr(dest), self.addr(src))

    def emit_xor(self, dest: int, src: int) -> None:
        self.emit(OP_XOR, self.addr(dest), self.addr(src))

    def emit_shl(self, dest: int) -> None:
        self.emit(OP_SHL, self.addr(dest), 0)

    def emit_shr(self, dest: int) -> None:
        self.emit(OP_SHR, self.addr(dest), 0)

    def emit_inc(self, dest: int) -> None:
        self.emit(OP_INC, self.addr(dest), 0)

    def emit_dec(self, dest: int) -> None:
        self.emit(OP_DEC, self.addr(dest), 0)

    def emit_jz(self, mem_idx: int, label: str) -> None:
        self.emit(OP_JZ, self.addr(mem_idx), label)

    def emit_jnz(self, mem_idx: int, label: str) -> None:
        self.emit(OP_JNZ, self.addr(mem_idx), label)

    def emit_cmp(self, mem_idx: int, label: str) -> None:
        self.emit(OP_CMP, self.addr(mem_idx), label)

    def emit_cmov(self, dest: int, src: int) -> None:
        self.emit(OP_CMOV, self.addr(dest), self.addr(src))

    def emit_mulacc(self, dest: int, src: int) -> None:
        self.emit(OP_MULACC, self.addr(dest), self.addr(src))

    def emit_swap(self, a: int, b: int) -> None:
        self.emit(OP_SWAP, self.addr(a), self.addr(b))

    def emit_load(self, dest: int, pointer: int) -> None:
        self.emit(OP_LOAD, self.addr(dest), self.addr(pointer))

    def emit_store(self, src: int, pointer: int) -> None:
        self.emit(OP_STORE, self.addr(src), self.addr(pointer))

    def emit_max(self, dest: int, src: int) -> None:
        self.emit(OP_MAX, self.addr(dest), self.addr(src))

    def emit_logadd(self, dest: int, src: int) -> None:
        self.emit(OP_LOGADD, self.addr(dest), self.addr(src))

    def emit_xlat(self, dest: int, table: int) -> None:
        self.emit(OP_XLAT, self.addr(dest), self.addr(table))

    def emit_shift(self, dest: int, amount: int) -> None:
        self.emit(OP_SHIFT, self.addr(dest), self.addr(amount))

    def emit_adc(self, dest: int, src: int) -> None:
        self.emit(OP_ADC, self.addr(dest), self.addr(src))

    def emit_mulstep(self, hi: int, src: int) -> None:
        self.emit(OP_MULSTEP, self.addr(hi), self.addr(src))

    def emit_divstep(self, rem: int, src: int) -> None:
        self.emit(OP_DIVSTEP, self.addr(rem), self.addr(src))

    def emit_lfsr(self, dest: int, taps: int) -> None:
        self.emit(OP_LFSR, self.addr(dest), self.addr(taps))

    def emit_clz(self, dest: int, src: int) -> None:
        self.emit(OP_CLZ, self.addr(dest), self.addr(src))

    def emit_abs(self, dest: int, src: int) -> None:
        self.emit(OP_ABS, self.addr(dest), self.addr(src))

    def emit_argmax(self, dest: int, base: int) -> None:
        self.emit(OP_ARGMAX, self.addr(dest), self.addr(base))

    def emit_jmp(self, label: str) -> None:
        self.emit(OP_JMP, 0, label)

    def emit_halt(self) -> None:
        self.emit(OP_HALT, 0, 0)

    def resolve(self) -> List[Tuple[int, int, int]]:
        cmd_start = self.cfg.s + self.cfg.m
        for cmd in self.cmds:
            if isinstance(cmd[2], str):
                if cmd[2] not in self.labels:
                    raise NameError(f"Undefined label '{cmd[2]}'")
                cmd[2] = cmd_start + self.labels[cmd[2]]
        return [tuple(cmd) for cmd in self.cmds]

    def compile_program(self, prog: Program) -> None:
        for func in prog.functions:
            self.functions[func.name] = func
        vector = set(VECTOR_BUILTINS) | set(VECTOR_REDUCE)
        for node in _walk(prog):
            if isinstance(node, FuncCall) and node.name in {'xlat', 'argmax', 'falign', 'fnorm'} | vector and node.args and isinstance(node.args[0], VarRef):
                self.aligned_arrays.add(node.args[0].name)
                if node.name in VECTOR_BUILTINS and VECTOR_BUILTINS[node.name][1] != 'scalar' and (len(node.args) > 1) and isinstance(node.args[1], VarRef):
                    self.aligned_arrays.add(node.args[1].name)
        if self.prefix_bodies:
            self.emit_label('__main_entry')
        for g in prog.globals:
            self.compile_var_decl(g)
        if 'main' not in self.functions:
            raise SyntaxError('No main() function found')
        main = self.functions['main']
        self.compile_body(main.body, '__main_ret')
        self.emit_halt()
        done = set(self.prefix_bodies)
        while len(done) < len(self.noinline_called):
            name = next((f for f in self.noinline_called if f not in done))
            done.add(name)
            self.compile_noinline_body(self.functions[name])

    def compile_block(self, block: Block) -> None:
        for stmt in block.stmts:
            self.compile_stmt(stmt)

    def compile_body(self, body: Block, label: str, always_label: bool=False) -> None:
        start = len(self.cmds)
        self.ret_labels.append(label)
        loops, self.loops = (self.loops, [])
        self.compile_block(body)
        self.loops = loops
        self.ret_labels.pop()
        while len(self.cmds) > start and self.cmds[-1][0] == OP_JMP and (self.cmds[-1][2] == label):
            self._delete(len(self.cmds) - 1)
        if always_label or any((c[2] == label for c in self.cmds[start:])):
            self.emit_label(label)

    def compile_stmt(self, stmt: Stmt) -> None:
        line = getattr(stmt, 'line', 0)
        if line > 0:
            self.source_map.append((len(self.cmds), line))
        if isinstance(stmt, VarDecl):
            self.compile_var_decl(stmt)
        elif isinstance(stmt, Assignment):
            self.compile_assignment(stmt)
        elif isinstance(stmt, IfStmt):
            self.compile_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self.compile_while(stmt)
        elif isinstance(stmt, ForStmt):
            self.compile_for(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.compile_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self.compile_expr(stmt.expr, self.T0)
        elif isinstance(stmt, JumpStmt):
            if not self.loops:
                raise SyntaxError(f'{stmt.kind} outside a loop')
            self.emit_jmp(self.loops[-1][0 if stmt.kind == 'break' else 1])
        elif isinstance(stmt, Block):
            self.compile_block(stmt)
        else:
            raise SyntaxError(f'Unknown statement type: {type(stmt)}')

    def compile_var_decl(self, decl: VarDecl) -> None:
        if decl.type == 'fix':
            if decl.size is not None:
                raise SyntaxError(f"fix arrays are not supported ('{decl.name}')")
            hi = self.fix_pair(self.mem.alloc_var(decl.name, 2, align=2))
            self.var_types[decl.name] = 'fix'
            if decl.init is not None:
                self.fix_copy((hi, hi + 1), self.compile_fix(decl.init, self.fix_temp('asg')))
            return
        self.var_types[decl.name] = 'int'
        if decl.size is not None:
            align = 1
            if decl.name in self.aligned_arrays:
                align = 1 << (decl.size - 1).bit_length()
            addr = self.mem.alloc_var(decl.name, decl.size, align=align)
            for i, v in enumerate(decl.values or []):
                self.mem.initial[addr + i] = _wrap(v, self.cfg.N)
        else:
            addr = self.mem.alloc_var(decl.name)
            if decl.init is not None:
                src = self.compile_expr(decl.init, self.T0)
                if src != addr:
                    self.emit_mov(addr, src)

    def compile_assignment(self, assign: Assignment) -> None:
        if isinstance(assign.target, VarRef) and self.var_types.get(assign.target.name) == 'fix':
            hi = self.mem.get_var(assign.target.name)
            target = (hi, hi + 1)
            value = self.compile_fix(assign.value, self.fix_temp('asg'))
            if assign.op == '=':
                self.fix_copy(target, value)
            elif assign.op == '+=':
                self.check('fadd', *target, *value)
                self.fix_add(target, value)
            else:
                self.check('fsub', *target, *value)
                self.fix_sub(target, value)
            return
        if isinstance(assign.target, VarRef):
            dest = self.mem.get_var(assign.target.name)
            if assign.op == '=':
                src = self.compile_expr(assign.value, self.T0)
                if src != dest:
                    self.emit_mov(dest, src)
            elif assign.op == '+=':
                src = self.compile_expr(assign.value, self.T0)
                self.check('add', dest, src)
                self.emit_add(dest, src)
            elif assign.op == '-=':
                src = self.compile_expr(assign.value, self.T0)
                self.check('sub', dest, src)
                self.emit_sub(dest, src)
        elif isinstance(assign.target, ArrayRef):
            base = self.mem.get_var(assign.target.name)
            slot = self.mem.slots[assign.target.name]
            if isinstance(assign.target.index, IntLiteral):
                dest = base + assign.target.index.value
                if assign.op == '=':
                    src = self.compile_expr(assign.value, self.T0)
                    if src != dest:
                        self.emit_mov(dest, src)
                elif assign.op == '+=':
                    src = self.compile_expr(assign.value, self.T0)
                    self.check('add', dest, src)
                    self.emit_add(dest, src)
                elif assign.op == '-=':
                    src = self.compile_expr(assign.value, self.T0)
                    self.check('sub', dest, src)
                    self.emit_sub(dest, src)
            else:
                self._check_pointer_range(assign.target.name)
                idx_addr = self.compile_expr(assign.target.index, self.T1)
                base_const = self.mem.alloc_constant(base)
                self.emit_mov(self.T2, base_const)
                self.emit_add(self.T2, idx_addr)
                size = self.mem.slots[assign.target.name].size
                src, (ptr,) = self._protect([self.T2], lambda: self.compile_expr(assign.value, self.T0))
                self.check('bounds', ptr, base, size)
                if assign.op == '=':
                    self.emit_store(src, ptr)
                else:
                    self.emit_load(self.T3, ptr)
                    self.check('add' if assign.op == '+=' else 'sub', self.T3, src)
                    (self.emit_add if assign.op == '+=' else self.emit_sub)(self.T3, src)
                    self.emit_store(self.T3, ptr)
        else:
            raise SyntaxError('Invalid assignment target')

    def compile_if(self, stmt: IfStmt) -> None:
        else_label = self.new_label('else')
        end_label = self.new_label('endif')
        self.compile_condition(stmt.cond, else_label, invert=True)
        self.compile_block(stmt.then_body)
        if stmt.else_body:
            self.emit_jmp(end_label)
        self.emit_label(else_label)
        if stmt.else_body:
            self.compile_block(stmt.else_body)
            self.emit_label(end_label)

    def compile_while(self, stmt: WhileStmt) -> None:
        loop_label = self.new_label('while')
        end_label = self.new_label('endwhile')
        self.emit_label(loop_label)
        self.compile_condition(stmt.cond, end_label, invert=True)
        self.loops.append((end_label, loop_label))
        self.compile_block(stmt.body)
        self.loops.pop()
        self.emit_jmp(loop_label)
        self.emit_label(end_label)

    def compile_for(self, stmt: ForStmt) -> None:
        if stmt.init:
            self.compile_stmt(stmt.init)
        loop_label = self.new_label('for')
        end_label = self.new_label('endfor')
        self.emit_label(loop_label)
        if stmt.cond:
            self.compile_condition(stmt.cond, end_label, invert=True)
        next_label = self.new_label('fornext')
        self.loops.append((end_label, next_label))
        self.compile_block(stmt.body)
        self.loops.pop()
        if any((c[2] == next_label for c in self.cmds)):
            self.emit_label(next_label)
        if stmt.update:
            self.compile_stmt(stmt.update)
        self.emit_jmp(loop_label)
        self.emit_label(end_label)

    def compile_return(self, stmt: ReturnStmt) -> None:
        if stmt.value and self.ret_types[-1] == 'fix':
            hi = self.fix_pair(self.temp_pair('retval_fix'))
            self.fix_copy((hi, hi + 1), self.compile_fix(stmt.value, self.fix_temp('ret')))
        elif stmt.value:
            src = self.compile_expr(stmt.value, self.T0)
            if src != self.RETVAL:
                self.emit_mov(self.RETVAL, src)
        if self.ret_labels[-1] is not None:
            self.emit_jmp(self.ret_labels[-1])

    def compile_condition(self, expr: Expr, false_label: str, invert: bool=False) -> None:
        if isinstance(expr, IntLiteral):
            if (expr.value != 0) != invert:
                self.emit_jmp(false_label)
            return
        if isinstance(expr, BinaryOp):
            if expr.op in ('==', '!=', '<', '>', '<=', '>=') and self.is_fix_compare(expr):
                flag = self.fix_compare(expr, self.T2)
                if invert:
                    self.emit_jz(flag, false_label)
                else:
                    self.emit_jnz(flag, false_label)
                return
            if expr.op in ('==', '!=', '<', '>', '<=', '>='):
                left, right = self.compile_operands([expr.left, expr.right], [self.T0, self.T1])
                self._emit_comparison(expr.op, left, right, false_label, invert)
                return
            if expr.op == '&&':
                if invert:
                    self.compile_condition(expr.left, false_label, invert=True)
                    self.compile_condition(expr.right, false_label, invert=True)
                else:
                    skip = self.new_label('and_skip')
                    self.compile_condition(expr.left, skip, invert=True)
                    self.compile_condition(expr.right, false_label, invert=False)
                    self.emit_label(skip)
                return
            if expr.op == '||':
                if invert:
                    skip = self.new_label('or_skip')
                    self.compile_condition(expr.left, skip, invert=False)
                    self.compile_condition(expr.right, false_label, invert=True)
                    self.emit_label(skip)
                else:
                    self.compile_condition(expr.left, false_label, invert=False)
                    self.compile_condition(expr.right, false_label, invert=False)
                return
        if isinstance(expr, UnaryOp) and expr.op == '!':
            self.compile_condition(expr.expr, false_label, invert=not invert)
            return
        if self.type_of(expr) == 'fix':
            result = self.fix_nonzero(expr, self.T0)
        else:
            result = self.compile_expr(expr, self.T0)
        if invert:
            self.emit_jz(result, false_label)
        else:
            self.emit_jnz(result, false_label)

    def _emit_comparison(self, op: str, left: int, right: int, false_label: str, invert: bool) -> None:
        if op == '>':
            op = '<'
            left, right = (right, left)
        elif op == '<=':
            op = '>='
            left, right = (right, left)
        if op in ('<', '>='):
            self.emit_mov(self.T2, left)
            self.emit_max(self.T2, right)
            self.emit_sub(self.T2, left)
            jump_if_ge = (op == '>=') != invert
            (self.emit_jz if jump_if_ge else self.emit_jnz)(self.T2, false_label)
            return
        self.emit_mov(self.T2, left)
        self.emit_sub(self.T2, right)
        if op == '==' and invert:
            self.emit_jnz(self.T2, false_label)
        elif op == '==' and (not invert):
            self.emit_jz(self.T2, false_label)
        elif op == '!=' and invert:
            self.emit_jz(self.T2, false_label)
        elif op == '!=' and (not invert):
            self.emit_jnz(self.T2, false_label)

    def compile_expr(self, expr: Expr, dest: int) -> int:
        if self.type_of(expr) == 'fix':
            hi, lo = self.compile_fix(expr, self.fix_temp(f'cvt{self._fix_level}'))
            if self.rational:
                self.checks[len(self.cmds)] = ('rcast', hi, lo)
                self.rat_call('floor', hi, lo)
                self.emit_mov(dest, self.RETVAL)
                return dest
            self.check('fcast', hi, lo)
            if hi != dest:
                self.emit_mov(dest, hi)
            return dest
        if isinstance(expr, IntLiteral):
            return self.literal(expr.value)
        if isinstance(expr, Slot):
            return expr.addr
        if isinstance(expr, VarRef):
            return self.mem.get_var(expr.name)
        if isinstance(expr, ArrayRef):
            base = self.mem.get_var(expr.name)
            if isinstance(expr.index, IntLiteral):
                return base + expr.index.value
            self._check_pointer_range(expr.name)
            idx_addr = self.compile_expr(expr.index, self.T1)
            base_const = self.mem.alloc_constant(base)
            self.emit_mov(self.T2, base_const)
            self.emit_add(self.T2, idx_addr)
            self.check('bounds', self.T2, base, self.mem.slots[expr.name].size)
            self.emit_load(dest, self.T2)
            return dest
        if isinstance(expr, UnaryOp):
            if expr.op in ('(int)', '(fix)'):
                return self.compile_expr(expr.expr, dest)
            if expr.op == '-':
                src = self.compile_expr(expr.expr, dest)
                if src == dest:
                    src = self.temp('neg')
                    self.emit_mov(src, dest)
                self.emit_mov(dest, self.CONST_ZERO)
                self.check('sub', dest, src)
                self.emit_sub(dest, src)
                return dest
            if expr.op == '~':
                src = self.compile_expr(expr.expr, dest)
                if src != dest:
                    self.emit_mov(dest, src)
                all_ones = self.mem.alloc_constant(-1)
                self.emit_xor(dest, all_ones)
                return dest
            if expr.op == '!':
                if self.type_of(expr.expr) == 'fix':
                    src = self.fix_nonzero(expr.expr, dest)
                else:
                    src = self.compile_expr(expr.expr, dest)
                nonzero_label = self.new_label('not_nz')
                end_label = self.new_label('not_end')
                self.emit_jnz(src, nonzero_label)
                self.emit_mov(dest, self.CONST_ONE)
                self.emit_jmp(end_label)
                self.emit_label(nonzero_label)
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_label(end_label)
                return dest
        if isinstance(expr, BinaryOp):
            if isinstance(expr.left, IntLiteral) and isinstance(expr.right, IntLiteral):
                a, b = (expr.left.value, expr.right.value)
                fold_ops = {'+': lambda: a + b, '-': lambda: a - b, '*': lambda: a * b, '&': lambda: a & b, '|': lambda: a | b, '^': lambda: a ^ b, '<<': lambda: a << b, '>>': lambda: a >> b, '==': lambda: int(a == b), '!=': lambda: int(a != b), '<': lambda: int(a < b), '>': lambda: int(a > b), '<=': lambda: int(a <= b), '>=': lambda: int(a >= b)}
                if b != 0:
                    q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
                    fold_ops['/'] = lambda: q
                    fold_ops['%'] = lambda: a - b * q
                if expr.op in fold_ops:
                    return self.compile_expr(IntLiteral(fold_ops[expr.op]()), dest)
            if expr.op in ('==', '!=', '<', '>', '<=', '>=') and self.is_fix_compare(expr):
                return self.fix_compare(expr, dest)
            if expr.op in ('==', '!=', '<', '>', '<=', '>='):
                true_label = self.new_label('cmp_true')
                end_label = self.new_label('cmp_end')
                left, right = self.compile_operands([expr.left, expr.right], [self.T0, self.T1])
                self._emit_comparison(expr.op, left, right, true_label, invert=False)
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_jmp(end_label)
                self.emit_label(true_label)
                self.emit_mov(dest, self.CONST_ONE)
                self.emit_label(end_label)
                return dest
            if expr.op in ('&&', '||'):
                true_label = self.new_label('log_true')
                end_label = self.new_label('log_end')
                self.compile_condition(expr, true_label, invert=False)
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_jmp(end_label)
                self.emit_label(true_label)
                self.emit_mov(dest, self.CONST_ONE)
                self.emit_label(end_label)
                return dest
            right_temp = self.T1 if dest != self.T1 else self.T3
            left, right = self.compile_operands([expr.left, expr.right], [dest, right_temp])
            if left != dest:
                self.emit_mov(dest, left)
            op_map = {'+': self.emit_add, '-': self.emit_sub, '&': self.emit_and, '|': self.emit_or, '^': self.emit_xor}
            if expr.op in op_map:
                if expr.op in ('+', '-'):
                    self.check('add' if expr.op == '+' else 'sub', dest, right)
                op_map[expr.op](dest, right)
                return dest
            if expr.op == '*':
                self.check('mul', dest, right)
                hi = self.emit_umul(dest, right)
                self.emit_mov(dest, hi + 1)
                return dest
            if expr.op in ('/', '%'):
                self.check('div' if expr.op == '/' else 'mod', dest, right)
                self.emit_divide(dest, dest, right, want_rem=expr.op == '%')
                return dest
            if expr.op in ('<<', '>>'):
                self.check('shl' if expr.op == '<<' else 'shr', dest, right)
                if isinstance(expr.right, IntLiteral):
                    k = expr.right.value
                    if k == 1:
                        (self.emit_shl if expr.op == '<<' else self.emit_shr)(dest)
                    elif k:
                        self.emit_shift(dest, self.const(k if expr.op == '<<' else -k))
                elif expr.op == '<<':
                    self.emit_shift(dest, right)
                else:
                    amount = self.temp('shift_amount')
                    self.emit_mov(amount, self.CONST_ZERO)
                    self.emit_sub(amount, right)
                    self.emit_shift(dest, amount)
                return dest
            raise SyntaxError(f'Unsupported binary operator: {expr.op}')
        if isinstance(expr, FuncCall):
            return self.compile_func_call(expr, dest)
        raise SyntaxError(f'Unknown expression type: {type(expr)}')

    def compile_func_call(self, call: FuncCall, dest: int) -> int:

        def nargs(n):
            if len(call.args) != n:
                raise SyntaxError(f"{call.name}() takes exactly {n} argument{('s' if n != 1 else '')}")
        if call.name == 'printf' or call.name == 'output':
            if call.args:
                src = self.compile_expr(call.args[0], self.T0)
                self.emit_mov(self.OUTPUT, src)
            return self.OUTPUT
        if call.name == 'ipowi':
            nargs(2)
            if '__rat_ipow' not in self.functions:
                raise SyntaxError("ipowi() needs numbers='rational'")
            if 'fix' in (self.type_of(call.args[0]), self.type_of(call.args[1])):
                raise SyntaxError('ipowi() takes ints')
            self.inline_call(FuncCall('__rat_ipow', list(call.args)))
            return self.RETVAL
        if call.name == 'isqrt':
            nargs(1)
            if '__rat_isqrt' not in self.functions:
                raise SyntaxError("isqrt() needs numbers='rational'")
            if self.type_of(call.args[0]) == 'fix':
                raise SyntaxError('isqrt() takes an int')
            self.inline_call(FuncCall('__rat_isqrt', [call.args[0]]))
            return self.RETVAL
        if call.name == 'abs':
            nargs(1)
            src = self.compile_expr(call.args[0], dest)
            self.check('abs', src)
            self.emit_abs(dest, src)
            return dest
        if call.name in ('min', 'max'):
            nargs(2)
            a, b = self.compile_operands(call.args, [dest, self.T1 if dest != self.T1 else self.T3])
            if a != dest:
                self.emit_mov(dest, a)
            if call.name == 'max':
                self.emit_max(dest, b)
            else:
                t = self.temp('minmax')
                self.emit_mov(t, dest)
                self.emit_max(t, b)
                self.emit_add(dest, b)
                self.emit_sub(dest, t)
            return dest
        if call.name == 'mul':
            nargs(2)
            return self.compile_expr(BinaryOp('*', call.args[0], call.args[1]), dest)
        if call.name == 'swap':
            nargs(2)
            a, b = self.compile_operands(call.args, [dest, self.T1 if dest != self.T1 else self.T3])
            self.emit_swap(a, b)
            return dest
        if call.name in ('logadd', 'lfsr'):
            nargs(2)
            a, b = self.compile_operands(call.args, [dest, self.T1 if dest != self.T1 else self.T3])
            if a != dest:
                self.emit_mov(dest, a)
            (self.emit_logadd if call.name == 'logadd' else self.emit_lfsr)(dest, b)
            return dest
        if call.name == 'clz':
            nargs(1)
            src = self.compile_expr(call.args[0], dest)
            self.emit_clz(dest, src)
            return dest
        if call.name == 'xlat':
            nargs(2)
            base = self._aligned_array(call)
            idx = self.compile_expr(call.args[1], dest)
            if idx != dest:
                self.emit_mov(dest, idx)
            self.emit_xlat(dest, base)
            return dest
        if call.name == 'argmax':
            if len(call.args) not in (1, 2):
                raise SyntaxError('argmax() takes 1 or 2 arguments')
            base = self._aligned_array(call)
            if len(call.args) == 1:
                size = self.mem.slots[call.args[0].name].size
                if size & size - 1:
                    raise SyntaxError('argmax(arr) needs a power-of-two array; use argmax(arr, log2n)')
                self.emit_mov(dest, self.const(size.bit_length() - 1))
            else:
                k = self.compile_expr(call.args[1], dest)
                if k != dest:
                    self.emit_mov(dest, k)
            self.emit_argmax(dest, base)
            return dest
        if call.name == 'umulhi':
            nargs(2)
            a, b = self.compile_operands(call.args, [dest, self.T1 if dest != self.T1 else self.T3])
            hi = self.emit_umul(a, b)
            self.emit_mov(dest, hi)
            return dest
        if call.name == 'qdiv':
            nargs(2)
            a, b = self.compile_operands(call.args, [dest, self.T1 if dest != self.T1 else self.T3])
            rem = self.temp_pair('div')
            self.emit_mov(rem, a)
            self.emit_mov(rem + 1, self.CONST_ZERO)
            for _ in range(self.cfg.N):
                self.emit_divstep(rem, b)
            self.emit_mov(dest, rem + 1)
            return dest
        if call.name == 'mulmod':
            nargs(3)
            ops = self.compile_operands(call.args, [self.temp(f'mulmod_{k}') for k in range(3)])
            hi = self.emit_umul(ops[0], ops[1])
            rem = self.temp_pair('div')
            self.emit_mov(rem + 1, hi)
            self.emit_mov(rem, self.CONST_ZERO)
            for _ in range(self.cfg.N):
                self.emit_divstep(rem, ops[2])
            self.emit_mov(rem + 1, hi + 1)
            for _ in range(self.cfg.N):
                self.emit_divstep(rem, ops[2])
            self.emit_mov(dest, rem)
            return dest
        if call.name in VECTOR_BUILTINS:
            nargs(2)
            opcode, kind = VECTOR_BUILTINS[call.name]
            base, size = self._vector_array(call.args[0], call.name)
            self._set_vl(size)
            if kind == 'scalar':
                v = self.compile_expr(call.args[1], dest)
                self.emit(opcode, self.addr(base), self.addr(v))
            else:
                other, osize = self._vector_array(call.args[1], call.name)
                if kind == 'array' and osize != size:
                    raise SyntaxError(f'{call.name}(): arrays differ in size ({size} vs {osize})')
                self.emit(opcode, self.addr(base), self.addr(other))
            return self.CONST_ZERO
        if call.name in VECTOR_REDUCE:
            nargs(1)
            base, size = self._vector_array(call.args[0], call.name)
            self._set_vl(size)
            stride = 1
            while stride < size:
                self.emit(VECTOR_REDUCE[call.name], self.addr(base), self.addr(self.const(stride)))
                stride *= 2
            if call.name == 'wht':
                return self.CONST_ZERO
            self.emit_mov(dest, base)
            return dest
        if call.name == 'diff':
            raise SyntaxError('diff() is fix-valued; use it in a fix expression')
        if call.name in ('falign', 'fnorm'):
            nargs(2 if call.name == 'falign' else 1)
            if not self.cfg.page2:
                raise SyntaxError(f'{call.name}() needs page 2 (n >= 128)')
            base, size = self._vector_array(call.args[0], call.name)
            if size != 2:
                raise SyntaxError(f'{call.name}() takes an int[2] float')
            if call.name == 'falign':
                e = self.compile_expr(call.args[1], dest)
                self.emit(OP_FALIGN, self.addr(base), self.addr(e))
            else:
                self.emit(OP_FNORM, self.addr(base), 0)
            return self.CONST_ZERO
        self.inline_call(call)
        return self.RETVAL

    def _vector_array(self, arg: Expr, name: str) -> Tuple[int, int]:
        if not isinstance(arg, VarRef) or arg.name not in self.mem.slots or self.mem.slots[arg.name].size < 2:
            raise SyntaxError(f'{name}() needs array arguments')
        slot = self.mem.slots[arg.name]
        if slot.size & slot.size - 1:
            raise SyntaxError(f"{name}(): '{arg.name}' must have a power-of-two size")
        return (slot.addr, slot.size)

    def _set_vl(self, size: int) -> None:
        if not self.cfg.vector:
            raise SyntaxError('vector builtins need the vector unit (n >= 128)')
        t = size.bit_length() - 1
        if self.cur_vl != t:
            self.emit(OP_VSETVL, self.addr(self.const(t)), 0)
            self.cur_vl = t

    def _check_pointer_range(self, name: str) -> None:
        slot = self.mem.slots[name]
        if slot.addr + slot.size > self.cfg.s:
            raise MemoryError(f"array '{name}' (words {slot.addr}..{slot.addr + slot.size - 1}) is indexed by a variable but LOAD/STORE only reach the first s={self.cfg.s} memory words; declare it earlier or use a larger s")

    def _aligned_array(self, call: FuncCall) -> int:
        arg = call.args[0]
        if not isinstance(arg, VarRef) or arg.name not in self.mem.slots or self.mem.slots[arg.name].size < 2:
            raise SyntaxError(f'{call.name}() needs an array as its first argument')
        return self.mem.get_var(arg.name)

    def inline_call(self, call: FuncCall) -> None:
        if call.name not in self.functions:
            raise NameError(f"Undefined function '{call.name}'")
        func = self.functions[call.name]
        if len(call.args) != len(func.params):
            raise SyntaxError(f"Function '{call.name}' expects {len(func.params)} args, got {len(call.args)}")
        types = func.param_types or ['int'] * len(func.params)
        if func.noinline:
            self.call_noinline(call, func, types)
            return
        self._bind_args(call, func, types)
        saved = self._map_params(func, types)
        self.ret_types.append(func.ret_type)
        self.compile_body(func.body, self.new_label(f'ret_{func.name}'))
        self.ret_types.pop()
        self._unmap_params(func, saved)

    def _bind_args(self, call: FuncCall, func: FuncDecl, types: List[str]) -> None:
        bound: List[int] = []
        for param, ptype, arg in zip(func.params, types, call.args):
            internal = f'__{func.name}_{param}'

            def bind():
                if ptype == 'fix':
                    hi = self.fix_pair(self.mem.alloc_var(internal, 2, align=2))
                    self.fix_copy((hi, hi + 1), self.compile_fix(arg, self.fix_temp('arg')))
                    return [hi, hi + 1]
                src = self.compile_expr(arg, self.T0)
                param_addr = self.mem.alloc_var(internal)
                if src != param_addr:
                    self.emit_mov(param_addr, src)
                return [param_addr]
            words, now = self._protect(bound, bind, words=set(bound))
            for w, v in zip(bound, now):
                if v != w:
                    self.emit_mov(w, v)
            bound += words

    def _map_params(self, func: FuncDecl, types: List[str]) -> Dict:
        saved = {}
        for param, ptype in zip(func.params, types):
            internal = f'__{func.name}_{param}'
            if internal not in self.mem.slots:
                addr = self.mem.alloc_var(internal, 2 if ptype == 'fix' else 1, align=2 if ptype == 'fix' else 1)
                if ptype == 'fix':
                    self.fix_pair(addr)
            saved[param] = (self.mem.slots.get(param), self.var_types.get(param))
            self.mem.slots[param] = self.mem.slots[internal]
            self.var_types[param] = ptype
        return saved

    def _unmap_params(self, func: FuncDecl, saved: Dict) -> None:
        for param in func.params:
            slot, vtype = saved[param]
            if slot is not None:
                self.mem.slots[param] = slot
                self.var_types[param] = vtype
            else:
                del self.mem.slots[param]
                self.var_types.pop(param, None)

    def call_noinline(self, call: FuncCall, func: FuncDecl, types: List[str]) -> None:
        if not self.cfg.page2:
            raise SyntaxError('noinline functions need JMPI (n >= 128)')
        if f'__fnret_{func.name}' in self.ret_labels:
            raise SyntaxError(f"'{func.name}' calls itself; noinline functions are not reentrant")
        self._bind_args(call, func, types)
        back = len(self.cmds) + 2
        if back >= 1 << self.cfg.N:
            raise SyntaxError(f"call to '{func.name}' at instruction {back}: JMPI reaches only the first 2^N = {1 << self.cfg.N} instructions")
        self.emit_mov(self.temp(f'ret_{func.name}'), self.const(back))
        self.emit_jmp(f'__fn_{func.name}')
        self.cur_vl = None
        if func.name not in self.noinline_called:
            self.noinline_called.append(func.name)

    def compile_noinline_body(self, func: FuncDecl) -> None:
        types = func.param_types or ['int'] * len(func.params)
        self.emit_label(f'__fn_{func.name}')
        saved = self._map_params(func, types)
        end = f'__fnret_{func.name}'
        self.ret_types.append(func.ret_type)
        self.spill_ns = func.name
        self.compile_body(func.body, end, always_label=True)
        self.spill_ns = ''
        self.ret_types.pop()
        self._unmap_params(func, saved)
        self.emit(OP_JMPI, 0, self.addr(self.temp(f'ret_{func.name}')))

    def emit_umul(self, a: int, b: int, hi: Optional[int]=None) -> int:
        if hi is None:
            hi = self.temp_pair('mul')
        self.emit_mov(hi + 1, b)
        self.emit_mov(hi, self.CONST_ZERO)
        for _ in range(self.cfg.N):
            self.emit_mulstep(hi, a)
        return hi

    def emit_sign_mask(self, dest: int, src: int) -> None:
        self.emit_mov(dest, src)
        self.emit_shift(dest, self.const(-(self.cfg.N - 1)))

    def emit_divide(self, dest: int, a: int, b: int, want_rem: bool) -> None:
        sign = self.temp('div_sign')
        ua = self.temp('div_ua')
        ub = self.temp('div_ub')
        self.emit_mov(sign, a)
        if not want_rem:
            self.emit_xor(sign, b)
        self.emit_shift(sign, self.const(-(self.cfg.N - 1)))
        self.emit_abs(ua, a)
        self.emit_abs(ub, b)
        rem = self.temp_pair('div')
        self.emit_mov(rem + 1, ua)
        self.emit_mov(rem, self.CONST_ZERO)
        for _ in range(self.cfg.N):
            self.emit_divstep(rem, ub)
        self.emit_mov(dest, rem if want_rem else rem + 1)
        self.emit_xor(dest, sign)
        self.emit_sub(dest, sign)

    def type_of(self, expr: Expr) -> str:
        if isinstance(expr, FixLiteral):
            return 'fix'
        if isinstance(expr, VarRef):
            return self.var_types.get(expr.name, 'int')
        if isinstance(expr, UnaryOp):
            if expr.op == '(fix)':
                return 'fix'
            if expr.op == '-':
                return self.type_of(expr.expr)
            return 'int'
        if isinstance(expr, BinaryOp) and expr.op in ('+', '-', '*', '/'):
            if 'fix' in (self.type_of(expr.left), self.type_of(expr.right)):
                return 'fix'
            return 'int'
        if isinstance(expr, FuncCall) and expr.name in ('diff', 'fsqrt', 'froot', 'fpowi'):
            return 'fix'
        if isinstance(expr, FuncCall) and expr.name not in BUILTINS and (expr.name in self.functions):
            return self.functions[expr.name].ret_type
        return 'int'

    def fix_temp(self, name) -> Tuple[int, int]:
        hi = self.fix_pair(self.temp_pair(f'fix_{name}'))
        return (hi, hi + 1)

    def fix_pair(self, hi: int) -> int:
        if self.rational:
            self.mem.initial[hi + 1] = 1
        return hi

    def fix_const(self, value: float) -> Tuple[int, int]:
        N = self.cfg.N
        if self.rational:
            q = Fraction(repr(value))
            if max(abs(q.numerator), q.denominator) >= 1 << N - 1:
                raise ExecutionFault('overflow', f'fix literal {value} needs more than {N} bits as a fraction')
            return (self.const(q.numerator), self.const(q.denominator))
        raw = math.floor(value * (1 << N) + 0.5)
        if not -(1 << 2 * N - 1) <= raw < 1 << 2 * N - 1:
            raise ExecutionFault('overflow', f'fix literal {value} is out of range for N={N}')
        if raw != value * (1 << N):
            self.fix_literal_rounding += 1
        return (self.const(raw >> N), self.const(raw & (1 << N) - 1))

    def compile_fix(self, expr: Expr, dest: Tuple[int, int]) -> Tuple[int, int]:
        self._fix_level += 1
        try:
            return self._compile_fix(expr, dest)
        finally:
            self._fix_level -= 1

    def _compile_fix(self, expr: Expr, dest: Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(expr, FixLiteral):
            return self.fix_const(expr.value)
        if self.type_of(expr) == 'int':
            return (self.compile_expr(expr, dest[0]), self.CONST_ONE if self.rational else self.CONST_ZERO)
        if isinstance(expr, VarRef):
            hi = self.mem.get_var(expr.name)
            return (hi, hi + 1)
        if isinstance(expr, UnaryOp):
            if expr.op == '(fix)':
                return self._compile_fix(expr.expr, dest)
            self.fix_copy(dest, self.compile_fix(expr.expr, dest))
            self.check('fneg', *dest)
            self.fix_negate(dest)
            return dest
        if isinstance(expr, FuncCall) and expr.name == 'diff':
            if len(expr.args) != 2 or not isinstance(expr.args[1], VarRef):
                raise SyntaxError('diff(expr, variable)')
            return self._compile_fix(self.derivative(expr.args[0], expr.args[1].name), dest)
        if isinstance(expr, FuncCall) and expr.name == 'fpowi':
            if '__rat_powi' not in self.functions:
                raise SyntaxError("fpowi() needs numbers='rational'")
            if len(expr.args) != 2 or self.type_of(expr.args[1]) == 'fix':
                raise SyntaxError('fpowi(x, n) needs an int n')
            self.fix_copy(dest, self.compile_fix(expr.args[0], dest))
            k, kept = self._protect(list(dest), lambda: self.compile_expr(expr.args[1], self.temp('fix_exponent')))
            self.fix_copy(dest, tuple(kept))
            self.rat_call('powi', dest[0], dest[1], k)
            self.emit_mov(dest[0], self.mem.get_var('__rat_n'))
            self.emit_mov(dest[1], self.mem.get_var('__rat_d'))
            return dest
        if isinstance(expr, FuncCall) and expr.name in ('fsqrt', 'froot'):
            if '__rat_root' not in self.functions:
                raise SyntaxError(f"{expr.name}() needs numbers='rational'")
            k = 2
            if expr.name == 'froot':
                if len(expr.args) != 2 or not isinstance(expr.args[1], IntLiteral) or (not 2 <= expr.args[1].value <= 64):
                    raise SyntaxError('froot(x, k) needs a constant k in 2..64')
                k = expr.args[1].value
            elif len(expr.args) != 1:
                raise SyntaxError('fsqrt() takes exactly 1 argument')
            src = self.compile_fix(expr.args[0], dest)
            self.rat_call('root', src[0], src[1], self.const(k))
            self.emit_mov(dest[0], self.mem.get_var('__rat_n'))
            self.emit_mov(dest[1], self.mem.get_var('__rat_d'))
            return dest
        if isinstance(expr, FuncCall):
            self.inline_call(expr)
            hi = self.fix_pair(self.temp_pair('retval_fix'))
            return (hi, hi + 1)
        if isinstance(expr, BinaryOp):
            self.fix_copy(dest, self.compile_fix(expr.left, dest))
            scratch = self.fix_temp(f'r{self._fix_level}')
            if expr.op == '/' and self.type_of(expr.right) == 'int':
                k, kept = self._protect(list(dest), lambda: self.compile_expr(expr.right, self.temp('fix_divisor')))
                self.fix_copy(dest, tuple(kept))
                self.check('fdivi', *dest, k)
                self.fix_div_int(dest, k)
                return dest
            right, kept = self._protect(list(dest), lambda: self.compile_fix(expr.right, scratch))
            self.fix_copy(dest, tuple(kept))
            if expr.op not in ('+', '-', '*', '/'):
                raise SyntaxError(f"Operator '{expr.op}' is not defined for fix")
            self.check({'+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv'}[expr.op], *dest, *right)
            if expr.op == '+':
                self.fix_add(dest, right)
            elif expr.op == '-':
                self.fix_sub(dest, right)
            elif expr.op == '*':
                self.fix_mul(dest, right)
            elif expr.op == '/':
                self.fix_div(dest, right)
            else:
                raise SyntaxError(f"Operator '{expr.op}' is not defined for fix")
            return dest
        raise SyntaxError(f'Cannot use {type(expr).__name__} as fix')

    def derivative(self, e: Expr, x: str) -> Expr:
        zero, one = (IntLiteral(0), IntLiteral(1))
        is0 = lambda t: isinstance(t, IntLiteral) and t.value == 0
        is1 = lambda t: isinstance(t, IntLiteral) and t.value == 1

        def add(a, b):
            return b if is0(a) else a if is0(b) else BinaryOp('+', a, b)

        def sub_(a, b):
            return a if is0(b) else UnaryOp('-', b) if is0(a) else BinaryOp('-', a, b)

        def mul(a, b):
            if is0(a) or is0(b):
                return zero
            return b if is1(a) else a if is1(b) else BinaryOp('*', a, b)
        if isinstance(e, (IntLiteral, FixLiteral, ArrayRef)):
            return zero
        if isinstance(e, VarRef):
            return one if e.name == x else zero
        if isinstance(e, UnaryOp):
            if e.op == '-':
                d = self.derivative(e.expr, x)
                return zero if is0(d) else UnaryOp('-', d)
            if e.op == '(fix)':
                return self.derivative(e.expr, x)
            if e.op == '(int)':
                return zero
        if isinstance(e, BinaryOp) and e.op in ('+', '-', '*', '/'):
            a, b = (e.left, e.right)
            da, db = (self.derivative(a, x), self.derivative(b, x))
            if e.op == '+':
                return add(da, db)
            if e.op == '-':
                return sub_(da, db)
            if e.op == '*':
                return add(mul(da, b), mul(a, db))
            if is0(db):
                return zero if is0(da) else BinaryOp('/', UnaryOp('(fix)', da), b)
            return BinaryOp('/', UnaryOp('(fix)', sub_(mul(da, b), mul(a, db))), BinaryOp('*', b, b))
        if isinstance(e, FuncCall) and e.name in self.functions and (e.name not in BUILTINS):
            f = self.functions[e.name]
            stmts = f.body.stmts
            if len(stmts) != 1 or not isinstance(stmts[0], ReturnStmt) or stmts[0].value is None:
                raise SyntaxError(f"diff(): '{e.name}' must be a single return statement")
            return self.derivative(_substitute(stmts[0].value, dict(zip(f.params, e.args))), x)
        if not any((isinstance(n, VarRef) and n.name == x for n in _walk(e))):
            return zero
        raise SyntaxError(f"diff(): cannot differentiate {type(e).__name__} {getattr(e, 'op', getattr(e, 'name', ''))}")

    def fix_copy(self, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        if dest[0] != src[0]:
            self.emit_mov(dest[0], src[0])
        if dest[1] != src[1]:
            self.emit_mov(dest[1], src[1])

    def rat_call(self, name: str, *slots: int) -> None:
        self.inline_call(FuncCall(f'__rat_{name}', [Slot(a) for a in slots]))

    def rat_op(self, name: str, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        self.rat_call(name, dest[0], dest[1], src[0], src[1])
        self.emit_mov(dest[0], self.mem.get_var('__rat_n'))
        self.emit_mov(dest[1], self.mem.get_var('__rat_d'))

    def fix_add(self, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        if self.rational:
            return self.rat_op('add', dest, src)
        self.emit_add(dest[1], src[1])
        self.emit_adc(dest[0], src[0])

    def fix_sub(self, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        if self.rational:
            return self.rat_op('sub', dest, src)
        t = self.temp('fix_t')
        self.emit_sub(dest[1], src[1])
        self.emit_mov(t, src[0])
        self.emit_xor(t, self.const(-1))
        self.emit_adc(dest[0], t)

    def fix_negate(self, dest: Tuple[int, int], mask: Optional[int]=None) -> None:
        if self.rational and mask is None:
            t = self.temp('fix_t')
            self.emit_mov(t, self.CONST_ZERO)
            self.check('sub', t, dest[0])
            self.emit_sub(t, dest[0])
            self.emit_mov(dest[0], t)
            return
        if mask is None:
            mask = self.const(-1)
            one = self.CONST_ONE
        else:
            one = self.temp('fix_one')
            self.emit_mov(one, self.CONST_ZERO)
            self.emit_sub(one, mask)
        self.emit_xor(dest[0], mask)
        self.emit_xor(dest[1], mask)
        self.emit_add(dest[1], one)
        self.emit_adc(dest[0], self.CONST_ZERO)

    def fix_abs(self, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        mask = self.temp('fix_mask')
        self.fix_copy(dest, src)
        self.emit_sign_mask(mask, dest[0])
        self.fix_negate(dest, mask)

    def _fix_sign(self, a: int, b: int) -> int:
        sign = self.temp('fix_sign')
        self.emit_mov(sign, a)
        self.emit_xor(sign, b)
        self.emit_shift(sign, self.const(-(self.cfg.N - 1)))
        return sign

    def fix_mul(self, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        if self.rational:
            return self.rat_op('mul', dest, src)
        sign = self._fix_sign(dest[0], src[0])
        x = self.fix_temp('a')
        y = self.fix_temp('b')
        self.fix_abs(x, dest)
        self.fix_abs(y, src)
        p = self.temp_pair('mul')
        self.emit_umul(x[1], y[1], p)
        self.emit_mov(dest[1], p)
        self.emit_umul(x[0], y[0], p)
        self.emit_mov(dest[0], p + 1)
        self.emit_umul(x[0], y[1], p)
        self.fix_add(dest, (p, p + 1))
        self.emit_umul(x[1], y[0], p)
        self.fix_add(dest, (p, p + 1))
        self.fix_negate(dest, sign)

    def fix_div_int(self, dest: Tuple[int, int], k: int) -> None:
        if self.rational:
            return self.rat_op('div', dest, (k, self.CONST_ONE))
        sign = self._fix_sign(dest[0], k)
        uk = self.temp('fix_uk')
        self.emit_abs(uk, k)
        x = self.fix_temp('a')
        self.fix_abs(x, dest)
        rem = self.temp_pair('div')
        self.emit_mov(rem + 1, x[0])
        self.emit_mov(rem, self.CONST_ZERO)
        for _ in range(self.cfg.N):
            self.emit_divstep(rem, uk)
        self.emit_mov(dest[0], rem + 1)
        self.emit_mov(rem + 1, x[1])
        for _ in range(self.cfg.N):
            self.emit_divstep(rem, uk)
        self.emit_mov(dest[1], rem + 1)
        self.fix_negate(dest, sign)

    def fix_div(self, dest: Tuple[int, int], src: Tuple[int, int]) -> None:
        if self.rational:
            return self.rat_op('div', dest, src)
        N = self.cfg.N
        ones = self.const(-1)
        sign = self._fix_sign(dest[0], src[0])
        b = self.fix_temp('b')
        self.fix_abs(b, src)
        num = self.fix_temp('a')
        self.fix_abs(num, dest)
        low = self.temp('fix_t')
        r = self.temp_pair('mul')
        r = (r, r + 1)
        q = dest
        t = self.temp_pair('div')
        t = (t, t + 1)
        nb = self.temp('fix_mask')
        over = self.temp('fix_one')
        take = self.temp('fix_cmp')
        count = self.temp('div_count')
        for slot in (low, r[0], r[1], q[0], q[1]):
            self.emit_mov(slot, self.CONST_ZERO)
        self.emit_mov(count, self.const(3 * N))
        loop = self.new_label('fdiv_loop')
        keep = self.new_label('fdiv_keep')
        self.emit_label(loop)
        self.emit_add(low, low)
        self.emit_adc(num[1], num[1])
        self.emit_adc(num[0], num[0])
        self.emit_adc(r[1], r[1])
        self.emit_adc(r[0], r[0])
        self.emit_mov(over, self.CONST_ZERO)
        self.emit_adc(over, self.CONST_ZERO)
        self.emit_mov(t[1], r[1])
        self.emit_sub(t[1], b[1])
        self.emit_mov(t[0], r[0])
        self.emit_mov(nb, b[0])
        self.emit_xor(nb, ones)
        self.emit_adc(t[0], nb)
        self.emit_mov(take, self.CONST_ZERO)
        self.emit_adc(take, self.CONST_ZERO)
        self.emit_or(take, over)
        self.emit_add(q[1], q[1])
        self.emit_adc(q[0], q[0])
        self.emit_or(q[1], take)
        self.emit_jz(take, keep)
        self.fix_copy(r, t)
        self.emit_label(keep)
        self.emit_dec(count)
        self.emit_jnz(count, loop)
        self.fix_negate(dest, sign)

    def fix_nonzero(self, expr: Expr, dest: int) -> int:
        v = self.compile_fix(expr, self.fix_temp(f'cvt{self._fix_level}'))
        self.emit_mov(dest, v[0])
        if not self.rational:
            self.emit_or(dest, v[1])
        return dest

    def is_fix_compare(self, expr: BinaryOp) -> bool:
        return 'fix' in (self.type_of(expr.left), self.type_of(expr.right))

    def fix_compare(self, expr: BinaryOp, dest: int) -> int:
        a = self.fix_temp('cmp_a')
        b = self.fix_temp('cmp_b')
        self.fix_copy(a, self.compile_fix(expr.left, a))
        _, kept = self._protect(list(a), lambda: self.fix_copy(b, self.compile_fix(expr.right, b)))
        self.fix_copy(a, tuple(kept))
        op = expr.op
        if self.rational:
            self.rat_call('cmp', a[0], a[1], b[0], b[1])
            yes, end = (self.new_label('rcmp_yes'), self.new_label('rcmp_end'))
            self._emit_comparison(op, self.RETVAL, self.CONST_ZERO, yes, invert=False)
            self.emit_mov(dest, self.CONST_ZERO)
            self.emit_jmp(end)
            self.emit_label(yes)
            self.emit_mov(dest, self.CONST_ONE)
            self.emit_label(end)
            return dest
        if op in ('>', '<='):
            a, b = (b, a)
        t = self.temp('fix_cmp')
        yes = self.new_label('fcmp_yes')
        no = self.new_label('fcmp_no')
        end = self.new_label('fcmp_end')
        self.emit_mov(t, a[0])
        self.emit_sub(t, b[0])
        if op in ('==', '!='):
            self.emit_jnz(t, no)
            self.emit_mov(t, a[1])
            self.emit_sub(t, b[1])
            self.emit_jnz(t, no)
            self.emit_jmp(yes)
        else:
            self.emit_jz(t, f'{no}_lo')
            self.emit_mov(t, a[0])
            self.emit_max(t, b[0])
            self.emit_sub(t, a[0])
            self.emit_jnz(t, yes)
            self.emit_jmp(no)
            self.emit_label(f'{no}_lo')
            self.emit_mov(t, a[1])
            self.emit_sub(t, b[1])
            self.emit_mov(t, self.CONST_ZERO)
            self.emit_adc(t, self.CONST_ZERO)
            self.emit_jz(t, yes)
        self.emit_label(no)
        self.emit_mov(dest, self.CONST_ZERO)
        self.emit_jmp(end)
        self.emit_label(yes)
        self.emit_mov(dest, self.CONST_ONE)
        self.emit_label(end)
        if op in ('!=', '<=', '>='):
            self.emit_xor(dest, self.CONST_ONE)
        return dest
RAT_RUNTIME = '\nint __rat_n; int __rat_d; int __rat_t; int __rat_u; int __rat_g; int __rat_a; int __rat_b; int __rat_c; int __rat_e;\nnoinline int __rat_gcd(int a, int b) {\n    if (a < 0) { a = -a; }\n    if (b < 0) { b = -b; }\n    while (b != 0) { __rat_t = a % b; a = b; b = __rat_t; }\n    return a;\n}\nnoinline int __rat_set(int n, int d) {\n    __rat_t = 0 / d;\n    if (d < 0) { n = -n; d = -d; }\n    __rat_g = __rat_gcd(n, d);\n    __rat_n = n / __rat_g;\n    __rat_d = d / __rat_g;\n    return 0;\n}\nnoinline int __rat_add(int an, int ad, int bn, int bd) {\n    __rat_g = __rat_gcd(ad, bd);\n    __rat_u = ad / __rat_g;\n    __rat_c = bd / __rat_g;\n    __rat_set(an * __rat_c + bn * __rat_u, __rat_u * bd);\n    return 0;\n}\nnoinline int __rat_sub(int an, int ad, int bn, int bd) {\n    __rat_add(an, ad, -bn, bd);\n    return 0;\n}\nnoinline int __rat_mul(int an, int ad, int bn, int bd) {\n    __rat_g = __rat_gcd(an, bd);\n    __rat_a = an / __rat_g;\n    __rat_b = bd / __rat_g;\n    __rat_g = __rat_gcd(bn, ad);\n    __rat_c = bn / __rat_g;\n    __rat_e = ad / __rat_g;\n    __rat_set(__rat_a * __rat_c, __rat_e * __rat_b);\n    return 0;\n}\nnoinline int __rat_div(int an, int ad, int bn, int bd) {\n    __rat_t = 0 / bn;\n    __rat_mul(an, ad, bd, bn);\n    return 0;\n}\nnoinline int __rat_cmp(int an, int ad, int bn, int bd) {\n    __rat_add(an, ad, -bn, bd);\n    if (__rat_n < 0) { return -1; }\n    if (__rat_n > 0) { return 1; }\n    return 0;\n}\nnoinline int __rat_floor(int n, int d) {\n    __rat_t = n % d;\n    if (__rat_t < 0) { __rat_t += d; }\n    return (n - __rat_t) / d;\n}\n'
RAT_ROOT_RUNTIME = '\nint __rat_inexact; int __rat_domain;\nint __rat_x; int __rat_y; int __rat_q; int __rat_p; int __rat_j; int __rat_lo; int __rat_hi; int __rat_mid;\nint __rat_rn; int __rat_rd; int __rat_ra; int __rat_rb;\nnoinline int __rat_isqrt(int n) {\n    if (n < 0) { __rat_t = 1 / __rat_domain; }\n    if (n < 2) { return n; }\n    __rat_x = (n - n % 2) / 2 + 1;\n    __rat_q = (n - n % __rat_x) / __rat_x;\n    __rat_y = __rat_x + __rat_q;\n    __rat_y = (__rat_y - __rat_y % 2) / 2;\n    while (__rat_y < __rat_x) {\n        __rat_x = __rat_y;\n        __rat_q = (n - n % __rat_x) / __rat_x;\n        __rat_y = __rat_x + __rat_q;\n        __rat_y = (__rat_y - __rat_y % 2) / 2;\n    }\n    return __rat_x;\n}\nnoinline int __rat_pow_le(int m, int k, int n) {\n    __rat_p = 1;\n    __rat_j = 0;\n    while (__rat_j < k) {\n        __rat_q = (n - n % m) / m;\n        if (__rat_p > __rat_q) { return 0; }\n        __rat_p = __rat_p * m;\n        __rat_j = __rat_j + 1;\n    }\n    return 1;\n}\nnoinline int __rat_iroot(int n, int k) {\n    __rat_lo = 0;\n    __rat_hi = __rat_isqrt(n) + 1;\n    while (__rat_hi - __rat_lo > 1) {\n        __rat_mid = __rat_hi - __rat_lo;\n        __rat_mid = __rat_lo + (__rat_mid - __rat_mid % 2) / 2;\n        if (__rat_pow_le(__rat_mid, k, n)) { __rat_lo = __rat_mid; } else { __rat_hi = __rat_mid; }\n    }\n    return __rat_lo;\n}\nnoinline int __rat_pow(int m, int k) {\n    __rat_p = 1;\n    __rat_j = 0;\n    while (__rat_j < k) {\n        __rat_p = __rat_p * m;\n        __rat_j = __rat_j + 1;\n    }\n    return __rat_p;\n}\nnoinline int __rat_root(int n, int d, int k) {\n    __rat_set(n, d);\n    __rat_rn = __rat_n;\n    __rat_rd = __rat_d;\n    if (k == 2) {\n        __rat_ra = __rat_isqrt(__rat_rn);\n        __rat_rb = __rat_isqrt(__rat_rd);\n    } else {\n        __rat_ra = __rat_iroot(__rat_rn, k);\n        __rat_rb = __rat_iroot(__rat_rd, k);\n    }\n    if (__rat_pow(__rat_ra, k) != __rat_rn) { __rat_t = 1 / __rat_inexact; }\n    if (__rat_pow(__rat_rb, k) != __rat_rd) { __rat_t = 1 / __rat_inexact; }\n    __rat_n = __rat_ra;\n    __rat_d = __rat_rb;\n    return 0;\n}\n'
RAT_POW_RUNTIME = '\nint __rat_limit; int __rat_negexp; int __rat_pn; int __rat_pd; int __rat_bn; int __rat_bd; int __rat_pi; int __rat_bi;\nnoinline int __rat_powi(int n, int d, int e) {\n    if (e > 4096) { __rat_t = 1 / __rat_limit; }\n    if (e < -4096) { __rat_t = 1 / __rat_limit; }\n    __rat_bn = n;\n    __rat_bd = d;\n    if (e < 0) {\n        __rat_set(d, n);\n        __rat_bn = __rat_n;\n        __rat_bd = __rat_d;\n        e = -e;\n    }\n    __rat_pn = 1;\n    __rat_pd = 1;\n    while (e > 0) {\n        if (e % 2 != 0) {\n            __rat_mul(__rat_pn, __rat_pd, __rat_bn, __rat_bd);\n            __rat_pn = __rat_n;\n            __rat_pd = __rat_d;\n        }\n        e = (e - e % 2) / 2;\n        if (e > 0) {\n            __rat_mul(__rat_bn, __rat_bd, __rat_bn, __rat_bd);\n            __rat_bn = __rat_n;\n            __rat_bd = __rat_d;\n        }\n    }\n    __rat_n = __rat_pn;\n    __rat_d = __rat_pd;\n    return 0;\n}\nnoinline int __rat_ipow(int a, int e) {\n    if (e > 4096) { __rat_t = 1 / __rat_limit; }\n    if (e < 0) { __rat_t = 1 / __rat_negexp; }\n    __rat_pi = 1;\n    __rat_bi = a;\n    while (e > 0) {\n        if (e % 2 != 0) { __rat_pi = __rat_pi * __rat_bi; }\n        e = (e - e % 2) / 2;\n        if (e > 0) { __rat_bi = __rat_bi * __rat_bi; }\n    }\n    return __rat_pi;\n}\n'
RAT_MODULES = {'roots': (RAT_ROOT_RUNTIME, re.compile('\\b(fsqrt|froot|isqrt)\\s*\\('), {'__rat_inexact': -1, '__rat_domain': -2}, 3), 'pow': (RAT_POW_RUNTIME, re.compile('\\b(fpowi|ipowi)\\s*\\('), {'__rat_limit': -3, '__rat_negexp': -4}, 4)}
_RT_TOKENS: Dict[tuple, List[Token]] = {}
_RT_PREFIX: Dict[tuple, 'CodeGen'] = {}

def _rational_prefix(cfg: Config, checked: bool, modules: Tuple[str, ...]=()) -> 'CodeGen':
    key = (cfg.s, cfg.m, cfg.n, cfg.N, checked) + tuple(modules)
    if key not in _RT_PREFIX:

        def build(start: Optional[int]) -> 'CodeGen':
            cg = CodeGen(cfg, checked=checked, rational=True)
            low = cg.mem.next_addr
            if start is not None:
                cg.mem.next_addr = start
            rt = Parser(_runtime_tokens(modules) + [Token(TT.EOF, '', 0)]).parse_program()
            for func in rt.functions:
                cg.functions[func.name] = func
            for g in rt.globals:
                cg.compile_var_decl(g)
            cg.emit_jmp('__main_entry')
            for func in rt.functions:
                cg.compile_noinline_body(func)
                cg.prefix_bodies.add(func.name)
                if func.name not in cg.noinline_called:
                    cg.noinline_called.append(func.name)
            if modules:
                marks = {cg.mem.get_var(w): mark for mod in modules for w, mark in RAT_MODULES[mod][2].items()}
                for i, c in cg.checks.items():
                    if c[0] == 'div' and len(c) == 3 and (c[2] in marks):
                        cg.checks[i] = c + (marks[c[2]],)
                assert sum((len(c) == 4 for c in cg.checks.values())) == sum((RAT_MODULES[m][3] for m in modules)), 'every stop must be marked'
            if start is not None:
                cg.mem.reserved = (start, cg.mem.next_addr)
                cg.mem.next_addr = low
            cg.prefix_low = low
            return cg
        probe = build(None)
        size = probe.mem.next_addr - probe.prefix_low
        start, built = (cfg.m - size, probe)
        while start > probe.prefix_low:
            try:
                built = build(start)
                break
            except RuntimeError:
                start -= 1
        _RT_PREFIX[key] = built
    return _RT_PREFIX[key].clone()

def _runtime_tokens(modules: Tuple[str, ...]=()) -> List[Token]:
    if modules not in _RT_TOKENS:
        _RT_TOKENS[modules] = lex(RAT_RUNTIME + ''.join((RAT_MODULES[m][0] for m in modules)))[:-1]
    return list(_RT_TOKENS[modules])

def compile_c(source: str, s: int=32, m: int=64, n: int=1024, N: int=8, checked: bool=False, numbers: str='binary') -> Tuple[Config, List[int], List[Tuple[int, int, int]], Dict]:
    cfg = Config(s=s, m=m, n=n, N=N)
    if numbers not in ('binary', 'rational'):
        raise ValueError(f"numbers must be 'binary' or 'rational', not {numbers!r}")
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse_program()
    if numbers == 'rational':
        codegen = _rational_prefix(cfg, checked, tuple((m for m, mod in RAT_MODULES.items() if mod[1].search(source))))
    else:
        codegen = CodeGen(cfg, checked=checked)
    codegen.compile_program(program)
    memory = codegen.mem.init_memory()
    commands = codegen.resolve()
    meta = {'variables': {name: slot.addr for name, slot in codegen.mem.slots.items() if not name.startswith('__')}, 'constants': codegen.mem.constants.copy(), 'num_instructions': len(commands), 'retval_addr': codegen.RETVAL, 'output_addr': codegen.OUTPUT, 'labels': codegen.labels.copy(), 'source_map': codegen.source_map, 'sizes': {name: slot.size for name, slot in codegen.mem.slots.items() if not name.startswith('__')}, 'checks': dict(codegen.checks), 'fix_literal_rounding': codegen.fix_literal_rounding, 'numbers': numbers, 'fix_variables': {name: codegen.mem.slots[name].addr for name, t in codegen.var_types.items() if t == 'fix' and (not name.startswith('__')) and (name in codegen.mem.slots)}}
    return (cfg, memory, commands, meta)

def fix_value(hi: int, lo: int, N: int) -> float:
    return hi + (lo & (1 << N) - 1) / (1 << N)
