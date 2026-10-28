# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read primitive allocator snapshots without executing serialization programs."""

import pickletools
from pathlib import Path


def load_snapshot(path: str | Path) -> dict:
    """Read a plain-data snapshot, rejecting object construction and globals.

    Only scalar, dictionary, list, tuple, framing and memo opcodes are accepted.
    This intentionally does not implement general pickle deserialization.
    """
    data = Path(path).read_bytes()
    stack = []
    marks = []
    memo = {}
    scalar_ops = {
        "BININT",
        "BININT1",
        "BININT2",
        "LONG",
        "LONG1",
        "LONG4",
        "INT",
        "BINFLOAT",
        "FLOAT",
        "UNICODE",
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
        "BINBYTES",
        "SHORT_BINBYTES",
        "BINBYTES8",
    }
    for opcode, argument, position in pickletools.genops(data):
        name = opcode.name
        if name in ("PROTO", "FRAME"):
            if name == "PROTO" and argument > 5:
                raise ValueError("Unsupported snapshot protocol")
        elif name in scalar_ops:
            if type(argument) not in (int, float, str, bytes, bool):
                raise ValueError("Nonprimitive snapshot scalar")
            stack.append(argument)
        elif name == "NONE":
            stack.append(None)
        elif name in ("NEWTRUE", "NEWFALSE"):
            stack.append(name == "NEWTRUE")
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "MARK":
            marks.append(len(stack))
        elif name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            if not stack:
                raise ValueError("Empty snapshot memo operand")
            memo[len(memo) if name == "MEMOIZE" else argument] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo[argument])
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            count = int(name[-1])
            if len(stack) < count:
                raise ValueError("Missing snapshot tuple operands")
            values = tuple(stack[-count:])
            del stack[-count:]
            stack.append(values)
        elif name in ("APPEND", "SETITEM"):
            count = 1 if name == "APPEND" else 2
            if len(stack) <= count:
                raise ValueError("Missing snapshot container operands")
            values = stack[-count:]
            del stack[-count:]
            if name == "APPEND" and type(stack[-1]) is list:
                stack[-1].append(values[0])
            elif name == "SETITEM" and type(stack[-1]) is dict:
                stack[-1][values[0]] = values[1]
            else:
                raise ValueError("Incorrect snapshot container type")
        elif name in ("APPENDS", "SETITEMS", "TUPLE", "LIST", "DICT"):
            if not marks:
                raise ValueError("Missing snapshot mark")
            start = marks.pop()
            values = stack[start:]
            del stack[start:]
            if name in ("TUPLE", "LIST"):
                stack.append(tuple(values) if name == "TUPLE" else values)
            elif name == "DICT":
                if len(values) % 2:
                    raise ValueError("Unpaired snapshot dictionary entry")
                stack.append(dict(zip(values[::2], values[1::2])))
            elif name == "APPENDS" and stack and type(stack[-1]) is list:
                stack[-1].extend(values)
            elif name == "SETITEMS" and stack and type(stack[-1]) is dict:
                if len(values) % 2:
                    raise ValueError("Unpaired snapshot dictionary entry")
                stack[-1].update(zip(values[::2], values[1::2]))
            else:
                raise ValueError("Incorrect snapshot container type")
        elif name == "STOP":
            if marks or len(stack) != 1 or type(stack[0]) is not dict:
                raise ValueError("Snapshot must contain one plain dictionary")
            if position + 1 != len(data):
                raise ValueError("Trailing snapshot payload")
            return stack[0]
        else:
            # In particular GLOBAL/STACK_GLOBAL/REDUCE/BUILD/NEWOBJ and
            # persistent references are rejected, never imported or invoked.
            raise ValueError(f"Unsupported snapshot opcode: {name}")
    raise ValueError("Snapshot has no STOP opcode")
