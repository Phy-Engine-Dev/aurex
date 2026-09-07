from __future__ import annotations

from dataclasses import dataclass
import re


class MemoryLoweringError(ValueError):
    pass


@dataclass(frozen=True)
class Memory:
    kind: str
    signedness: str
    packed_msb: int
    packed_lsb: int
    name: str
    array_left: int
    array_right: int

    @property
    def width(self) -> int:
        return abs(self.packed_msb - self.packed_lsb) + 1

    @property
    def indices(self) -> list[int]:
        step = 1 if self.array_right >= self.array_left else -1
        return list(range(self.array_left, self.array_right + step, step))

    def element(self, index: int) -> str:
        suffix = f"n{abs(index)}" if index < 0 else str(index)
        return f"{self.name}__aurex_mem_{suffix}"


_DECL = re.compile(
    r"\b(?P<kind>reg|logic)\s+"
    r"(?:(?P<signed>signed|unsigned)\s+)?"
    r"\[\s*(?P<pmsb>-?\d+)\s*:\s*(?P<plsb>-?\d+)\s*\]\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_$]*)\s*"
    r"\[\s*(?P<aleft>-?\d+)\s*:\s*(?P<aright>-?\d+)\s*\]\s*;"
)


def _mask_noncode(source: str) -> str:
    out = list(source)
    i = 0
    while i < len(source):
        if source.startswith("//", i):
            j = source.find("\n", i)
            j = len(source) if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif source.startswith("/*", i):
            j = source.find("*/", i + 2)
            if j < 0:
                raise MemoryLoweringError("unterminated block comment")
            j += 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif source[i] == '"':
            j = i + 1
            while j < len(source):
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == '"':
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, len(source))):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def _replace_spans(source: str, replacements: list[tuple[int, int, str]]) -> str:
    for start, end, value in sorted(replacements, reverse=True):
        source = source[:start] + value + source[end:]
    return source


def _find_close(masked: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for i in range(start, len(masked)):
        if masked[i] == opening:
            depth += 1
        elif masked[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    raise MemoryLoweringError(f"unterminated {opening}{closing} expression")


def _declarations(source: str) -> list[tuple[Memory, tuple[int, int]]]:
    masked = _mask_noncode(source)
    found: list[tuple[Memory, tuple[int, int]]] = []
    names: set[str] = set()
    for match in _DECL.finditer(masked):
        memory = Memory(
            match.group("kind"),
            match.group("signed") or "",
            int(match.group("pmsb")),
            int(match.group("plsb")),
            match.group("name"),
            int(match.group("aleft")),
            int(match.group("aright")),
        )
        if memory.name in names:
            raise MemoryLoweringError(f"duplicate unpacked array name {memory.name!r} in one source")
        if memory.width > 256 or len(memory.indices) > 256 or memory.width * len(memory.indices) > 16384:
            raise MemoryLoweringError(f"unpacked array {memory.name!r} exceeds export lowering limits")
        names.add(memory.name)
        found.append((memory, match.span()))
    return found


def _read_expression(memory: Memory, index_expr: str) -> str:
    default = "{" + str(memory.width) + "{1'bx}}"
    value = default
    for index in reversed(memory.indices):
        value = f"(({index_expr}) == {index} ? {memory.element(index)} : {value})"
    return value


def _lower_memory_rvalues(source: str, memory: Memory) -> str:
    masked = _mask_noncode(source)
    name_re = re.compile(r"\b" + re.escape(memory.name) + r"\s*\[")
    replacements = []
    for match in name_re.finditer(masked):
        open_pos = masked.find("[", match.start(), match.end())
        close_pos = _find_close(masked, open_pos, "[", "]")
        index_expr = source[open_pos + 1:close_pos].strip()
        tail = close_pos + 1
        while tail < len(masked) and masked[tail].isspace():
            tail += 1
        if tail < len(masked) and masked[tail] == "[":
            raise MemoryLoweringError(
                f"packed selection of unpacked array element {memory.name!r} is not supported by export lowering"
            )
        replacements.append((match.start(), close_pos + 1, _read_expression(memory, index_expr)))
    return _replace_spans(source, replacements)


def _lower_one(source: str, memory: Memory) -> str:
    masked = _mask_noncode(source)
    declaration_span = next(
        match.span() for match in _DECL.finditer(masked) if match.group("name") == memory.name
    )
    name_re = re.compile(r"\b" + re.escape(memory.name) + r"\s*\[")
    replacements: list[tuple[int, int, str]] = []
    declaration_ranges = [declaration_span]
    skip_until = 0
    for match in name_re.finditer(masked):
        if match.start() < skip_until:
            continue
        if any(a <= match.start() < b for a, b in declaration_ranges):
            continue
        open_pos = masked.find("[", match.start(), match.end())
        close_pos = _find_close(masked, open_pos, "[", "]")
        index_expr = source[open_pos + 1:close_pos].strip()
        if not index_expr:
            raise MemoryLoweringError(f"empty index into unpacked array {memory.name!r}")
        tail = close_pos + 1
        while tail < len(masked) and masked[tail].isspace():
            tail += 1
        assignment = None
        if masked.startswith("<=", tail):
            assignment = "<="
        elif tail < len(masked) and masked[tail] == "=" and not masked.startswith("==", tail):
            assignment = "="
        if assignment:
            rhs_start = tail + len(assignment)
            semicolon = masked.find(";", rhs_start)
            if semicolon < 0:
                raise MemoryLoweringError(f"unterminated assignment to unpacked array {memory.name!r}")
            rhs = _lower_memory_rvalues(source[rhs_start:semicolon].strip(), memory)
            arms = " ".join(
                ("if" if position == 0 else "else if")
                + f" (({index_expr}) == {index}) {memory.element(index)} {assignment} {rhs};"
                for position, index in enumerate(memory.indices)
            )
            replacements.append((match.start(), semicolon + 1, f"begin {arms} end"))
            skip_until = semicolon + 1
            continue
        if tail < len(masked) and masked[tail] == "[":
            raise MemoryLoweringError(
                f"packed selection of unpacked array element {memory.name!r} is not supported by export lowering"
            )
        replacements.append((match.start(), close_pos + 1, _read_expression(memory, index_expr)))

    signed = f" {memory.signedness}" if memory.signedness else ""
    decls = "\n".join(
        f"{memory.kind}{signed} [{memory.packed_msb}:{memory.packed_lsb}] {memory.element(index)};"
        for index in memory.indices
    )
    replacements.append((declaration_span[0], declaration_span[1], decls))
    return _replace_spans(source, replacements)


def _lower_module(source: str) -> tuple[str, list[dict[str, int | str]]]:
    declarations = _declarations(source)
    if not declarations:
        return source, []
    lowered = source
    metadata = []
    # Repeat discovery after each edit so spans stay valid.
    for original, _ in declarations:
        current = next((item for item, _ in _declarations(lowered) if item.name == original.name), None)
        if current is None:
            raise MemoryLoweringError(f"lost unpacked array declaration {original.name!r}")
        lowered = _lower_one(lowered, current)
        metadata.append({"name": original.name, "width": original.width, "depth": len(original.indices)})
    return lowered, metadata


def lower_unpacked_register_arrays(source: str) -> tuple[str, list[dict[str, int | str]]]:
    masked = _mask_noncode(source)
    module_tokens = list(re.finditer(r"\b(?:module|endmodule)\b", masked))
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    for token in module_tokens:
        if token.group() == "module":
            if stack:
                raise MemoryLoweringError("nested module declarations are invalid")
            stack.append(token.start())
        elif not stack:
            raise MemoryLoweringError("endmodule without module")
        else:
            spans.append((stack.pop(), token.end()))
    if stack:
        raise MemoryLoweringError("module without endmodule")
    if not spans:
        return source, []

    replacements = []
    metadata = []
    for start, end in spans:
        lowered, module_metadata = _lower_module(source[start:end])
        if module_metadata:
            replacements.append((start, end, lowered))
            metadata.extend(module_metadata)
    return _replace_spans(source, replacements), metadata
