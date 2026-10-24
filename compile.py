"""PyCompile - компилятор для удобной работы с Python."""

from __future__ import annotations

import argparse
import ast
import builtins
import difflib
import io
import json
import os
import shutil
import shlex
import site
import subprocess
import sys
import sysconfig
import symtable
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


APP_NAME = "PyCompile"
JDK_URL = "https://adoptium.net/temurin/releases/"


@dataclass
class Diagnostic:
    severity: str
    code: str
    message: str
    line: int | None = None
    column: int | None = None
    advice: str | None = None
    phase: str = "АНАЛИЗ"

    def render(self, filename: str) -> str:
        location = filename
        if self.line is not None:
            location += f":{self.line}"
            if self.column is not None:
                location += f":{self.column}"
        result = f"{location}: {self.severity} [{self.phase}] {self.code}: {self.message}"
        if self.advice:
            result += f"\n  Совет: {self.advice}"
        return result


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.diagnostics: list[Diagnostic] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        for operator, right in zip(node.ops, node.comparators):
            if isinstance(operator, (ast.Eq, ast.NotEq)) and (
                isinstance(node.left, ast.Constant) and node.left.value is None
                or isinstance(right, ast.Constant) and right.value is None
            ):
                self.diagnostics.append(
                    Diagnostic(
                        "ПРЕДУПРЕЖДЕНИЕ",
                        "PY001",
                        "Для сравнения с None рекомендуется использовать is или is not.",
                        node.lineno,
                        node.col_offset + 1,
                        "Замените == на is или != на is not.",
                        "СЕМАНТИЧЕСКИЙ АНАЛИЗ",
                    )
                )
                break
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.diagnostics.append(
                Diagnostic(
                    "ПРЕДУПРЕЖДЕНИЕ",
                    "PY002",
                    "Перехватываются все исключения, включая системные.",
                    node.lineno,
                    node.col_offset + 1,
                    "Укажите конкретный тип исключения, например except ValueError:.",
                    "СЕМАНТИЧЕСКИЙ АНАЛИЗ",
                )
            )
        self.generic_visit(node)


class ModuleSemanticAnalyzer(ast.NodeVisitor):
    """Проверяет явные обращения к неизвестным именам на уровне модуля."""

    def __init__(self) -> None:
        self.bound_names: set[str] = set(dir(builtins))
        self.bound_names.update(
            {"__name__", "__file__", "__package__", "__doc__", "__loader__", "__spec__", "__cached__", "__builtins__"}
        )
        self.loads: list[ast.Name] = []

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bound_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind_target(item)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.append(node)
        elif isinstance(node.ctx, ast.Store):
            self.bound_names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound_names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.bound_names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound_names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound_names.add(node.name)


def has_jdk() -> bool:
    javac = shutil.which("javac")
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        executable = "javac.exe" if os.name == "nt" else "javac"
        candidate = Path(java_home) / "bin" / executable
        if candidate.is_file():
            javac = str(candidate)
    return bool(javac)


def require_jdk() -> bool:
    if has_jdk():
        return True
    print(
        f"ОШИБКА: не найден JDK (компилятор javac).\n"
        f"Установите JDK: {JDK_URL}\n"
        "После установки задайте JAVA_HOME и добавьте папку JDK\\bin в PATH.",
        file=sys.stderr,
    )
    return False


def read_python_file(path: Path) -> str:
    if path.suffix.lower() != ".py":
        raise ValueError("Поддерживаются только файлы с расширением .py.")
    with tokenize.open(path) as source_file:
        return source_file.read()


def syntax_error_message(error: SyntaxError) -> tuple[str, str]:
    text = (error.msg or "").lower()
    if "expected ':'" in text:
        return "После этой строки пропущено двоеточие.", "Добавьте : в конце заголовка конструкции."
    if "unexpected eof" in text or "was never closed" in text or "never closed" in text:
        return "Конструкция не завершена: возможно, не хватает закрывающей скобки или кавычки.", "Проверьте парные скобки и кавычки."
    if "invalid syntax" in text:
        return "Обнаружена синтаксическая ошибка Python.", "Проверьте операторы, отступы и написание ключевых слов."
    if "expected an indented block" in text:
        return "После конструкции требуется блок кода с отступом.", "Добавьте тело блока и используйте единый отступ."
    return f"Синтаксическая ошибка: {error.msg}.", "Исправьте указанную строку и запустите проверку повторно."


def lexical_analysis(source: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for item in tokens:
            if item.type == tokenize.OP and item.string not in tokenize.EXACT_TOKEN_TYPES:
                message = f"Недопустимый оператор или символ {item.string!r}."
            elif item.type == tokenize.ERRORTOKEN and not item.string.isspace():
                message = (
                    "Похоже, строковый литерал не закрыт кавычкой."
                    if item.string in {"'", '"'}
                    else f"Недопустимый символ {item.string!r}."
                )
            else:
                continue
            diagnostics.append(
                Diagnostic(
                    "ОШИБКА",
                    "PYLEX001",
                    message,
                    item.start[0],
                    item.start[1] + 1,
                    "Проверьте кавычки и допустимые символы Python.",
                    "ЛЕКСИЧЕСКИЙ АНАЛИЗ",
                )
            )
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        message = error.args[0] if isinstance(error, tokenize.TokenError) else str(error)
        location = error.args[1] if isinstance(error, tokenize.TokenError) else (
            getattr(error, "lineno", None),
            getattr(error, "offset", None),
        )
        line, column = location if isinstance(location, tuple) else (None, None)
        diagnostics.append(
            Diagnostic(
                "ОШИБКА",
                "PYLEX002",
                f"Не удалось разобрать последовательность токенов: {message}.",
                line,
                column,
                "Проверьте незакрытые скобки, строки и отступы.",
                "ЛЕКСИЧЕСКИЙ АНАЛИЗ",
            )
        )
    return diagnostics


def semantic_analysis(tree: ast.Module, source: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    try:
        compile(tree, "<pycompile>", "exec")
    except SyntaxError as error:
        diagnostics.append(
            Diagnostic(
                "ОШИБКА",
                "PYSEM001",
                f"Ошибка семантической проверки компилятором Python: {error.msg}.",
                error.lineno,
                error.offset,
                "Проверьте корректность использования return, break, continue и аргументов.",
                "СЕМАНТИЧЕСКИЙ АНАЛИЗ",
            )
        )
        return diagnostics

    analyzer = ModuleSemanticAnalyzer()
    analyzer.visit(tree)
    reported: set[str] = set()
    load_nodes: dict[str, ast.Name] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            load_nodes.setdefault(node.id, node)

    def inspect_scope(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if (
                symbol.is_global()
                and symbol.is_referenced()
                and not symbol.is_assigned()
                and symbol.get_name() not in analyzer.bound_names
            ):
                name = symbol.get_name()
                if name not in load_nodes or name in reported:
                    continue
                name_node = load_nodes[name]
                reported.add(name)
                diagnostics.append(
                    Diagnostic(
                        "ПРЕДУПРЕЖДЕНИЕ",
                        "PYSEM002",
                        f"Имя «{name}» используется, но не объявлено и не является встроенным.",
                        name_node.lineno,
                        name_node.col_offset + 1,
                        "Объявите переменную, импортируйте её или проверьте написание имени.",
                        "СЕМАНТИЧЕСКИЙ АНАЛИЗ",
                    )
                )
        for child in table.get_children():
            inspect_scope(child)

    inspect_scope(symtable.symtable(source, "<pycompile>", "exec"))

    for node in analyzer.loads:
        if node.id in analyzer.bound_names or node.id in reported:
            continue
        reported.add(node.id)
        diagnostics.append(
            Diagnostic(
                "ПРЕДУПРЕЖДЕНИЕ",
                "PYSEM002",
                f"Имя «{node.id}» используется на уровне модуля, но не объявлено.",
                node.lineno,
                node.col_offset + 1,
                "Объявите переменную или проверьте написание имени.",
                "СЕМАНТИЧЕСКИЙ АНАЛИЗ",
            )
        )
    return diagnostics


def analyze_source(source: str) -> list[Diagnostic]:
    diagnostics = lexical_analysis(source)
    if any(item.severity == "ОШИБКА" for item in diagnostics):
        return diagnostics

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        message, advice = syntax_error_message(error)
        diagnostics.append(
            Diagnostic(
                "ОШИБКА",
                "PY000",
                message,
                error.lineno,
                error.offset,
                advice,
                "СИНТАКСИЧЕСКИЙ АНАЛИЗ",
            )
        )
        return diagnostics

    diagnostics.extend(semantic_analysis(tree, source))
    analyzer = PythonAnalyzer()
    analyzer.visit(tree)
    diagnostics.extend(analyzer.diagnostics)
    lines = source.splitlines()
    for line_number, line in enumerate(lines, 1):
        if line.rstrip(" \t") != line:
            diagnostics.append(
                Diagnostic(
                    "ПРЕДУПРЕЖДЕНИЕ",
                    "PY003",
                    "В конце строки есть лишние пробелы.",
                    line_number,
                    len(line.rstrip(" \t")) + 1,
                    "Удалите пробелы в конце строки.",
                    "СТИЛИСТИЧЕСКИЙ АНАЛИЗ",
                )
            )
    return sorted(
        diagnostics,
        key=lambda item: (item.line or 0, item.column or 0, item.code),
    )


def technical_diagnostics(path: Path, source: str, diagnostics: list[Diagnostic]) -> str:
    try:
        tokens = sum(
            1
            for item in tokenize.generate_tokens(io.StringIO(source).readline)
            if item.type not in {tokenize.ENCODING, tokenize.ENDMARKER}
        )
    except (tokenize.TokenError, IndentationError):
        tokens = "не определено: ошибка токенизации"
    try:
        with path.open("rb") as source_file:
            encoding, _ = tokenize.detect_encoding(source_file.readline)
    except (OSError, SyntaxError, UnicodeError):
        encoding = "не определена"

    lines = len(source.splitlines())
    byte_count = len(source.encode("utf-8"))
    errors = sum(item.severity == "ОШИБКА" for item in diagnostics)
    warnings = sum(item.severity == "ПРЕДУПРЕЖДЕНИЕ" for item in diagnostics)
    phases = ", ".join(
        f"{phase}: {sum(item.phase == phase for item in diagnostics)}"
        for phase in (
            "ЛЕКСИЧЕСКИЙ АНАЛИЗ",
            "СИНТАКСИЧЕСКИЙ АНАЛИЗ",
            "СЕМАНТИЧЕСКИЙ АНАЛИЗ",
            "СТИЛИСТИЧЕСКИЙ АНАЛИЗ",
        )
    )
    return (
        "Техническая диагностика: "
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; "
        f"кодировка файла: {encoding}; "
        f"строк: {lines}; UTF-8 байт: {byte_count}; токенов: {tokens}; "
        f"ошибок: {errors}; предупреждений: {warnings}.\n"
        f"  Этапы — {phases}."
    )


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(source):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _absolute_offset(source: str, offsets: list[int], line: int, byte_column: int) -> int:
    start = offsets[line - 1]
    text_line = source[start:].split("\n", 1)[0]
    encoded = text_line.encode("utf-8")[:byte_column]
    return start + len(encoded.decode("utf-8", errors="ignore"))


def autofix_source(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    offsets = _line_offsets(source)
    edits: list[tuple[int, int, str]] = []

    class SafeFixes(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            if len(node.ops) == 1 and len(node.comparators) == 1:
                operator = node.ops[0]
                right = node.comparators[0]
                left_is_none = isinstance(node.left, ast.Constant) and node.left.value is None
                right_is_none = isinstance(right, ast.Constant) and right.value is None
                if isinstance(operator, (ast.Eq, ast.NotEq)) and (left_is_none or right_is_none):
                    if node.end_lineno is None or node.end_col_offset is None:
                        self.generic_visit(node)
                        return
                    start_abs = _absolute_offset(source, offsets, node.lineno, node.col_offset)
                    end_abs = _absolute_offset(source, offsets, node.end_lineno, node.end_col_offset)
                    left_text = ast.get_source_segment(source, node.left)
                    right_text = ast.get_source_segment(source, right)
                    if left_text is not None and right_text is not None:
                        expression = (
                            f"{right_text} is{' not' if isinstance(operator, ast.NotEq) else ''} None"
                            if left_is_none
                            else f"{left_text} is{' not' if isinstance(operator, ast.NotEq) else ''} None"
                        )
                        edits.append((start_abs, end_abs, expression))
            self.generic_visit(node)

    SafeFixes().visit(tree)

    for line_number, line in enumerate(source.splitlines(keepends=True), 1):
        content = line.rstrip("\r\n")
        stripped = content.rstrip(" \t")
        if stripped != content:
            start = offsets[line_number - 1] + len(stripped)
            end = offsets[line_number - 1] + len(content)
            edits.append((start, end, ""))

    fixed = source
    for start, end, replacement in sorted(edits, key=lambda edit: edit[0], reverse=True):
        fixed = fixed[:start] + replacement + fixed[end:]
    try:
        ast.parse(fixed)
    except SyntaxError:
        return source
    return fixed


class JavaScriptTranslator:
    def __init__(self) -> None:
        self.indent = 0

    def _line(self, text: str) -> str:
        return "    " * self.indent + text

    def translate(self, source: str) -> str:
        tree = ast.parse(source)
        lines: list[str] = []
        for statement in tree.body:
            lines.extend(self.statement(statement))
        return "\n".join(lines) + ("\n" if lines else "")

    def statement(self, node: ast.stmt) -> list[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError("async-функции пока не поддерживаются переводчиком.")
            if node.args.defaults or node.args.kwonlyargs or node.args.vararg or node.args.kwarg:
                raise ValueError("Параметры со значениями по умолчанию и *args/**kwargs пока не поддерживаются.")
            args = ", ".join(arg.arg for arg in node.args.args)
            result = [self._line(f"function {node.name}({args}) {{")]
            self.indent += 1
            for child in node.body:
                result.extend(self.statement(child))
            self.indent -= 1
            result.append(self._line("}"))
            return result
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise ValueError("Поддерживается присваивание одной переменной.")
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                raise ValueError("Поддерживается присваивание одной переменной.")
            return [self._line(f"var {target.id} = {self.expression(node.value)};")]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = self.expression(node.value) if node.value else "undefined"
            return [self._line(f"var {node.target.id} = {value};")]
        if isinstance(node, ast.Return):
            value = "" if node.value is None else " " + self.expression(node.value)
            return [self._line(f"return{value};")]
        if isinstance(node, ast.Expr):
            return [self._line(self.expression(node.value) + ";")]
        if isinstance(node, ast.If):
            result = [self._line(f"if ({self.expression(node.test)}) {{")]
            self.indent += 1
            for child in node.body:
                result.extend(self.statement(child))
            self.indent -= 1
            if node.orelse:
                result.append(self._line("} else {"))
                self.indent += 1
                for child in node.orelse:
                    result.extend(self.statement(child))
                self.indent -= 1
            result.append(self._line("}"))
            return result
        if isinstance(node, ast.While):
            result = [self._line(f"while ({self.expression(node.test)}) {{")]
            self.indent += 1
            for child in node.body:
                result.extend(self.statement(child))
            self.indent -= 1
            result.append(self._line("}"))
            return result
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
                raise ValueError("Поддерживается только цикл вида for переменная in range(...).")
            if not 1 <= len(node.iter.args) <= 3:
                raise ValueError("range() должен содержать от одного до трёх аргументов.")
            args = [self.expression(arg) for arg in node.iter.args]
            start, stop = ("0", args[0]) if len(args) == 1 else (args[0], args[1])
            step = args[2] if len(args) == 3 else "1"
            condition = f"{node.target.id} < {stop}" if step == "1" else f"{node.target.id} > {stop}" if step == "-1" else f"({step} > 0 ? {node.target.id} < {stop} : {node.target.id} > {stop})"
            result = [self._line(f"for (let {node.target.id} = {start}; {condition}; {node.target.id} += {step}) {{")]
            self.indent += 1
            for child in node.body:
                result.extend(self.statement(child))
            self.indent -= 1
            result.append(self._line("}"))
            return result
        if isinstance(node, ast.Pass):
            return [self._line(";")]
        if isinstance(node, ast.Break):
            return [self._line("break;")]
        if isinstance(node, ast.Continue):
            return [self._line("continue;")]
        raise ValueError(f"Конструкция {type(node).__name__} пока не поддерживается переводчиком.")

    def expression(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "null"
            if node.value is True:
                return "true"
            if node.value is False:
                return "false"
            return json.dumps(node.value, ensure_ascii=False)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return "[" + ", ".join(self.expression(item) for item in node.elts) + "]"
        if isinstance(node, ast.Tuple):
            return "[" + ", ".join(self.expression(item) for item in node.elts) + "]"
        if isinstance(node, ast.Dict):
            entries = []
            for key, value in zip(node.keys, node.values):
                if key is None:
                    raise ValueError("Распаковка словаря пока не поддерживается.")
                entries.append(f"[{self.expression(key)}]: {self.expression(value)}")
            return "{" + ", ".join(entries) + "}"
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left), self.expression(node.right)
            if isinstance(node.op, ast.Add):
                return f"({left} + {right})"
            if isinstance(node.op, ast.Sub):
                return f"({left} - {right})"
            if isinstance(node.op, ast.Mult):
                return f"({left} * {right})"
            if isinstance(node.op, ast.Div):
                return f"({left} / {right})"
            if isinstance(node.op, ast.FloorDiv):
                return f"Math.floor({left} / {right})"
            if isinstance(node.op, ast.Mod):
                return f"({left} % {right})"
            if isinstance(node.op, ast.Pow):
                return f"({left} ** {right})"
            raise ValueError(f"Оператор {type(node.op).__name__} не поддерживается.")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                operator = "!"
            elif isinstance(node.op, ast.USub):
                operator = "-"
            elif isinstance(node.op, ast.UAdd):
                operator = "+"
            elif isinstance(node.op, ast.Invert):
                operator = "~"
            else:
                raise ValueError("Этот унарный оператор не поддерживается.")
            return f"({operator}{self.expression(node.operand)})"
        if isinstance(node, ast.BoolOp):
            operator = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + operator.join(self.expression(value) for value in node.values) + ")"
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError("Цепочки сравнений пока не поддерживаются.")
            operation = node.ops[0]
            if isinstance(operation, (ast.Eq, ast.Is)):
                operator = "==="
            elif isinstance(operation, (ast.NotEq, ast.IsNot)):
                operator = "!=="
            elif isinstance(operation, ast.Lt):
                operator = "<"
            elif isinstance(operation, ast.LtE):
                operator = "<="
            elif isinstance(operation, ast.Gt):
                operator = ">"
            elif isinstance(operation, ast.GtE):
                operator = ">="
            else:
                raise ValueError("Этот вид сравнения не поддерживается.")
            return f"({self.expression(node.left)} {operator} {self.expression(node.comparators[0])})"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                function = "console.log"
            else:
                function = self.expression(node.func)
            args = [self.expression(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("Именованные аргументы функций пока не поддерживаются в переводе.")
            return f"{function}({', '.join(args)})"
        if isinstance(node, ast.Attribute):
            return f"{self.expression(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{self.expression(node.value)}[{self.expression(node.slice)}]"
        if isinstance(node, ast.IfExp):
            return f"({self.expression(node.test)} ? {self.expression(node.body)} : {self.expression(node.orelse)})"
        raise ValueError(f"Выражение {type(node).__name__} пока не поддерживается переводчиком.")


def check_files(paths: Iterable[Path]) -> int:
    error_count = 0
    for path in paths:
        try:
            source = read_python_file(path)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"{path}: ОШИБКА: {error}", file=sys.stderr)
            print(
                f"Техническая диагностика: чтение файла не выполнено "
                f"({type(error).__name__}). Проверьте путь, права доступа и кодировку.",
                file=sys.stderr,
            )
            error_count += 1
            continue
        diagnostics = analyze_source(source)
        if not diagnostics:
            print(f"{path}: ошибок и предупреждений не найдено.")
        for diagnostic in diagnostics:
            print(diagnostic.render(str(path)))
            if diagnostic.severity == "ОШИБКА":
                error_count += 1
        print(technical_diagnostics(path, source, diagnostics))
    return 1 if error_count else 0


def fix_file(path: Path, write: bool) -> int:
    try:
        with path.open("rb") as source_file:
            encoding, _ = tokenize.detect_encoding(source_file.readline)
        source = path.read_bytes().decode(encoding)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"{path}: ОШИБКА: {error}", file=sys.stderr)
        return 1
    fixed = autofix_source(source)
    if fixed == source:
        print(f"{path}: безопасных автоматических исправлений не найдено.")
        return 0
    diff = difflib.unified_diff(
        source.splitlines(keepends=True),
        fixed.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (исправлено)",
    )
    print("".join(diff), end="")
    if write:
        try:
            path.write_bytes(fixed.encode(encoding))
        except OSError as error:
            print(f"Не удалось сохранить исправления: {error}", file=sys.stderr)
            return 1
        print(f"Изменения сохранены: {path}")
    else:
        print("Это предварительный просмотр. Для сохранения используйте флаг --write.")
    return 0


def scripts_directory() -> Path:
    if os.name == "nt":
        return Path(site.getusersitepackages()).parent / "Scripts"
    return Path(sysconfig.get_path("scripts", scheme="nt_user" if os.name == "nt" else "posix_user"))


def connect() -> int:
    scripts = scripts_directory()
    scripts.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    if os.name == "nt":
        launcher = scripts / "pycompile.cmd"
        command = subprocess.list2cmdline([sys.executable, str(script)]) + " %*"
        launcher.write_text(f"@echo off\r\n{command}\r\n", encoding="utf-8")
    else:
        launcher = scripts / "pycompile"
        launcher.write_text(
            "#!/bin/sh\nexec "
            + shlex.join([sys.executable, str(script)])
            + ' "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    print(f"{APP_NAME} подключён: {launcher}")
    print("Запуск: pycompile check путь\\к\\файлу.py")
    if str(scripts) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"Если команда не найдена, добавьте эту папку в PATH: {scripts}")
    return 0


def disconnect() -> int:
    scripts = scripts_directory()
    names = ["pycompile.cmd", "pycompile", "python-ide-compile.cmd", "python-ide-compile"]
    removed = False
    for name in names:
        launcher = scripts / name
        if launcher.is_file():
            launcher.unlink()
            removed = True
    print("Подключение PyCompile отменено." if removed else "Команда ранее не подключалась.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pycompile",
        description="PyCompile — проверка Python-кода и перевод в JavaScript.",
        epilog=(
            "Примеры: python compile.py check main.py | "
            "python compile.py fix main.py --write | "
            "python compile.py translate main.py -o main.js | "
            "python compile.py connect / disconnect"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Проверить один или несколько .py-файлов.")
    check_parser.add_argument("files", nargs="+", type=Path)

    fix_parser = subparsers.add_parser("fix", help="Показать или сохранить безопасные исправления.")
    fix_parser.add_argument("file", type=Path)
    fix_parser.add_argument("--write", action="store_true", help="Записать исправленный код в файл.")

    translate_parser = subparsers.add_parser("translate", help="Перевести поддерживаемый Python-код в JavaScript.")
    translate_parser.add_argument("file", type=Path)
    translate_parser.add_argument("-o", "--output", type=Path, help="Файл для результата; по умолчанию вывод в терминал.")

    subparsers.add_parser("connect", help="Подключить команду python-ide-compile в терминале.")
    subparsers.add_parser("disconnect", help="Удалить команду python-ide-compile из пользовательских Scripts.")
    return parser


def print_welcome() -> None:
    print(f"Добро пожаловать в {APP_NAME}!")
    print("Выполняйте лексический, синтаксический и семантический анализ,")
    print("получайте техническую диагностику, исправляйте код и переводите его:\n")
    print("  python compile.py check main.py")
    print("      Проверить один или несколько файлов .py.")
    print("  python compile.py fix main.py")
    print("      Показать безопасные исправления без изменения файла.")
    print("  python compile.py fix main.py --write")
    print("      Применить безопасные исправления к файлу.")
    print("  python compile.py translate main.py -o main.js")
    print("      Перевести поддерживаемый Python-код в JavaScript.")
    print("  python compile.py connect")
    print("      Подключить короткую команду pycompile.")
    print("  python compile.py disconnect")
    print("      Удалить подключённую команду pycompile.")
    print("\nПодробная справка: python compile.py --help")


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        if not require_jdk():
            return 2
        print_welcome()
        return 0
    args = parser.parse_args()
    if args.command == "disconnect":
        return disconnect()
    if not require_jdk():
        return 2
    if args.command == "check":
        return check_files(args.files)
    if args.command == "fix":
        return fix_file(args.file, args.write)
    if args.command == "translate":
        try:
            source = read_python_file(args.file)
            result = JavaScriptTranslator().translate(source)
            if args.output:
                if args.output.resolve() == args.file.resolve():
                    raise ValueError("Нельзя записывать JavaScript поверх исходного Python-файла.")
                args.output.write_text(result, encoding="utf-8", newline="")
                print(f"Перевод сохранён: {args.output}")
            else:
                sys.stdout.write(result)
        except (OSError, UnicodeError, SyntaxError, ValueError) as error:
            print(f"{args.file}: ОШИБКА перевода: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command == "connect":
        try:
            return connect()
        except OSError as error:
            print(f"Не удалось подключить команду: {error}", file=sys.stderr)
            return 1
    parser.error("Неизвестная команда.")


if __name__ == "__main__":
    raise SystemExit(main())
