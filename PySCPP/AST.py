from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TypeAlias


Pos: TypeAlias = tuple[int, int, str]


@dataclass
class Element:
	pos: Pos


class Expression(Element):
	pass


@dataclass
class Container(Element):
	children: list[Element]


@dataclass
class Accessible(Element):
	name: str
	private: bool | None = field(default=None, init=False)


@dataclass
class Root(Container):
	includes: list[Include]
	resolved_includes: list[str] = field(default_factory=list)


@dataclass
class Preprocessor(Element):
	value: str


@dataclass
class TokenElement(Element):
	token: Any


@dataclass
class Namespace(Accessible,Container):
	default_private: bool | None = None
	is_: str | None = None


@dataclass
class If(Container):
	condition: Expression


@dataclass
class While(Container):
	condition: Expression


@dataclass
class For(Container):
	var: str
	start: Expression
	end: Expression
	step: Expression | None


@dataclass
class VarDef(Element):
	var: Var
	value: Expression | None
	private: bool | None = None
	offset: Expression | None = None


@dataclass
class Return(Element):
	value: Expression | None


@dataclass
class FuncDef(Accessible, Container):
	args: list[Var]
	inline: bool = False


@dataclass
class Var(Expression):
	name: str
	memory_specifier: str | None = None


@dataclass
class VarSet(Element):
	l_value: Var | Expression
	value: Expression
	offset: Expression | None = None
	modifier: str | None = None


@dataclass
class MemoryModifier(Expression):
	modifier: str
	value: Expression


@dataclass
class FuncCall(Expression):
	name: str
	args: list[Expression]
	is_builtin: bool = False


@dataclass
class Operation(Expression):
	op: str
	left: Expression
	right: Expression


class GetValueOfA(Expression):
	pass


@dataclass
class RawASM(Expression):
	arguments: list[Expression]


@dataclass
class Literal(Expression):
	value: str | int


@dataclass
class LiteralArray(Expression):
	values: list[Expression]


@dataclass
class Include(Element):
	value: str


@dataclass
class DefineRef(Expression):
	expr: Expression


@dataclass
class StrongArrayRef(Expression):
	array: Var
	index: Expression


@dataclass
class WeakArrayRef(Expression):
	array: Expression
	index: Expression


@dataclass
class Switch(Element):
	cases: list[tuple[list[Literal], Case]]
	value: Expression
	default: Case | None = None


@dataclass
class Case(Container):
	pass
