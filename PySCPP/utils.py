from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeAlias, TypeVar


Pos: TypeAlias = tuple[int, int, str]

VERBOSITY: int = 0


@dataclass
class Warning:
	type: str
	message: str
	stack: list[Pos] = field(default_factory=list)


@dataclass
class Error:
	message: str
	pos: Pos
	stack: list[Pos] = field(default_factory=list)


T = TypeVar("T")
TR = TypeVar("TR")


class Monad(Generic[T]):
	value: T
	errors: list[Error]
	warnings: list[Warning]
	force: bool

	def __init__(
		self, value: T, errors: list[Error] | None = None,
		warnings: list[Warning] | None = None, force: bool = False
	):
		if errors is None:
			errors = []
		if warnings is None:
			warnings = []
		self.value = value
		self.errors = errors
		self.warnings = warnings
		self.force = force

	def bind(self, transform: Callable[[T],Monad[TR]]) -> Monad[TR]:
		if self.errors and not self.force:
			return self  # type:ignore
		res = transform(self.value)
		res.errors.extend(self.errors)
		res.warnings.extend(self.warnings)
		if self.force:
			res.force = True
		return res

	# meta-programing: >>=
	def __rshift__(self, transform: Callable[[T],Monad[TR]]) -> Monad[TR]:
		return self.bind(transform)


file_cache: dict[str,str] = {}


def display_errors(errors: list[Error]) -> None:
	"""
	It prints the error message, the line of code where the error occurred,
	and the line of code.

	@param errors
	@param code The code that was being parsed, to take lines from
	"""

	print("Errors:")
	p_stack: list[Pos] = []
	for error in errors:
		if error.pos[2] not in file_cache:
			try:
				file_cache[error.pos[2]] = open(error.pos[2], "r").read()
			except FileNotFoundError:
				print(f"Could not find file: {error.pos[2]}")
				continue
		lines = file_cache[error.pos[2]].split("\n")
		if error.stack:
			for i,j in zip(error.stack, p_stack):
				if i != j:
					print("In:")
					print(f"{i[0]+1: >2}| ",lines[i[0]])
			for i in error.stack[len(p_stack):]:
				print("In:")
				print(f"{i[0]+1: >2}| ",lines[i[0]])

			p_stack = error.stack

		print(error.message, f"({error.pos[2]}:{error.pos[0]+1}:{error.pos[1]})")
		try:
			print(f"{error.pos[0]+1: >2}| ",lines[error.pos[0]])
			tabs = lines[error.pos[0]][:error.pos[1]].count("\t")
			print("   ",'\t'*tabs," "*(error.pos[1]-tabs-4),"^")
		except IndexError:
			print(f"{error.pos[0]+1: >2}| ","<EOF>")


def log(*things: object, level: int = 2) -> None:
	"""
	Logs things to the console.
	:param things:
	:param level: The level of the log. 0 is the most important.
	"""
	if level > VERBOSITY:
		return
	print(*things)


def debug(*things: object):
	log(*things, level=3)


def info(*things: object):
	log(*things, level=1)
