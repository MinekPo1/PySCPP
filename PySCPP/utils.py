from dataclasses import dataclass, field
from typing import TypeAlias


Pos: TypeAlias = tuple[int, int, str]


@dataclass
class Error:
	message: str
	pos: Pos
	stack: list[Pos] = field(default_factory=list)


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

		print(error.message)
		print(f"{error.pos[0]+1: >2}| ",lines[error.pos[0]])
		tabs = lines[error.pos[0]][:error.pos[1]].count("\t")
		print("   ",'\t'*tabs," "*(error.pos[1]-tabs-4),"^")
