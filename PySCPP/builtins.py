from __future__ import annotations
from PySCPP import AST
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from PySCPP.compiler import Assembler


def print_(asm: Assembler, *args: AST.Expression) -> None:
	for i in args[:-1]:
		asm.assemble_auto(i)
		asm.lines.append('print')
		asm.lines.append('ldi')
		asm.lines.append(' ')
		asm.lines.append('print')

	asm.assemble_auto(args[-1])
	asm.lines.append('print')


def println(asm: Assembler, *args: AST.Expression) -> None:
	for i in args[:-1]:
		asm.assemble_auto(i)
		asm.lines.append('print')
		asm.lines.append('ldi')
		asm.lines.append(' ')
		asm.lines.append('print')

	asm.assemble_auto(args[-1])
	asm.lines.append('println')


def malloc(asm: Assembler, *args: AST.Expression) -> None:
	assert len(args) == 1, 'malloc takes exactly one argument'
	if isinstance(args[0], AST.Literal):
		asm.lines.append('imalloc')
		asm.lines.append(str(args[0].value))
		return
	asm.assemble_auto(args[0])
	asm.lines.append('malloc')


def free(asm: Assembler, *args: AST.Expression) -> None:
	assert len(args) in {1,2}, 'free takes one or two arguments'
	assert isinstance(args[0], AST.Var), 'the first arg of free must be a variable'
	addr = asm.op_var()
	size = asm.op_var()
	asm.lines.append("getVarAddress")
	asm.lines.append(args[0].name)
	asm.lines.append("storeAtVar")
	asm.lines.append('free')


def exit(asm: Assembler, *args: AST.Expression) -> None:
	assert len(args) == 1, 'free takes no argument'
	asm.lines.append('done')


def concat(asm: Assembler, *args: AST.Expression) -> None:
	var = asm.op_var()
	asm.assemble_auto(args[-1])
	for i in args[1::-1]:
		asm.lines.append('storeAtVar')
		asm.lines.append(var)
		asm.assemble_auto(i)
		asm.lines.append('join')
		asm.lines.append(var)


all_builtins = {
	"print": print_,
	"println": println,
	"malloc": malloc,
	"free": free,
	"exit": exit,
	"concat": concat,
}
