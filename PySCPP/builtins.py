from PySCPP import AST
from copy import deepcopy

print_ = AST.FuncDef(
	(4, 0, __file__),
	[
		AST.RawASM(
			(7, 0, __file__),
			[
				AST.Literal(
					(10, 0, __file__),
					"print",
				)
			]
		)
	],
	"print",
	[
		AST.Var(
			(19, 0, __file__),
			"!",
		)
	],
	True
)

println = AST.FuncDef(
	(27, 0, __file__),
	[
		AST.RawASM(
			(30, 0, __file__),
			[
				AST.Literal(
					(33, 0, __file__),
					"println",
				)
			]
		)
	],
	"println",
	[
		AST.Var(
			(42, 0, __file__),
			"!",
		)
	],
	True
)

malloc = AST.FuncDef(
	(50, 0, __file__),
	[
		AST.RawASM(
			(53, 0, __file__),
			[
				AST.Literal(
					(56, 0, __file__),
					"malloc"
				),
				AST.Var(
					(62, 0, __file__),
					"size"
				)
			]
		)
	],
	"malloc",
	[
		AST.Var(
			(56, 0, __file__),
			"size"
		)
	],
	True
)

all_builtins = {
	"print": print_,
	"println": println,
	"malloc": malloc
}


def get_builtins():
	return deepcopy(all_builtins)
