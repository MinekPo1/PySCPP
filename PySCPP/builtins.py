from PySCPP import AST
from copy import deepcopy

print_ = AST.FuncDef(
	(2, 0, __file__),
	[
		AST.RawASM(
			(3, 0, __file__),
			[
				AST.Literal(
					(3, 0, __file__),
					"print",
				)
			]
		)
	],
	"print",
	[
		AST.Var(
			(3, 0, __file__),
			"!",
		)
	],
	True
)

println = AST.FuncDef(
	(26, 0, __file__),
	[
		AST.RawASM(
			(3, 0, __file__),
			[
				AST.Literal(
					(3, 0, __file__),
					"println",
				)
			]
		)
	],
	"println",
	[
		AST.Var(
			(3, 0, __file__),
			"!",
		)
	],
	True
)


all_builtins = {
	"print": print_,
	"println": println,
}

def get_builtins():
	return deepcopy(all_builtins)
