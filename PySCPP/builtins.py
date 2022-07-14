from PySCPP import AST

print_ = AST.FuncDef(
	(3, 0, __file__),
	[
		AST.RawASM(
			(3, 0, __file__),
			[
				# AST.Literal(
				# 	(3, 0, __file__),
				# 	"loadAtVar"
				# ),
				# AST.Var(
				# 	(3, 0, __file__),
				# 	"val"
				# ),
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


all_builtins = {
	"print": print_,
}
