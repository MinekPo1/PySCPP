from sys import argv
from pprint import pprint, pformat
from PySCPP import compiler


def main() -> None:
	if len(argv) == 1 or argv[1] == "--help" or len(argv) > 3:
		print("Usage:")
		print("    PySCPP <file>")
		print("    PySCPP <file> <output>")
		quit(1)
	with open(argv[1], "r") as f:
		code = f.read()
	tokens = compiler.tokenize(code, argv[1])
	print(*tokens,sep="\n")
	tree, errors = compiler.parse(tokens)
	if errors:
		print("Errors:")
		lines = code.splitlines()
		for error in errors:
			print(error.message)
			print(f"{error.pos[0]: >2}| ",lines[error.pos[0]])

		quit(1)
	if len(argv) == 2:
		pprint(tree)
	else:
		with open(argv[2], "w") as f:
			f.write(pformat(tree))


if __name__ == '__main__':
	main()
