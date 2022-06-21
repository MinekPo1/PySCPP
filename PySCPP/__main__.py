from sys import argv
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
	if len(argv) == 2:
		for token in tokens:
			print(token)
	else:
		with open(argv[2], "w") as f:
			for token in tokens:
				f.write(str(token) + "\n")

if __name__ == '__main__':
	main()
