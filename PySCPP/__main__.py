from sys import argv
from pprint import pprint, pformat
from PySCPP import compiler
from typing import TypedDict


class Options(TypedDict):
	tokens: bool
	silent: bool
	test: str


def print_help():
	print("Usage:")
	print("    PySCPP <file>")
	print("    PySCPP <file> <output>")
	print()
	print("Options:")
	print("    --help, -h    Show this help message and exit")
	print("    --tokens, -t  Output tokens. Don't parse the file.")
	print("    --silent, -s  Don't print errors.")
	quit(1)


flags = {
	"--tokens": "tokens",
	"-t": "tokens",
	"--silent": "silent",
	"-s": "silent",
}

options = {
	"--test": "test",
}


def display_errors(errors: list[compiler.Error], code:str) -> None:
	print("Errors:")
	lines = code.splitlines()
	p_stack: list[compiler.Pos] = []
	for error in errors:
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


def parse_args() -> tuple[str, str | None, Options]:
	if len(argv) == 1 or "--help" in argv or "-h" in argv:
		print_help()
	inp = None
	out = None
	opts = {
		"tokens": False,
		"silent": False,
		"test": ""
	}
	for i in argv[1:]:
		if i.startswith("-"):
			if i in flags:
				opts[flags[i]] = True
				continue
			if "=" in i:
				k,v = i.split("=")
				if k in options:
					opts[options[k]] = v
					continue
				print(f"Unknown option: {k}")
				quit(1)
			if i in options:
				print(f"Option {i} requires a value")
				quit(1)
			print(f"Unknown flag: {i}")
			quit(1)

		elif inp is None:
			inp = i
		else:
			out = i

	if inp is None:
		print("No input file specified.")
		quit(1)

	return inp, out, opts  # type: ignore


def main() -> None:
	inp, out, options = parse_args()
	with open(inp, "r") as f:
		code = f.read()
	tokens = compiler.tokenize(code, inp)
	if not options["tokens"]:
		tree, errors = compiler.parse(tokens)
		if errors:
			if not options["silent"]:
				display_errors(errors, code)
			quit(1)

		if out is None:
			pprint(tree)
		else:
			with open(out, "w") as f:
				f.write(pformat(tree))
	elif out is None:
		for token in tokens:
			print(token.type,token.value,sep="\t")
	else:
		with open(out, "w") as f:
			for token in tokens:
				f.write(str(token.type))
				f.write("\t")
				f.write(str(token.value))
				f.write("\n")


if __name__ == '__main__':
	main()
