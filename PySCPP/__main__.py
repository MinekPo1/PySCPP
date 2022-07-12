from sys import argv
from pprint import pprint, pformat
from PySCPP import compiler
from typing import TypedDict


class DirtyOptions(TypedDict):
	"""
		See help for description of options.
	"""
	input: str | None
	tokens: bool
	silent: bool
	run: bool
	out: str | None
	opt: int


class Options(TypedDict):
	"""
		See help for description of options.
	"""
	input: str
	tokens: bool
	silent: bool
	run: bool
	out: str | None
	opt: int


def print_help():
	"""
		Prints the help message.
		Probably should be just a string but oh well.
	"""
	print("Usage:")
	print("    PySCPP <input> [<output>] [options]")
	print()
	print("Options:")
	print("    --help, -h    Show this help message and exit")
	print("    --tokens, -t  Output tokens. Don't parse the file.")
	print("    --silent, -s  Don't print errors.")
	print("    --run, -r     Run the created assembly.")
	print("    --inp, -i     Where to take the code from.")
	print("    --out, -o     Where to put the result.")
	print("    --opt, -O     When entered once the compiler will not obfuscate")
	print("                  variables to save space. When entered a second time")
	print("                  the compiler will not preform optimizations other")
	print("                  than dead code elimination. When entered a third")
	print(
		"                  time the compiler will not preform any optimizations."
	)
	quit(1)


arguments = [
	"input",
]

flags = {
	"--tokens": "tokens",
	"t": "tokens",
	"--silent": "silent",
	"s": "silent",
}

multi_flags = {
	"--opt": "opt",
	"O": "opt",
}

options = {
	"--out": "out",
	"o": "out",
	"i": "input",
	"--input": "input",
}


def display_errors(errors: list[compiler.Error], code:str) -> None:
	"""
	It prints the error message, the line of code where the error occurred,
	and the line of code.

	@param errors
	@param code The code that was being parsed, to take lines from
	"""

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


def parse_args() -> Options:
	"""
		Responsible for parsing the command line arguments.
		Returns:
		- input file
		- output file
		- options

	"""
	if len(argv) == 1 or "--help" in argv or "-h" in argv:
		print_help()
	opts: DirtyOptions = {
		"input": None,
		"tokens": False,
		"silent": False,
		"run": False,
		"out": None,
		"opt": 0,
	}
	for i in argv[1:]:
		if i.startswith("--"):
			if i in flags:
				opts[flags[i]] = True  # type: ignore
				continue
			if i in multi_flags:
				opts[multi_flags[i]] += 1  # type: ignore
				continue
			if "=" in i:
				k,v = i.split("=")
				if k in options:
					opts[options[k]] = v  # type:ignore
					continue
				print(f"Unknown option: {k}")
				quit(1)
			if i in options:
				print(f"Option {i} requires a value")
				quit(1)
			print(f"Unknown flag: {i}")
			quit(1)

		if i.startswith("-"):
			v = None
			if "=" in i:
				i,v = i.split("=")
			for j in i[1:-1]:
				if j in flags:
					opts[flags[j]] = True  # type: ignore
					continue
				if j in multi_flags:
					opts[multi_flags[j]] += 1  # type: ignore
					continue
				print(f"Unknown flag: {j}")
				quit(1)
			if v is not None:
				if i[-1] in options:
					opts[options[i[-1]]] = v  # type:ignore
					continue
				print(f"Unknown option: {k}")
				quit(1)
			elif i[-1] in flags:
				opts[flags[j]] = True  # type: ignore
			elif j in multi_flags:
				opts[multi_flags[j]] += 1  # type: ignore



		for j in arguments:
			if opts[j] is None:
				opts[j] = i
				break

	for i in arguments:
		if opts[i] is None:  # type:ignore
			print(f"Missing argument: {i}")
			quit(1)

	return opts  # type:ignore


def main() -> None:
	"""
		Main function.
	"""
	options = parse_args()
	print(options['opt'])
	with open(options["input"], "r") as f:
		code = f.read()
	tokens = compiler.tokenize(code, options["input"])
	if not options["tokens"]:
		tree, errors = compiler.parse(tokens)
		if errors:
			if not options["silent"]:
				display_errors(errors, code)
			quit(1)

		if options["out"] is None:
			pprint(tree)
		else:
			with open(options["out"], "w") as f:
				f.write(pformat(tree))
	elif options["out"] is None:
		for token in tokens:
			print(token.type,token.value,sep="\t")
	else:
		with open(options["out"], "w") as f:
			for token in tokens:
				f.write(str(token.type))
				f.write("\t")
				f.write(str(token.value))
				f.write("\n")


if __name__ == '__main__':
	main()
