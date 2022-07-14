from sys import argv, stdin, stdout
from pprint import pprint
from PySCPP import compiler, vm
from PySCPP.utils import display_errors
from typing import TypedDict
from glob import glob

class DirtyOptions(TypedDict):
	"""
		See help for description of options.
	"""
	input: str | None
	tokens: bool
	silent: bool
	tree: bool
	scan: bool
	run: bool
	out: str
	opt: int


class Options(TypedDict):
	"""
		See help for description of options.
	"""
	input: str
	tokens: bool
	silent: bool
	tree: bool
	scan: bool
	run: bool
	out: str
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
	print("    --tree, -T    Output the parsed tree. Don't scan the file.")
	print("    --scan, -S    Output the scanned tree. Don't assemble the file.")
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
	"--run": "run",
	"r": "run",
	"T": "tree",
	"--tree": "tree",
	"--scan": "scan",
	"S": "scan",
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

defaults: DirtyOptions = {
	"input": None,
	"tokens": False,
	"silent": False,
	"tree": False,
	"scan": False,
	"run": False,
	"out": "out.slvm.txt",
	"opt": 0,
}


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
	opts: DirtyOptions = defaults.copy()
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
				print(f"Unknown option: {v}")
				quit(1)
			elif i[-1] in flags:
				opts[flags[i[-1]]] = True  # type: ignore
				continue
			elif i[-1] in multi_flags:
				opts[multi_flags[i[-1]]] += 1  # type: ignore
				continue
			else:
				print(f"Unknown flag: {i[-1]}")
				quit(1)

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
	compiler.OPT = options["opt"]
	print(options['opt'])
	with open(options["input"], "r") as f:
		code = f.read()
	if options["tokens"] or options["tree"] or options["scan"]:
		errors = show_tree_or_tokens(code, options)
	out,errors = compiler.compile(code, options["input"])
	if errors:
		if not options["silent"]:
			display_errors(errors)
		quit(1)
	if options["run"]:
		class IO:
			write = stdout.write
			flush = stdout.flush
			read = stdin.read
		vm.SLVM.console = IO
		vm.SLVM(out).run()
		quit(1)
	with open(options["out"], "w") as f:
		f.write(out)


def show_tree_or_tokens(code, options):
	tokens = compiler.tokenize(code, options["input"])
	if options["tokens"]:
		for token in tokens:
			print(token.type,token.value,sep="\t")
		quit(0)
	tree, result = compiler.parse(tokens)
	if result:
		if not options["silent"]:
			display_errors(result)
		quit(1)
	if options["tree"]:
		pprint(tree)
		quit(0)
	tree, result = compiler.Scanner(tree).scan()
	if result:
		if not options["silent"]:
			display_errors(result)
		quit(1)
	pprint(tree)
	quit(0)

	return result

if __name__ == '__main__':
	main()
