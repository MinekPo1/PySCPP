from sys import argv
from pprint import pprint
from PySCPP import compiler, vm, AST, utils
from PySCPP.utils import display_errors, Monad, info
from typing import TypedDict
from pathlib import Path
from glob import glob
from time import time


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
	out: str | None
	opt: int
	lib: bool
	objs: bool
	graphics: bool
	verbose: int
	debug: bool
	dump: bool
	memory: str | None


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
	out: str | None
	opt: int
	lib: bool
	objs: bool
	graphics: bool
	verbose: int
	debug: bool
	dump: bool
	memory: str | None


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
	print("    --inp, -i     Where to take the code from. Can be a glob pattern.")
	print("    --out, -o     Where to put the result. {} will be replaced with")
	print("                  the input file name.")
	print("    --opt, -O     When entered once the compiler will not obfuscate")
	print("                  variables to save space. When entered a second time")
	print("                  the compiler will not preform optimizations other")
	print("                  than dead code elimination. When entered a third")
	print(
		"                  time the compiler will not preform any optimizations."
	)
	print("    --lib, -l     Show the list of available libraries and exit.")
	print("    --graph, -g   Enable graphics mode in the vm.")
	print("    --verbose, -v Show more info. Can be entered multiple times.")
	print("    --obj         Show the list of available objects and exit.")
	print("    --debug, -D   Place debug symbols in the output.")
	print("    --dump, -d    Dump the VM memory and var address after running.")
	quit(0)


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
	"--lib": "lib",
	"l": "lib",
	"--objs": "objs",
	"--debug": "debug",
	"D": "debug",
	"--dump": "dump",
	"d": "dump",
	"--graph": "graphics",
	"g": "graphics",
}

multi_flags = {
	"--opt": "opt",
	"O": "opt",
	"v": "verbose",
	"--verbose": "verbose",
}

options = {
	"--out": "out",
	"o": "out",
	"i": "input",
	"--inp": "input",
	"--mem": "memory"
}

defaults: DirtyOptions = {
	"input": None,
	"tokens": False,
	"silent": False,
	"tree": False,
	"scan": False,
	"run": False,
	"out": None,
	"opt": 0,
	"lib": False,
	"objs": False,
	"graphics": False,
	"verbose": 0,
	"debug": False,
	"dump": False,
	"memory": None,
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
	if options["lib"]:
		print("Available libraries:")
		libs: list[Path] = []
		for i in compiler.INCLUDE_PATH:
			libs.extend(i.glob("*.sc"))
		for i in sorted(libs):
			print(i.name)
		quit(0)

	if options["graphics"]:
		vm.GRAPHICS = True
	compiler.OPT = options["opt"]
	compiler.DEBUG = options["debug"]
	utils.VERBOSITY = options["verbose"]
	if options["memory"] is not None:
		vm.SLVM.MAX_MEMORY_SIZE = int(options["memory"])  # type:ignore
	for fn in glob(options["input"]):
		if not options["silent"]:
			print(f"{fn}:")

		file = Path(fn)
		code = file.read_text()
		if not fn.endswith(".txt"):
			t1 = time()
			if options["tokens"] or options["tree"] or options["scan"]\
				or options["objs"]:
				show_tree_or_tokens(code, options, fn)
				continue
			monad = compiler.compile(code, fn)
			if monad.errors:
				if not options["silent"]:
					display_errors(monad.errors)
				continue

			if not options["silent"]:
				print(f"Compiled in {(time() - t1) * 1000:.3f} ms")
		else:
			monad = Monad(errors=[], value=code)
			if not options["silent"]:
				info("Already compiled")

		if options["run"]:
			class IO:
				def write(self, s: str) -> None:
					print(s, end="")

				def read(self) -> str:
					return input()

				def flush(self) -> None:
					pass

			vm.SLVM.console = IO()  # type:ignore
			the_vm = vm.SLVM(monad.value)
			the_vm.console = IO()  # type:ignore
			try:
				the_vm.run()
			except KeyboardInterrupt:
				print("\nInterrupted")
			if options["dump"]:
				with open("dump.txt", "w") as f:
					f.write("\n".join(map(str,the_vm._memory._raw)))
					f.write("\n")
					for k,v in the_vm._var_lookup.items():
						f.write(f"{k}:{v}\n")
			if options["out"] is None:
				continue
		with open(
			(
				options["out"] or "{}.slvm.txt"
			).format(file.name.removesuffix(".sc")),
			"w"
		) as f:
			f.write(monad.value)


def show_tree_or_tokens(code, options, fn):
	tokens = compiler.tokenize(code, fn)
	if options["tokens"]:
		for token in tokens:
			print(token.type,token.value,sep="\t")
		return
	monad = Monad(tokens)
	monad >>= compiler.parse
	if options["tree"]:
		if monad.errors:
			display_errors(monad.errors)
			return
		pprint(monad.value)
		return
	if options["objs"]:
		scanner = compiler.Scanner(monad.value)
		scanner.scan()
		# if scanner.errors:
		# 	display_errors(scanner.errors)
		# 	return
		for k,v in scanner.objects.items():
			type_ = "    "
			if isinstance(v, (AST.FuncDef, compiler.ScannedFunction)):
				type_ = "func"
			elif isinstance(v, AST.Namespace):
				type_ = " ns "
			else:
				type_ = " var"
			if scanner.is_private.get(k):
				print("private",type_,k)
			else:
				print(" public",type_,k)
	monad >>= compiler.Scanner.do
	if options["scan"]:
		if monad.errors:
			display_errors(monad.errors)
			return
		pprint(monad.value)
		return


if __name__ == '__main__':
	main()
