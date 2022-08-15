from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Generic, Type, TypeAlias, TypeVar, NewType, \
	cast, Iterator
from re import compile as rec
from pathlib import Path
from string import printable

from PySCPP import AST
from PySCPP import utils
from PySCPP.utils import Error, Monad, debug, info  # log
from PySCPP import builtins
from copy import deepcopy

Pos: TypeAlias = tuple[int, int, str]


OPT = 0
DEBUG = False
"""
.. table:: ``OPT`` value and effect on the compiler
	:widths: auto

	===== ================ ================== =====================
	Value Name obfuscation Misc Optimizations Dead Code elimination
	===== ================ ================== =====================
	0     Yes              Yes                Yes
	1     No               Yes                Yes
	2     No               No                 Yes
	3     No               No                 No
	===== ================ ================== =====================
"""


INCLUDE_PATH = [
	Path(__file__).parent.parent / "lib",
	Path(__file__).parent.parent / "lib" / "std",
	Path(__file__).parent.parent / "lib" / "std" / "base",
]


class CodePointer:
	"""
		Tracks the position during parsing
	"""
	def __init__(self, code: str, source: str):
		self.code = code
		self.code_iter = iter(code)
		self.source = source
		self.line = 0
		self.column = 0

	def __next__(self):
		try:
			c = next(self.code_iter)
			if c == '\n':
				self.line += 1
				self.column = 0
			else:
				self.column += 1
		except StopIteration:
			return None
		return c

	@property
	def pos(self) -> Pos:
		return (self.line, self.column, self.source)


class RegexBank:
	"""
		Class for storing regex patterns, so they don't flood the namespace.
	"""
	number = rec(r'-?([0-9]+(\.[0-9]*)?|\.[0-9]+)')
	identifier = rec(r'[a-zA-Z_][a-zA-Z0-9_]*')
	hex_number = rec(r'0x[0-9a-fA-F]+')
	bin_number = rec(r'0b[01]+')
	tri_number = rec(r'0t[0-2]+')
	non_number = rec(r'0n[0-8]+')


@dataclass
class Token:
	"""
		Holds a token, its type and its position in the code.
	"""
	class Type(Enum):
		UNKNOWN = auto()
		NUMBER = auto()
		STRING = auto()
		IDENTIFIER = auto()
		OPERATOR = auto()
		COLON = auto()
		MEMBER_SELECT = auto()
		PREPROCESSOR = auto()
		KEYWORD = auto()
		BRACKET_OPEN = auto()
		BRACKET_CLOSE = auto()
		SQ_BRACKET_OPEN = auto()
		SQ_BRACKET_CLOSE = auto()
		PAREN_OPEN = auto()
		PAREN_CLOSE = auto()
		SEMICOLON = auto()
		COMMA = auto()
		EQUALS_SIGN = auto()
		MEMORY_MODIFIER = auto()
		ARROW = auto()
		MODIFIER = auto()

	type: Token.Type
	value: str
	pos: Pos


SingleCharacterTokens = {
	"+": Token.Type.OPERATOR,
	"-": Token.Type.OPERATOR,
	"*": Token.Type.OPERATOR,
	"/": Token.Type.OPERATOR,
	"%": Token.Type.OPERATOR,
	# "^": Token.Type.OPERATOR,
	"=": Token.Type.EQUALS_SIGN,
	"<": Token.Type.OPERATOR,
	">": Token.Type.OPERATOR,
	",": Token.Type.COMMA,
	";": Token.Type.SEMICOLON,
	"|": Token.Type.OPERATOR,
	"&": Token.Type.OPERATOR,
	"(": Token.Type.PAREN_OPEN,
	")": Token.Type.PAREN_CLOSE,
	"[": Token.Type.SQ_BRACKET_OPEN,
	"]": Token.Type.SQ_BRACKET_CLOSE,
	"{": Token.Type.BRACKET_OPEN,
	"}": Token.Type.BRACKET_CLOSE,
	":": Token.Type.COLON,
	"~": Token.Type.MEMORY_MODIFIER,
	"$": Token.Type.MEMORY_MODIFIER,
}

CompoundTokens = {
	"->": Token.Type.ARROW,
	"<=": Token.Type.OPERATOR,
	">=": Token.Type.OPERATOR,
	"!=": Token.Type.OPERATOR,
	"==": Token.Type.OPERATOR,
	"&&": Token.Type.OPERATOR,
	"||": Token.Type.OPERATOR,
	"..": Token.Type.OPERATOR,
	">>": Token.Type.OPERATOR,
	"<<": Token.Type.OPERATOR,
	"++": Token.Type.MODIFIER,
	"--": Token.Type.MODIFIER,
	"::": Token.Type.MEMBER_SELECT,
}

Keywords = {
	"if": Token.Type.KEYWORD,
	"while": Token.Type.KEYWORD,
	"for": Token.Type.KEYWORD,
	"return": Token.Type.KEYWORD,
	"var": Token.Type.KEYWORD,
	"namespace": Token.Type.KEYWORD,
	"from": Token.Type.KEYWORD,
	"to": Token.Type.KEYWORD,
	"by": Token.Type.KEYWORD,
	"func": Token.Type.KEYWORD,
	"public": Token.Type.KEYWORD,
	"private": Token.Type.KEYWORD,
	"inline": Token.Type.KEYWORD,
	"_asm_": Token.Type.KEYWORD,
	"_valueOfA_": Token.Type.KEYWORD,
	"is": Token.Type.KEYWORD,
	"else": Token.Type.KEYWORD,
	"using": Token.Type.KEYWORD,
	"switch": Token.Type.KEYWORD,
	"case": Token.Type.KEYWORD,
	"default": Token.Type.KEYWORD,
}


def tokenize(code: str, source: str, starting_pos: Pos | None = None) \
		-> list[Token]:
	"""
	It takes in code and a source file name both as strings,
	and returns a list of tokens

	:param code: The code to tokenize as a string
	:param source: name of the source file

	:return: A list of tokens.

	:warning: The code is very bad
	"""

	if source not in utils.file_cache:
		utils.file_cache[source] = code

	code += " "
	pointer = CodePointer(code, source)
	if starting_pos is not None:
		pointer.line = starting_pos[0]
		pointer.column = starting_pos[1]
	tokens: list[Token] = []
	current = ""
	token_shelf = None
	while True:

		if token_shelf is not None:
			tokens.append(token_shelf)
			token_shelf = None

		c = next(pointer)
		if c is None:
			break

		# comments
		if tokens != [] and c == "/" and current == "" and tokens[-1].value == "/":
			while True:
				c = next(pointer)
				if c is None:
					break
				if c == "\n":
					break
			del tokens[-1]
			continue
		if tokens != [] and c == "*" and current == "" and tokens[-1].value == "/":
			while True:
				c = next(pointer)
				if c is None:
					break
				if c == "*" and next(pointer) == "/":
					break
			del tokens[-1]
			continue

		if current + c in CompoundTokens:
			tokens.append(Token(CompoundTokens[current+c], current+c, pointer.pos))
			current = ""
			continue

		if tokens != [] and current == "" and tokens[-1].value + c in CompoundTokens:
			tokens[-1].value += c
			tokens[-1].type = CompoundTokens[tokens[-1].value]
			continue

		if c in SingleCharacterTokens:
			token_shelf = (Token(SingleCharacterTokens[c], c, pointer.pos))
			c = ""

		if c in ' \n\t' or not(c):
			if not(current):
				continue
			# decide token type
			if RegexBank.number.fullmatch(current):
				tokens.append(Token(Token.Type.NUMBER, current, pointer.pos))
			elif RegexBank.hex_number.fullmatch(current):
				tokens.append(Token(Token.Type.NUMBER, str(int(current,0)), pointer.pos))
			elif RegexBank.bin_number.fullmatch(current):
				tokens.append(Token(Token.Type.NUMBER, str(int(current,0)), pointer.pos))
			elif RegexBank.tri_number.fullmatch(current):
				tokens.append(Token(Token.Type.NUMBER, str(int(current[2:],3)),
					pointer.pos))
			elif RegexBank.non_number.fullmatch(current):
				tokens.append(Token(Token.Type.NUMBER, str(int(current[2:],9)),
					pointer.pos))
			elif current in Keywords:
				tokens.append(Token(Keywords[current], current, pointer.pos))
			elif RegexBank.identifier.fullmatch(current):
				tokens.append(Token(Token.Type.IDENTIFIER, current, pointer.pos))
			else:
				tokens.append(Token(Token.Type.UNKNOWN, current, pointer.pos))
			current = ""
			if token_shelf is not None:
				tokens.append(token_shelf)
				token_shelf = None
			continue

		if current == "" and c == "\"":

			# look for end of string
			while True:
				current += c
				c = next(pointer)
				if c is None:
					tokens.append(Token(Token.Type.UNKNOWN, current, pointer.pos))
					break
				if c == "\"" and current[-1] != "\\":
					current += c
					tokens.append(Token(Token.Type.STRING, current, pointer.pos))
					break
				if c == "\n":
					tokens.append(Token(Token.Type.UNKNOWN, current, pointer.pos))
					break
			current = ""
			continue

		if current == "#":
			pos = pointer.pos
			# look for newline
			while True:
				current += c
				c = next(pointer)
				if c is None:
					tokens.append(Token(Token.Type.UNKNOWN, current, pointer.pos))
					current = ""
					break
				if c == "\n":
					tokens.append(Token(Token.Type.PREPROCESSOR, current, pos))
					current = ""
					break
			continue

		current += c

	return tokens


T = TypeVar('T')


def _wrap(func: Callable[[Parser], T]) -> Callable[[Parser], T]:
	def wrapper(parser: Parser) -> T:
		try:
			return func(parser)
		except AssertionError as e:
			parser.assert_(False,e.args[0])
			parser.consume_token()
			return None  # type:ignore
	return wrapper


oper_weight = {
	'+': 1,
	'-': 1,
	'..': 2,
	'*': 2,
	'/': 2,
	'%': 2,
	'&': 3,
	'|': 3,
	# '^': 3,
	'<': 4,
	'>': 4,
	'<=': 4,
	'>=': 4,
	'==': 4,
	'!=': 4,
	'&&': 5,
	'||': 5,
	'<<': 4,
	'>>': 4,
}


class Parser:
	"""
		Takes a list of tokens and parses it into an AST.
	"""
	class _TokenViewType:
		def __init__(self, parser: Parser):
			self.parser = parser

		def __getitem__(self, index: int) -> Token:
			if index+self.parser.tokens_i >= len(self.parser.tokens):
				return Token(Token.Type.UNKNOWN, "<EOF>", self.parser.tokens[-1].pos)
			return self.parser.tokens[index+self.parser.tokens_i]

		def __delitem__(self, index: int) -> None:
			del self.parser.tokens[index+self.parser.tokens_i]

	tokens: list[Token]
	tokens_i = 0
	errors: list[Error]
	stack: list[AST.Container]
	root: AST.Root
	definitions: dict[str, AST.Expression]

	def __init__(
			self,
			tokens: list[Token], source: str, root: AST.Root | None = None
		):
		self.tokens = tokens
		self.errors = []
		if root is None:
			self.root = AST.Root(
				includes=[], pos=(0, 0, source), children=[]
			)
		else:
			self.root = root

		self.stack = [self.root]
		self.container = self.root
		self.token_view = self._TokenViewType(self)
		self.definitions = {}

	def consume_token(self) -> bool:
		self.tokens_i += 1
		return self.tokens_i < len(self.tokens)

	def consumer(self):
		# do first passthrough without consuming
		yield True
		while self.consume_token():
			yield True
		yield False

	def assert_(self, condition: bool, message: str = "") -> bool:
		"""
			Used when an breaking out would prevent the parser from continuing
			thus preventing more errors being reported or when a error is recoverable

			:meta private:
		"""
		if not(condition):
			stack = [c.pos for c in self.stack[1:]]
			debug("error:", message)
			try:
				self.errors.append(Error(message, self.token.pos,stack))
			except AssertionError:
				self.errors.append(Error(f"{message}(EOF)", self.tokens[-1].pos, stack))
		return condition

	@property
	def token(self):
		if self.tokens_i >= len(self.tokens):
			raise AssertionError("EOF")
		return self.tokens[self.tokens_i]

	@property
	def head(self):
		return self.stack[-1]

	def parse(self):
		"""
			Main method of the parser.
			Parses the tokens.
			The AST can be accessed via the ``root`` property.

			See :py:func:`parse`
		"""
		# we start out at root, since else this would
		# just call a method called `parse_root`
		# and that would be stupid

		# we are expecting either:
		# - a preprocessor
		# - a namespace
		# - a function definition
		# - a struct definition
		# - a statement

		consumer = self.consumer()

		while next(consumer):
			if self.token.type == Token.Type.PREPROCESSOR:
				self.parse_preprocessor()
				continue

			if self.token.type == Token.Type.KEYWORD:

				if self.token.value == "namespace":
					self.parse_namespace()
					continue

				if self.token.value == 'func':
					self.parse_func()
					continue

				if self.token.value in {'private','public'}:
					# this should be accessed later
					continue

				self.assert_(False, "Unexpected keyword")

			self.parse_statement()

	@_wrap
	def parse_preprocessor(self):
		# we expect a preprocessor thats either
		# - a #include
		# - a #define
		# something else entirely

		if self.token.value.startswith("#include"):
			# place the include in the root
			assert isinstance(self.container, AST.Root), "#include must be in root"
			self.root.includes.append(
				AST.Include(self.token.pos,self.token.value.removeprefix("#include "))
			)
			return

		if self.token.value.startswith("#define"):
			# place the define in the root
			assert isinstance(self.container, AST.Root), "#define must be in root"
			# lex the value
			tokens = tokenize(
				" ".join(self.token.value.split()[2:]), self.token.pos[2], self.token.pos
			)
			# parse the value
			sub_scan = type(self)(tokens, self.token.pos[2], self.root)
			value = sub_scan.parse_expression()
			self.errors.extend([
				(i,i.stack.append(self.token.pos))[0] for i in sub_scan.errors
			])

			self.definitions[self.token.value.split()[1]] = value

			return

		# if we get here, we have a preprocessor
		# that we don't know how to handle
		raise AssertionError(
			f"Unknown preprocessor: {self.token.value[1:].split()[0]}"
		)

	@_wrap
	def parse_namespace(self):
		# we expect a `namespace` keyword
		# then optionally a `public` or `private` keyword
		# then a name
		# then either `is` or/and the body

		# create the namespace
		# we will update it as we go
		namespace = AST.Namespace(
			pos=self.token.pos,
			name="<UNNAMED>",
			children=[],
		)
		# check if we have a access specifier
		if self.token_view[-1].type == Token.Type.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			namespace.private = self.token_view[-1].value == 'private'

		self.consume_token()

		# check if there is a default access specifier

		if self.token.type == Token.Type.KEYWORD \
			and self.token.value in {'public','private'}:
			namespace.default_private = self.token.value == 'private'
			self.consume_token()

		assert self.token.type == Token.Type.IDENTIFIER, "Expected namespace name"

		debug("ns", self.token.value)
		namespace.name = self.token.value

		self.consume_token()

		if self.token.value == "is":
			debug("namespace is")
			self.consume_token()
			assert self.token.type == Token.Type.IDENTIFIER, "Expected namespace name"
			namespace.is_ = self.token.value
			self.consume_token()
			# expect semicolon
			assert self.token.value == ";", "Expected semicolon"
			# self.consume_token()
			self.head.children.append(namespace)
			return

		assert self.token.type == Token.Type.BRACKET_OPEN, "Expected namespace body"
		self.consume_token()

		self.stack.append(namespace)

		consumer = self.consumer()
		while next(consumer):
			if self.token.type == Token.Type.BRACKET_CLOSE:
				# self.consume_token()
				break
			if self.token.type == Token.Type.KEYWORD:
				if self.token.value == "namespace":
					self.parse_namespace()
					continue
				if self.token.value == "func":
					self.parse_func()
					continue
				if self.token.value == "inline":
					self.parse_func()
					continue
				if self.token.value in {'private','public'}:
					# this should be accessed later
					continue

			self.parse_statement()
		else:
			assert False, "Expected closing bracket "

		self.stack.pop()

		self.head.children.append(namespace)

	@_wrap
	def parse_func(self):

		# construct the function
		# we will update it as we go
		func = AST.FuncDef(
			pos=self.token.pos,
			name="<UNNAMED>",
			children=[],
			args=[],
		)

		# check if we have a access specifier
		if self.token_view[-1].type == Token.Type.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			func.private = self.token_view[-1].value == 'private'

		# check if we have a inline kw
		if self.token.type == Token.Type.KEYWORD \
			and self.token.value == "inline":
			func.inline = True
			self.consume_token()

			if self.token.type == Token.Type.KEYWORD \
				and self.token.value in {'public','private'}:
				self.assert_(False, "Access specifier after the `inline` keyword.")
				self.consume_token()

		self.consume_token()

		assert self.token.type == Token.Type.IDENTIFIER, "Expected function name"
		func.name = self.token.value
		self.consume_token()

		# there should be arguments in brackets

		assert self.token.value == '(', "Expected opening bracket"
		self.consume_token()

		consumer = self.consumer()
		memory_specifier: str = ""
		while next(consumer):
			if self.token.value == ')':
				self.consume_token()
				break

			if self.token.type == Token.Type.MEMORY_MODIFIER:
				memory_specifier += self.token.value
				continue

			if self.token.type == Token.Type.IDENTIFIER:
				func.args.append(AST.Var(
					pos=self.token.pos,
					name=self.token.value,
					memory_specifier=memory_specifier,
				))
				memory_specifier = ""
				self.consume_token()
				self.assert_(
					self.token.value in {',', ')'},
					"Expected comma or closing bracket"
				)
				if self.token.value == ')':
					self.consume_token()
					break
				continue
			if self.token.type == Token.Type.COMMA:
				self.assert_(True, "Expected identifier")
				continue
			self.assert_(False,f"Unexpected token: {self.token.value}")
		else:
			assert False, "Expected closing parenthesis"

		# self.consume_token()

		assert self.token.type == Token.Type.BRACKET_OPEN, "Expected function body"
		self.consume_token()

		self.stack.append(func)

		consumer = self.consumer()

		while next(consumer):
			if self.token.type == Token.Type.BRACKET_CLOSE:

				# self.consume_token()
				break
			self.parse_statement()
		else:
			assert False, "Expected closing bracket "

		self.stack.pop()

		self.head.children.append(func)

	@_wrap
	def parse_statement(self):
		# this can be either:
		# - a variable declaration
		# - a variable assignment
		# - a function call
		# - raw assembly
		# - a return statement

		if self.token.type == Token.Type.KEYWORD:
			debug("kw:", self.token.value)
			if self.token.value == "return":
				self.parse_return()
				# expect a semi-colon
				self.assert_(self.token.value == ';', "Expected semi-colon")

				return
			if self.token.value == "if":
				self.parse_if()
				return
			if self.token.value == "while":
				self.parse_while()
				return
			if self.token.value == "for":
				debug("for")
				self.parse_for()
				return

			if self.token.value == "_asm_":
				self.parse_raw_asm()
				assert self.token.value == ';', "Expected semi-colon"
				return
			if self.token.value == "var":
				self.parse_var_def()
				return

		# if we get here, we have a function call or a variable assignment
		# we need to figure out which
		if self.token_view[1].type == Token.Type.MEMBER_SELECT:
			self.parse_member_select()

		if self.token_view[1].type == Token.Type.PAREN_OPEN and\
			self.token.type == Token.Type.IDENTIFIER:
			self.parse_func_call()
			# self.consume_token()
			assert self.token.value == ';', "Expected semi-colon"
			return
		# must be a variable assignment
		self.parse_var_assignment()
		# self.consume_token()

	@_wrap
	def parse_var_def(self):
		# construct the variable
		# we will update it as we go
		var = AST.VarDef(
			pos=self.token.pos,
			var=AST.Var(
				pos=self.token.pos,
				name="<UNNAMED>"
			),
			value=None,
		)

		# check if we have a access specifier
		if self.token_view[-1].type == Token.Type.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			var.private = self.token_view[-1].value == 'private'

		self.consume_token()

		while self.token.type == Token.Type.MEMORY_MODIFIER:
			if var.var.memory_specifier is None:
				var.var.memory_specifier = ""
			var.var.memory_specifier += self.token.value
			self.consume_token()

		assert self.token.type == Token.Type.IDENTIFIER, "Expected variable name"
		var.var.name = self.token.value
		self.consume_token()

		if self.token.type == Token.Type.SQ_BRACKET_OPEN:
			self.consume_token()
			var.offset = self.parse_expression()
			self.consume_token()
			self.assert_(self.token.value == ']', "Expected closing square bracket")
			self.consume_token()

		self.assert_(self.token.value in {'=',';'}, "Expected equals or semicolon")

		if self.token.value == '=':
			self.consume_token()
			var.value = self.parse_expression()
			self.consume_token()

		self.assert_(self.token.value == ';', "Expected semicolon")

		self.head.children.append(var)

	@_wrap
	def parse_var_assignment(self):
		var = AST.VarSet(
			pos=self.token.pos,
			l_value=AST.Var(
				pos=self.token.pos,
				name="<UNNAMED>"
			),
			value=None,  # type:ignore
		)
		if self.token.type == Token.Type.MEMORY_MODIFIER:
			var.l_value.memory_specifier = self.token.value  # type:ignore
			self.consume_token()

		if self.token.type == Token.Type.PAREN_OPEN:
			self.consume_token()
			expr = self.parse_expression()
			self.assert_(self.token.value == ')', "Expected closing parenthesis")
			if var.l_value.memory_specifier:  # type:ignore
				var.l_value = AST.MemoryModifier(
					self.token.pos,
					var.l_value.memory_specifier,  # type:ignore
					expr
				)
			else:
				var.l_value = expr

		else:
			assert self.token.type == Token.Type.IDENTIFIER, "Syntax error: " \
				f"unexpected token `{self.token.value}`"
			# the error is generic because parse_var_assignment is used
			# as a guard clause for parse_statement, which itself is used
			# as a guard clause for parse_namespace

			var.l_value.name = self.token.value  # type:ignore
			self.consume_token()

		if self.token.type == Token.Type.SQ_BRACKET_OPEN:
			self.consume_token()
			var.offset = self.parse_expression()
			self.consume_token()
			self.assert_(self.token.type == Token.Type.SQ_BRACKET_CLOSE,
				"Expected closing square bracket")
			self.consume_token()

		if self.token.type == Token.Type.MEMORY_MODIFIER:
			var.modifier = self.token.value
			self.consume_token()

		if self.token.type == Token.Type.MODIFIER:
			var.modifier = self.token.value
			self.consume_token()
			var.value = AST.Literal(var.pos, 1)

		else:
			if self.token.type == Token.Type.OPERATOR:
				var.modifier = self.token.value
				self.consume_token()

			self.assert_(self.token.value == '=', "Expected equals")
			self.consume_token()
			var.value = self.parse_expression()
			self.consume_token()

		# self.consume_token()
		self.assert_(self.token.value == ';', "Expected semicolon")

		self.head.children.append(var)

	@_wrap
	def parse_func_call(self):
		func = AST.FuncCall(
			pos=self.token.pos,
			name="<UNNAMED>",
			args=[],
		)

		func.name = self.token.value

		self.consume_token()
		self.assert_(self.token.value == '(', "Expected opening parenthesis")
		self.consume_token()

		if self.token.value != ')':
			func.args.append(self.parse_expression())
			self.consume_token()
			while self.token.value == ',':
				self.consume_token()
				func.args.append(self.parse_expression())
				self.consume_token()

		self.assert_(self.token.value == ')', "Expected closing parenthesis, got "
				f"`{self.token.value}`")
		self.consume_token()

		self.head.children.append(func)

	@_wrap
	def parse_expression(self) -> AST.Expression:
		# This can be:
		# - a variable reference
		# - a literal
		# - a function call
		# - a operation
		# - a parenthesized expression
		# - raw assembly
		# - `A` register query
		# - a literal array

		a: AST.Expression | None = None

		for _ in [0]:
			# check if we can scan ahead
			if len(self.tokens) > self.tokens_i:
				if self.token_view[1].type == Token.Type.MEMBER_SELECT:
					self.parse_member_select()

				if self.token_view[1].type == Token.Type.SQ_BRACKET_OPEN \
					and self.token.type == Token.Type.IDENTIFIER:
					a = self.parse_strong_array_ref()
					break

				if self.token_view[1].type == Token.Type.PAREN_OPEN \
					and self.token.type == Token.Type.IDENTIFIER:
					self.parse_func_call()
					a = self.head.children.pop()  # type:ignore
					self.tokens_i -= 1
					break

				if self.token_view[1].type == Token.Type.NUMBER \
					and self.token.value == '-':
					self.consume_token()
					self.token.value = f'-{self.token.value}'

			if self.token.type in {Token.Type.IDENTIFIER, Token.Type.MEMORY_MODIFIER}:
				if self.token.value in self.definitions:
					a = AST.DefineRef(self.token.pos,self.definitions[self.token.value])
					if a is None:
						a = AST.Var(pos=self.token.pos, name=self.token.value)
					with contextlib.suppress(AssertionError):
						self.consume_token()
						break
				else:
					a = self.parse_var_ref()
					break

			if self.token.type == Token.Type.BRACKET_OPEN:
				a = self.parse_literal_array()
				break

			if self.token.type == Token.Type.NUMBER:
				a = self.parse_number()
				break
			if self.token.type == Token.Type.STRING:
				a = self.parse_string()
				# with contextlib.suppress(AssertionError):
				# 	self.consume_token()
				break
			if self.token.type == Token.Type.KEYWORD \
				and self.token.value == '_getValueOfA_':
				a = self.parse_getValueOfA()
				break

			if self.token.type == Token.Type.KEYWORD and self.token.value == '_asm_':
				self.parse_raw_asm()
				a = self.head.children.pop()  # type:ignore
				break

			# if self.token.type == TokenType.PAREN_CLOSE:
			# 	assert False, "Unexpected closing parenthesis"

			if self.token.type == Token.Type.PAREN_OPEN:
				self.consume_token()
				a = self.parse_expression()
				if a is None:
					# enter a replacement for the empty expression
					a = AST.Expression(pos=self.token.pos)
				if self.assert_(self.token.value == ')', "Expected closing parenthesis"):
					self.consume_token()
				break
		assert a is not None, f"Expected expression, got {self.token.type}"

		# check if we can scan ahead
		if len(self.tokens) > self.tokens_i and self.token_view[1].type == Token.Type.SQ_BRACKET_OPEN:
			a = self.parse_weak_array_ref(a)

		with contextlib.suppress(AssertionError):
			a = self.parse_oper(a)

		return a

	def parse_oper(self, a):
		if self.token_view[1].type == Token.Type.OPERATOR:
			self.consume_token()
			op = self.token.value
			self.consume_token()
			b = self.parse_expression()
			self.consume_token()
			self.tokens_i -= 1
			a = AST.Operation(pos=a.pos, op=op, left=a, right=b)
			# check if the weight of the operation is higher than the current one
			if isinstance(b, AST.Operation) and oper_weight[b.op] < oper_weight[op]:
				a.right = b.left
				b.left = a
				a = b

		return a

	@_wrap
	def parse_getValueOfA(self):
		self.consume_token()
		if self.token.type == Token.Type.PAREN_OPEN:
			self.consume_token()
			self.assert_(self.token.value == ')', "Expected closing parenthesis")
			self.consume_token()
		return AST.GetValueOfA(pos=self.token.pos)

	@_wrap
	def parse_raw_asm(self):
		self.consume_token()
		if paren := self.token.type == Token.Type.PAREN_OPEN:
			self.consume_token()

		# we expect expressions separated by commas
		asm = AST.RawASM(self.token.pos,[])

		consumer = self.consumer()
		while next(consumer):
			if paren and self.token.value == ')':
				self.consume_token()
				break
			asm.arguments.append(self.parse_expression())
			self.consume_token()
			if self.token.value != ',':
				if paren and self.token.value == ')':
					self.consume_token()
					break
				self.assert_(
					not paren,
					"Expected closing parenthesis or comma"
					if paren else "Expected comma or a semicolon"
				)
				break
		else:
			if paren:
				self.assert_(False, "Expected closing parenthesis")
			else:
				self.assert_(not paren, "Expected semicolon")

		self.head.children.append(asm)

	@_wrap
	def parse_var_ref(self):
		memory_descriptor = ""
		while self.token.type == Token.Type.OPERATOR:
			memory_descriptor = self.token.value
			self.consume_token()
		return AST.Var(self.token.pos,self.token.value,memory_descriptor)

	@_wrap
	def parse_number(self):
		return AST.Literal(pos=self.token.pos, value=self.token.value)

	@_wrap
	def parse_string(self):
		return AST.Literal(pos=self.token.pos, value=self.token.value[1:-1])

	@_wrap
	def parse_member_select(self):
		assert self.token.type == Token.Type.IDENTIFIER, "Expected namespace name"
		while self.token_view[1].type == Token.Type.MEMBER_SELECT:
			assert self.token_view[2].type == Token.Type.IDENTIFIER,\
				"Expected member name"
			self.token.value += f"::{self.token_view[2].value}"
			del self.token_view[2]
			del self.token_view[1]

	@_wrap
	def parse_literal_array(self):
		self.consume_token()
		array = AST.LiteralArray(pos=self.token.pos, values=[])
		while self.token.value != '}':
			array.values.append(self.parse_expression())
			self.consume_token()
			if self.token.value != ',':
				self.assert_(self.token.value == '}', "Expected closing bracket")
				break
			self.consume_token()
		# self.consume_token()
		return array

	@_wrap
	def parse_return(self):
		self.consume_token()
		if self.token.type == Token.Type.SEMICOLON:
			self.head.children.append(AST.Return(pos=self.token.pos,
				value=None))
			return
		self.head.children.append(AST.Return(pos=self.token.pos,
			value=self.parse_expression()))
		self.consume_token()

	@_wrap
	def parse_if(self):
		self.consume_token()
		if_ = AST.If(pos=self.token.pos, condition=None, children=[])  # type:ignore
		assert self.token.value == '(', "Expected opening parenthesis"
		self.consume_token()
		if_.condition = self.parse_expression()
		self.consume_token()
		assert self.token.value == ')', "Expected closing parenthesis"
		self.consume_token()
		self.stack.append(if_)

		if self.token.value == '{':
			self.consume_token()
			while self.token.value != '}':
				self.parse_statement()
				self.consume_token()
			assert self.token.value == '}', "Expected closing bracket "

		else:
			self.parse_statement()
		self.stack.pop()

		self.head.children.append(if_)

	@_wrap
	def parse_while(self):
		self.consume_token()
		while_ = AST.While(
			pos=self.token.pos, condition=None, children=[])  # type:ignore
		assert self.token.value == '(', "Expected opening parenthesis"
		self.consume_token()
		while_.condition = self.parse_expression()
		assert self.token.value == ')', "Expected closing parenthesis"
		self.consume_token()
		self.stack.append(while_)
		if self.token.value == '{':
			self.consume_token()
			while self.token.value != '}':
				self.parse_statement()
				self.consume_token()
			assert self.token.value == '}', "Expected closing bracket "
		else:
			self.parse_statement()
		self.stack.pop()

		self.head.children.append(while_)

	@_wrap
	def parse_for(self):
		self.consume_token()
		for_ = AST.For(self.token.pos, [], "<UNNAMED>",None,None,None)  # type:ignore
		assert self.token.value == '(', "Expected opening parenthesis"
		self.consume_token()
		assert self.token.type == Token.Type.IDENTIFIER, "Expected identifier"
		for_.var = self.token.value
		self.consume_token()
		assert self.token.value == 'from', "Expected 'from'"
		self.consume_token()
		for_.start = self.parse_expression()
		self.consume_token()
		assert self.token.value == 'to', "Expected 'to'"
		self.consume_token()
		for_.end = self.parse_expression()
		self.consume_token()
		if self.token.value == 'by':
			self.consume_token()
			for_.step = self.parse_expression()
			self.parse_statement()
		assert self.token.value == ')', "Expected closing parenthesis"
		self.consume_token()
		self.stack.append(for_)

		debug("for")

		if self.token.value == '{':
			self.consume_token()
			while self.token.value != '}':
				self.parse_statement()
				self.consume_token()
			assert self.token.value == '}', "Expected closing bracket "

		else:
			self.parse_statement()
		self.stack.pop()

		self.head.children.append(for_)

	@_wrap
	def parse_strong_array_ref(self) -> AST.StrongArrayRef:
		array_ref = AST.StrongArrayRef(pos=self.token.pos,
			array=None, index=None)  # type:ignore
		array_ref.array = self.parse_var_ref()
		self.consume_token()
		# [<index>]
		assert self.token.value == '[', "Expected opening square bracket, "\
						f"got {self.token.type}"
		self.consume_token()
		array_ref.index = self.parse_expression()
		self.consume_token()
		assert self.token.value == ']', "Expected closing square bracket, "\
						f"got {self.token.type}"
		# self.consume_token()

		return array_ref

	def parse_weak_array_ref(self,array: AST.Expression) -> AST.WeakArrayRef:
		array_ref = AST.WeakArrayRef(pos=self.token.pos,
			array=array, index=None)
		# we assume that the `[` token is there
		# we consume two tokens because one is the array expression.
		self.consume_token()
		self.consume_token()
		array_ref.index = self.parse_expression()
		self.consume_token()
		assert self.token.value == ']', "Expected closing square bracket, "\
						f"got {self.token.type}"

		debug("done!")
		return array_ref


def parse(tokens: list[Token]) -> Monad[AST.Root]:
	"""
		wrapper around :py:meth:`Parser.parse`

		:param tokens: list of tokens
		:return: AST, list of errors
	"""
	parser = Parser(tokens, tokens[0].pos[2])
	parser.parse()

	return Monad(parser.root, parser.errors)


@dataclass
class ScannedFunction:
	"""
	It's a container for a function's name, its arguments, and its return type
	"""
	implementations: dict[int, AST.FuncDef]


TE = TypeVar('TE', bound=AST.Element)


def _wrapS(func: Callable[[Scanner, TE],T])\
		-> Callable[[Scanner, TE],T]:
	def wrapper(self: Scanner, elm: TE) -> T:
		try:
			return func(self, elm)
		except AssertionError as e:
			error = Error(pos=elm.pos, message=str(e))
			for i in self.namespace_stack:
				error.stack.append(i[1])
			self.errors.append(error)
			return None  # type:ignore
	return wrapper


AT = TypeVar('AT')


class Auto(Generic[AT]):
	_functions: dict[Type[AST.Element], Callable[[Any], Any]]

	def __init__(self):
		self._functions = {}

	def __call__(self,type: Type[TE])\
		-> Callable[[Callable[[AT, TE],T]],Callable[[AT, TE],T]]:
		def wrapper(func: Callable[[AT, TE],T])\
			-> Callable[[AT, TE],T]:
			self._functions[type] = func  # type:ignore
			return func
		return wrapper

	def __getitem__(self,type: Type[TE]) -> Callable[[AT, TE], Any]:
		return self._functions[type]  # type:ignore

	def __contains__(self, type):
		return type in self._functions


class Scanner:
	"""
		The scanner does:

		- dead code elimination
		- name resolution
		- namespace flattening
		- includes
	"""
	objects: dict[str, AST.VarDef | AST.FuncDef | ScannedFunction | AST.Namespace]
	is_private: dict[str,bool]
	to_be_checked: dict[str,AST.FuncDef]
	remote: bool
	errors: list[Error]
	namespace_stack: list[tuple[str,Pos]]
	accessed: dict[str, int]
	bucket: list[AST.Element]
	def_private: bool

	ScannedRoot = NewType('ScannedRoot', AST.Root)

	auto: Auto[Scanner] = Auto()

	class Namespace:
		def __init__(self, scanner: Scanner, elm_or_pos: AST.Container | Pos,
			name: str, override: bool = False):
			self.name = name
			if isinstance(elm_or_pos, AST.Container):
				self.pos = elm_or_pos.pos
			else:
				self.pos = elm_or_pos
			self.scanner = scanner
			self.override = override

		def __enter__(self):
			if self.override:
				self.old_ns = self.scanner.namespace_stack
				self.scanner.namespace_stack = [(self.name,self.pos)]
			else:
				self.scanner.namespace_stack.append((self.name,self.pos))

		def __exit__(self, exc_type, exc_val, exc_tb):
			if self.override:
				self.scanner.namespace_stack = self.old_ns
			else:
				self.scanner.namespace_stack.pop()

	def __init__(self, tree: AST.Root):
		self.tree = tree
		self.errors = []
		self.namespace_stack = []
		self.objects = {}
		self.is_private = {}
		self.to_be_checked = {}
		self.remote = True
		self.has_main = False
		self.bucket = []
		self.accessed = {}
		self.definitions = {}

	@property
	def namespace(self) -> str:
		return "::".join([i[0] for i in self.namespace_stack if i[0]])

	def scan(self) -> ScannedRoot:
		"""
			Main function of the scanner.
			After scanning the tree it returns it along with errors
		"""
		# add builtins
		self.def_private = True
		for i in builtins.get_builtins().values():
			self.tree.children.insert(0, i)
		self.scan_root(self.tree)
		if OPT < 3:
			for k, v in self.objects.items():
				if k == "Func:main:0":
					continue
				if k not in self.accessed or self.accessed[k] == 0:
					# find the element that defines the variable
					if v in self.tree.children:
						self.tree.children.remove(v)
						continue
					if isinstance(v, (AST.Namespace,ScannedFunction)):
						continue

		return cast(Scanner.ScannedRoot,self.tree)

	@classmethod
	def do(cls, root: AST.Root) -> Monad[ScannedRoot]:
		"""
			wrapper around :py:meth:`Scanner.scan`
		"""
		scanner = Scanner(root)
		return Monad(scanner.scan(), scanner.errors)

	def jns(self, ns1: str, ns2: str | None = None) -> str:
		if ns2 is None:
			return f"{self.namespace}::{ns1}".removeprefix("::")
		return f"{ns1}::{ns2}".removeprefix("::")

	def get(self, name: str, namespace: str | None = None) -> str | None:
		name_l = name.split("::")
		ns = self.namespace if namespace is None else namespace
		# check if the name is in the current namespace
		if f"{ns}::{name_l[0]}" in self.objects and (
				not self.is_private.get(f"{ns}::{name_l[0]}") or namespace is None
			):
			if len(name_l) == 1:
				return self.jns(ns, name_l[0])
			r = self.get(
				"::".join(name_l[1:]),
				f"{ns}::{name_l[0]}".removeprefix("::")
			)
			if r is not None:
				return self.jns(ns,r)
		if namespace is not None:
			return None
		# check if the name is in namespaces upwards
		while "::" in ns:
			ns = ns[:ns.rfind("::")]
			if f"{ns}::{name_l[0]}" in self.objects:
				if len(name_l) == 1:
					return self.jns(ns, name_l[0])
				r = self.get(
					"::".join(name_l[1:]),
					f"{ns}::{name_l[0]}".removeprefix("::")
				)
				if r is not None:
					return self.jns(ns,r)
		if len(name_l) == 1:
			return name_l[0]
		r = self.get(
			"::".join(name_l[1:]),
			name_l[0]
		)
		return r

	def update_private(self, elm: AST.Element):
		if not hasattr(elm, "private"):
			return
		if elm.private is None:  # type:ignore
			elm.private = self.def_private  # type:ignore

	@auto(AST.Root)
	@_wrapS
	def scan_root(self, root: AST.Root):
		remote = self.remote
		self.remote = False
		for include in root.includes:
			self.scan_include(include)
		self.remote = remote
		self.scan_container(root)

	@_wrapS
	def scan_include(self, include: AST.Include):
		# locate the file
		include_val = include.value
		if include_val in self.tree.resolved_includes:
			return  # already included
		self.tree.resolved_includes.append(include_val)
		if include_val[0] == "<" and include_val[-1] == ">":
			include_val = include_val[1:-1]
			for i_path in INCLUDE_PATH:
				path = i_path / f"{include_val}.sc"
				if path.exists():
					break
			else:
				assert False, f"Unable to locate library <{include_val}>"
		elif include_val[0] == "\"" and include_val[-1] == "\"":
			include_val = include_val[1:-1]
			path = Path() / include_val
			assert path.exists(), f"Unable to locate file \"{include_val}\""
		else:
			raise AssertionError(f"Invalid include: {include_val}")

		# parse the file
		with path.open() as f:
			tokens = tokenize(f.read(),str(path))
			monad = parse(tokens)
			for error in monad.errors:
				error.stack.insert(0,include.pos)
			self.errors.extend(monad.errors)

			# place the contents into root
			for include in monad.value.includes:
				self.scan_include(include)
			for i in monad.value.children:
				self.tree.children.insert(0,i)

	def scan_auto(self, elm: AST.Element):
		# assert elm.__class__ in self.auto, f"No auto-scanner for {elm.__class__}"
		if elm.__class__ not in self.auto:
			info(f"No auto-scanner for {elm.__class__}")
			return
		return self.auto[elm.__class__](self, elm)

	def scan_container(self, elm: AST.Container):
		p_bucket = self.bucket
		self.bucket = []
		namespaces: list[AST.Namespace] = []
		for child in elm.children:
			self.scan_auto(child)
			if isinstance(child, AST.Namespace):
				namespaces.append(child)
		for namespace in namespaces:
			elm.children.remove(namespace)
		elm.children.extend(self.bucket)
		self.bucket = p_bucket

	@auto(AST.FuncDef)
	@_wrapS
	def scan_function(self, func: AST.FuncDef):

		self.update_private(func)

		if func.name == "main":
			if not self.has_main and len(func.args) == 0 and not func.private:
				self.remote = False
				self.has_main = True
			else:
				func.name = "0main"

		if f"{self.namespace}::{func.name}".removeprefix("::") in self.objects:
			func_obj = self.objects[f"{self.namespace}::{func.name}".removeprefix("::")]
			assert isinstance(func_obj, ScannedFunction),\
				f"Namespace collision: {func.name} for {len(func.args)} args"
			if not func.private \
				and self.is_private[f"{self.namespace}::{func.name}".removeprefix("::")]:
				self.is_private[f"{self.namespace}::{func.name}".removeprefix("::")] \
					= False

		else:
			func_obj = ScannedFunction(implementations={})
			self.objects[f"{self.namespace}::{func.name}".removeprefix("::")] = func_obj
			if func.private:
				self.is_private[f"{self.namespace}::{func.name}".removeprefix("::")] = True

		# TODO: Implement complex signatures
		assert len(func.args) not in func_obj.implementations, \
			f"Duplicate implementation for {func.name}(\
				{', '.join(arg.name for arg in func.args)})"

		if func.name == "main":
			self.objects[f"Func:{func.name}:0"] = func
		else:
			self.objects[self.jns(f"Func:{func.name}:{len(func.args)}")] = func
			if func.private:
				self.is_private[self.jns(f"Func:{func.name}:{len(func.args)}")] = True

		func_obj.implementations[len(func.args)] = func

		func.name = f"Func:{func.name}:{len(func.args)}"

		for var in func.args:
			var.name = f"{func.name}::{var.name}"
			# construct vardef
			vardef = AST.VarDef(
				pos=var.pos,
				var=var,
				value=None
			)
			self.objects[self.jns(var.name)] = vardef

		if self.remote:
			self.to_be_checked[self.jns(func.name)] = func
			debug(f"{self.jns(func.name)} is to be checked")
		else:
			with self.Namespace(self, func, func.name):
				self.scan_container(func)
				# pseudo dump
				for child in func.children:
					if isinstance(child, AST.VarDef):
						child.var.name = f"{func.name}::{child.var.name}"
		if func.name == "main":
			self.remote = False

	@auto(AST.Namespace)
	@_wrapS
	def scan_namespace(self, namespace: AST.Namespace):
		if namespace.name in self.objects:
			raise AssertionError(f"Duplicate namespace {namespace.name}")
		self.objects[self.jns(namespace.name)] = namespace
		self.update_private(namespace)
		if namespace.private:
			self.is_private[self.jns(namespace.name)] = True
		p_def_priv = self.def_private
		if namespace.default_private is not None:
			self.def_private = namespace.default_private
		with self.Namespace(self, namespace, namespace.name):  # type: ignore
			self.scan_container(namespace)  # type: ignore

		if namespace.is_ is not None:
			debug(f"{namespace.name} -> {namespace.is_}")
			# find the namespace
			ns = self.objects.get(self.get(namespace.is_))  # type: ignore
			assert isinstance(ns, AST.Namespace), f"{namespace.is_} is not a namespace"
			# convert all inner children to use the new namespace
			copy_queue: list[AST.Element] = []
			for child in ns.children:
				child = deepcopy(child)  # make a copy to avoid changing the original
				copy_queue.append(child)

			for item in copy_queue:
				self.bucket.append(item)
				if isinstance(item, AST.VarDef):
					item.var.name = item.var.name.replace(ns.name, namespace.name, 1)
					# register the function
					self.objects[self.jns(item.var.name)] = item
					continue
				if isinstance(item, AST.FuncDef):
					self.rescan_func(namespace, ns, copy_queue, item)
					continue
		# dump
		for child in namespace.children:
			self.bucket.append(child)
			if isinstance(child,(AST.FuncDef, AST.Namespace))\
					and child.name != "Func:main:0":
				child.name = f"{namespace.name}::{child.name}"
			if isinstance(child, AST.VarDef):
				child.var.name = f"{namespace.name}::{child.var.name}"
			if isinstance(child, AST.FuncDef):
				for g_child in child.children:
					if isinstance(g_child, AST.VarDef):
						g_child.var.name = f"{namespace.name}::{g_child.var.name}"

		self.def_private = p_def_priv

	def rescan_func(self, namespace, ns, copy_queue, item):
		item.name = item.name.replace(ns.name, namespace.name, 1)
		debug(f"func {item.name}", ns.name, namespace.name)
		self.objects[self.jns(item.name)] = item
		for arg in item.args:
			arg.name.replace(ns.name, namespace.name, 1)
		copy_queue.extend(iter(item.children))

	@auto(AST.VarDef)
	@_wrapS
	def scan_variable(self, var: AST.VarDef):
		if self.jns(var.var.name) in self.objects:
			raise AssertionError(f"Duplicate variable {var.var.name}")
		self.objects[self.jns(var.var.name)] = var
		self.update_private(var)
		if var.private:
			self.is_private[self.jns(var.var.name)] = True
		if var.offset is not None:
			self.scan_auto(var.offset)
		if var.value is not None:
			self.scan_auto(var.value)

	@auto(AST.FuncCall)
	@_wrapS
	def scan_call(self, call: AST.FuncCall):
		# get the function
		if call.name == "main":
			call.name = "0main"

		func = self.get(
			(
				f"{'::'.join(call.name.split('::')[:-1])}::"
				f"Func:{call.name.split('::')[-1]}:{len(call.args)}"
			).removeprefix("::")
		)

		if func is None or func not in self.objects:
			func = self.get(call.name)
			if func is None or func not in self.objects:
				if call.name == "0main":
					raise AssertionError(
						"Name error: \"main\"."
						" Note: main function cannot be called."
					)
				raise AssertionError(f"Name error: \"{call.name}\"")
			assert isinstance(self.objects[func], ScannedFunction),\
				f"\"{call.name}\" is not a function"
			raise AssertionError(
				f"\"{call.name}\" cannot be bound to {len(call.args)} arguments."
			)

		debug(f"call {call.name}")

		if func in self.accessed:
			self.accessed[func] += 1
		else:
			self.accessed[func] = 1
		if func in self.to_be_checked:
			debug(f"{call.name} is called for the #1 time")
			# set namespace to the function's namespace
			pos = self.objects[func].pos  # type:ignore
			the_func = self.to_be_checked[func]
			del self.to_be_checked[func]
			with self.Namespace(self, pos, func, True):
				self.scan_container(the_func)
		call.name = func

		for child in call.args:
			self.scan_auto(child)

	@auto(AST.VarSet)
	@_wrapS
	def scan_set(self, set: AST.VarSet):
		self.scan_auto(set.l_value)
		if set.offset is not None:
			if isinstance(set.offset, AST.Literal) and set.offset.value in {"0","0.",".0","0.0"} and OPT <= 1:
				set.offset = None
			else:
				self.scan_auto(set.offset)
		self.scan_auto(set.value)

	@auto(AST.Var)
	@_wrapS
	def scan_var(self, var: AST.Var):
		name = self.get(var.name)
		assert name is not None, f"Name error: \"{var.name}\""
		debug("name=",name)
		assert self.objects.get(name) is not None, f"Name error: \"{var.name}\""
		var.name = name
		if var.name in self.accessed:
			self.accessed[var.name] += 1
		else:
			self.accessed[var.name] = 1

	@auto(AST.If)
	@_wrapS
	def scan_if(self, if_: AST.If):
		self.scan_auto(if_.condition)
		self.scan_container(if_)

	@auto(AST.While)
	@_wrapS
	def scan_while(self, while_: AST.While):
		self.scan_auto(while_.condition)
		self.scan_container(while_)

	@auto(AST.For)
	@_wrapS
	def scan_for(self, for_: AST.For):
		self.scan_auto(for_.start)
		self.scan_auto(for_.end)
		if for_.step is not None:
			self.scan_auto(for_.step)
		# define the variable
		for_.var = self.jns(for_.var)
		assert for_.var not in self.objects, \
			f"Duplicate variable '{for_.var}'"
		self.objects[for_.var] = AST.VarDef(
			for_.pos, AST.Var(for_.pos,for_.var), for_.start, private=True)

		self.scan_container(for_)

	@auto(AST.RawASM)
	@_wrapS
	def scan_asm(self, asm: AST.RawASM):
		for child in asm.arguments:
			self.scan_auto(child)

	@auto(AST.Return)
	@_wrapS
	def scan_return(self, ret: AST.Return):
		if ret.value is not None:
			debug("return value")
			self.scan_auto(ret.value)

	@auto(AST.Operation)
	@_wrapS
	def scan_operation(self, op: AST.Operation):
		self.scan_auto(op.left)
		self.scan_auto(op.right)

	@auto(AST.Literal)
	def scan_literal(self, lit: AST.Literal):
		pass

	@auto(AST.DefineRef)
	def scan_define_ref(self, ref: AST.DefineRef):
		with self.Namespace(self, ref.pos, ""):
			self.scan_auto(ref.expr)

	@auto(AST.StrongArrayRef)
	def scan_strong_array_ref(self, ref: AST.StrongArrayRef):
		self.scan_auto(ref.array)
		self.scan_auto(ref.index)

	@auto(AST.WeakArrayRef)
	def scan_weak_array_ref(self, ref: AST.WeakArrayRef):
		self.scan_auto(ref.array)
		self.scan_auto(ref.index)

	@auto(AST.LiteralArray)
	def scan_literal_array(self, arr: AST.LiteralArray):
		for child in arr.values:
			self.scan_auto(child)


def _wrapA(func):
	def wrapper(self, elm: AST.Element):
		try:
			return func(self, elm)
		except AssertionError as e:
			self.assert_(False, e.args[0], elm.pos)
	return wrapper


SHORT_NAME_CHARS = printable


def short_name_generator(min_size: int = 1,prefix="") -> Iterator[str]:
	if min_size <= 1:
		yield from (prefix+i for i in SHORT_NAME_CHARS)
	n = short_name_generator(min_size - 1)
	while True:
		for i in SHORT_NAME_CHARS:
			yield from (prefix+i+j for j in n)


post_assembler_optimizations = [
	(
		[
			"loadAtVar",
			"<1>",
			"storeAtVar",
			"<1>",
		],
		[
			"loadAtVar",
			"<1>",
		]
	),
	(
		[
			"loadAtVar",
			"<1>",
			"storeAtVar",
			"<2>",
			"storeAtVar",
			"<1>",
		],
		[
			"loadAtVar",
			"<1>",
			"storeAtVar",
			"<2>",
		]
	),
	(
		[
			"storeAtVar",
			"<1>",
			"loadAtVar",
			"<1>",
		],
		[
			"storeAtVar",
			"<1>"
		]
	)
]


oper_map = {
	"+":  "addWithVar",
	"-":  "subWithVar",
	"*":  "mulWithVar",
	"/":  "divWithVar",
	"%":  "modWithVar",
	"<<": "bitwiseLsfWithVar",
	">>": "bitwiseRsfWithVar",
	"&":  "bitwiseAndWithVar",
	"|":  "bitwiseOrWithVar",
	"..": "join",
	"&&": "boolAndWithVar",
	"||": "boolOrWithVar",
	"==": "boolEqualsWithVar",
	">=": "largerThanOrEqualWithVar",
	"<=": "smallerThanOrEqualWithVar",
	"!=": "boolNotEqualsWithVar",
	">":  "largerThanWithVar",
	"<":  "smallerThanWithVar",
}


class Assembler:
	"""
		Takes in the tree after scanning and creates the assembly code.
	"""
	errors: list[Error]
	lines: list[str]
	labels_to_be_defined: list[str]
	labels: dict[str, int]
	functions: dict[str, AST.FuncDef]

	sng: Iterator[str]
	ovg: Iterator[str]

	ret_as_jump: int | str | None

	var_lookup: dict[str,str]
	op_vars: list[str]
	op_depth: int

	end: Callable[[], None]

	auto: Auto[Assembler] = Auto()

	arr_var: str | None
	arr_ptr: bool = False

	def __init__(self, tree: Scanner.ScannedRoot) -> None:
		self.tree = tree
		self.lines = []
		self.errors = []
		self.in_main = False
		self.labels = {}
		self.labels_to_be_defined = []
		self.functions = {}
		self.ret_as_jump = None
		self.op_vars = []

	def assemble(self) -> str:
		"""
		This function generates the assembly as a string
		and returns it with a list of errors

		:return: A tuple of a string containing the assembly and a list of errors.
		"""

		self.arr_var = None
		self.end = lambda: None
		if DEBUG:
			self.init_debug()
		self.var_lookup = {}
		self.sng = short_name_generator(prefix="%")
		self.ovg = short_name_generator(prefix="o" if OPT == 0 else "%Oper_")
		self.op_depth = 0

		main_id, main = self.find_main(self.tree)
		# set global variables
		for child in self.tree.children:
			if isinstance(child, AST.VarDef):
				if child.offset is not None:
					self.lines.append("createArray")
					self.lines.append(self.var(child.var.name))
				if child.value is not None:
					self.assemble_auto(child.value)
					self.lines.append("storeAtVar")
					self.lines.append(self.var(child.var.name))
			if isinstance(child, AST.FuncDef):

				self.functions[child.name] = child

		if main is not None:
			self.in_main = True
			self.assemble_container(main)
			self.in_main = False

			# incase the main function doesn't have a return statement
			self.lines.append("done")

		for child in self.tree.children:
			if isinstance(child, AST.FuncDef) and child.name != main_id \
					and not child.inline:
				# create label
				self.labels[f":{child.name}"] = len(self.lines)
				end = self.symbol(child.name)
				if f":{child.name}" in self.labels_to_be_defined:
					self.labels_to_be_defined.remove(f":{child.name}")
				self.assemble_container(child)
				if self.lines[-1] != "ret":
					end()
					self.lines.append("ret")

		for label in self.labels_to_be_defined:
			self.errors.append(Error(
				f"Internal assembler error: Label {label} is not defined",
				self.tree.pos
				)
			)

		if OPT < 2:
			self.post_asm_opt()

		for i,line in enumerate(self.lines):
			if line.startswith(":"):
				self.lines[i] = f"{self.labels.get(line,0)}"

		if self.op_depth != 0:
			debug(f"Op_depth is {self.op_depth}, while it should be 0")

		return "\n".join(self.lines)

	def init_debug(self):
		self.lines.append("ldi")
		self.lines.append("main")
		self.lines.append("storeAtVar")
		self.lines.append("$SYMBOL")
		self.lines.append("ldi")
		self.lines.append("0")

	def symbol(self, name: str) -> Callable[[], None]:
		if not DEBUG:
			return lambda: None

		if self.op_depth == len(self.op_vars):
			self.op_vars.append(next(self.ovg))
		p_var = self.op_vars[self.op_depth]
		self.op_depth += 1
		if self.op_depth == len(self.op_vars):
			self.op_vars.append(next(self.ovg))
		var = self.op_vars[self.op_depth]
		self.lines.append("storeAtVar")
		self.lines.append(var)
		self.lines.append("loadAtVar")
		self.lines.append("$SYMBOL")
		self.lines.append("storeAtVar")
		self.lines.append(p_var)
		self.lines.append("ldi")
		self.lines.append(name)
		self.lines.append("storeAtVar")
		self.lines.append("$SYMBOL")
		self.lines.append("loadAtVar")
		self.lines.append(var)
		def end():
			if self.op_depth == len(self.op_vars):
				self.op_vars.append(next(self.ovg))
			var = self.op_vars[self.op_depth]
			self.lines.append("storeAtVar")
			self.lines.append(var)
			self.lines.append("loadAtVar")
			self.lines.append(p_var)
			self.lines.append("storeAtVar")
			self.lines.append("$SYMBOL")
			self.lines.append("loadAtVar")
			self.lines.append(var)
			self.end = p_end

		p_end = self.end
		self.end = end
		return end

	@classmethod
	def do(cls, tree: Scanner.ScannedRoot) -> Monad[str]:
		asm = cls(tree)
		return Monad(asm.assemble(),asm.errors)

	def post_asm_opt(self):
		for i,_ in enumerate(self.lines):
			for pattern, replacement in post_assembler_optimizations:
				repl_vars = {}
				for a,b in zip(pattern,self.lines[i:]):
					if a.startswith("<") and a.endswith(">"):
						if a not in repl_vars:
							repl_vars[a] = b
							continue
						if repl_vars[a] != b:
							break
						continue
					if a != b:
						break
				else:
					self.lines[i:i+len(pattern)] = replacement
					for j in range(len(replacement)):
						if replacement[j] in repl_vars:
							self.lines[i+j] = repl_vars[replacement[j]]
					for label, v in self.labels.items():
						if v > i:
							self.labels[label] = v + len(replacement) - len(pattern)
		if self.lines[-1] == "done":
			self.lines.pop()

	def var(self,name:str) -> str:
		if OPT != 0:
			return name
		if name not in self.var_lookup:
			self.var_lookup[name] = next(self.sng)
		return self.var_lookup[name]

	def op_var(self) -> str:
		if self.op_depth == len(self.op_vars):
			self.op_vars.append(next(self.ovg))
		self.op_depth += 1
		return self.op_vars[self.op_depth - 1]

	@_wrapA
	def find_main(self, _) -> tuple[str, AST.FuncDef | None]:
		main_id = "Func:main:0"
		main = next(
			(
				child for child in self.tree.children
				if isinstance(child, AST.FuncDef) and child.name == main_id
			),
			None
		)
		self.assert_(main is not None, "No main function found", self.tree.pos)
		return main_id, main

	def assert_(self, condition: bool, message: str, pos: Pos) -> None:
		if not condition:
			self.errors.append(Error(message, pos))

	def assemble_auto(self, elm: AST.Element) -> None:
		if type(elm) not in self.auto._functions:
			info(f"{type(elm)} is not auto-assembled")
			return
		self.auto._functions[type(elm)](self,elm)  # type:ignore

	def assemble_container(self, container: AST.Container) -> None:
		for child in container.children:
			self.assemble_auto(child)

	@_wrapA
	@auto(AST.VarDef)
	def assemble_var_def(self, var: AST.VarDef) -> None:
		name = self.var(var.var.name)
		if var.offset is not None:
			offset = self.op_var()
			self.assemble_auto(var.offset)
			self.lines.append("storeAtVar")
			self.lines.append(offset)
			self.lines.append("createArray")
			self.lines.append(name)
			self.lines.append(offset)
			self.op_depth -= 1

		if var.value is not None:
			if isinstance(var.value, AST.LiteralArray):
				self.arr_var = name
				self.assemble_auto(var.value)
				self.arr_var = None
			else:
				self.assemble_auto(var.value)
				self.lines.append("storeAtVar")
				self.lines.append(name)

	@_wrapA
	@auto(AST.RawASM)
	def assemble_asm(self, asm: AST.RawASM) -> None:
		for i in asm.arguments:
			if isinstance(i, AST.Literal):
				self.lines.append(f"{i.value}")
			elif isinstance(i, AST.Var):
				self.lines.append(self.var(i.name))
			else:
				self.assert_(False, "Invalid expression", i.pos)

	@_wrapA
	@auto(AST.FuncCall)
	def assemble_func_call(self, call: AST.FuncCall) -> None:

		if self.functions[call.name].inline:
			self.assemble_inline_function(call)
			return

		func = self.functions[call.name]
		for arg, val in zip(func.args, call.args):
			if arg.memory_specifier == "$":
				self.assert_(
					isinstance(val, AST.Var),
					f"Argument {arg.name} is a reference thus variable was expected",
					val.pos
				)
				# TODO: bind
			else:
				self.assemble_auto(val)
				if not arg.name.endswith("!"):
					self.lines.append("storeAtVar")
					self.lines.append(self.var(arg.name))
		self.lines.append("jts")
		if call.name not in self.labels_to_be_defined:
			self.labels_to_be_defined.append(f":{call.name}")
		self.lines.append(f":{call.name}")

	@_wrapA
	def assemble_inline_function(self, call: AST.FuncCall) -> None:
		end = self.symbol(call.name)
		func = self.functions[call.name]
		# bind arguments
		for arg, val in zip(func.args, call.args):
			if arg.memory_specifier == "$":
				self.assert_(
					isinstance(val, AST.Var),
					f"Argument {arg.name} is a reference thus variable was expected",
					val.pos
				)
				# TODO: bind
			else:
				self.assemble_auto(val)
				if not arg.name.endswith("!"):
					self.lines.append("storeAtVar")
					self.lines.append(self.var(arg.name))

		p_raj = self.ret_as_jump

		self.ret_as_jump = f":{call.name}:{len(self.lines)}"

		self.assemble_container(func)

		if self.lines[-1] == self.ret_as_jump:
			self.lines.pop()
			self.lines.pop()
		self.labels[self.ret_as_jump] = len(self.lines)
		if self.ret_as_jump in self.labels_to_be_defined:
			self.labels_to_be_defined.remove(self.ret_as_jump)
		self.ret_as_jump = p_raj
		end()

	@_wrapA
	@auto(AST.Literal)
	def assemble_literal(self, literal: AST.Literal) -> None:
		self.lines.append("ldi")
		self.lines.append(f"{literal.value}")

	@auto(AST.Return)
	def assemble_return(self, ret: AST.Return) -> None:
		if ret.value is not None:
			self.assemble_auto(ret.value)
		self.end()
		if self.ret_as_jump is None:
			self.lines.append("ret")
			return
		self.lines.append("jmp")
		if isinstance(self.ret_as_jump,str)\
			and self.ret_as_jump not in self.labels_to_be_defined:
			self.labels_to_be_defined.append(self.ret_as_jump)
		self.lines.append(f"{self.ret_as_jump}")

	@auto(AST.Var)
	def assemble_var(self, var: AST.Var) -> None:
		name = self.var(var.name)
		self.lines.append("loadAtVar")
		self.lines.append(name)

	@auto(AST.If)
	@_wrapA
	def assemble_if(self, if_: AST.If) -> None:
		label = f":if:{len(self.lines)}"
		self.assemble_auto(if_.condition)
		self.lines.append("jf")
		self.lines.append(label)
		self.assemble_container(if_)
		self.labels[label] = len(self.lines)

	@auto(AST.While)
	@_wrapA
	def assemble_while(self, while_: AST.While) -> None:
		label_start = f":while:{len(self.lines)}"
		label_end = f":while_end:{len(self.lines)}"
		# condition
		self.labels[label_start] = len(self.lines)
		self.assemble_auto(while_.condition)
		self.lines.append("jf")
		self.lines.append(label_end)

		self.assemble_container(while_)
		self.lines.append("jmp")
		self.lines.append(label_start)

		self.labels[label_end] = len(self.lines)

	@auto(AST.Operation)
	@_wrapA
	def assemble_operation(self, op: AST.Operation) -> None:
		self.assemble_auto(op.left)
		if self.op_depth == len(self.op_vars):
			self.op_vars.append(next(self.ovg))
		self.lines.append("storeAtVar")
		self.lines.append(self.op_vars[self.op_depth])
		self.op_depth += 1
		self.assemble_auto(op.right)
		self.op_depth -= 1
		assert op.op in oper_map, "Unknown operation"
		self.lines.append(oper_map[op.op])
		self.lines.append(self.op_vars[self.op_depth])

	@auto(AST.DefineRef)
	def assemble_define_ref(self, ref: AST.DefineRef) -> None:
		self.assemble_auto(ref.expr)

	@auto(AST.For)
	@_wrapA
	def assemble_for(self, for_: AST.For) -> None:
		label_start = f":for:{len(self.lines)}"
		label_end = f":for_end:{len(self.lines)}"
		var = self.var(for_.var)
		self.assemble_auto(for_.start)
		self.lines.append("storeAtVar")
		self.lines.append(var)

		# create end and step vars
		if len(self.op_vars) == self.op_depth:
			self.op_vars.append(next(self.ovg))
		step_var = self.op_vars[self.op_depth]
		self.op_depth += 1
		if len(self.op_vars) == self.op_depth:
			self.op_vars.append(next(self.ovg))
		end_var = self.op_vars[self.op_depth]

		self.op_depth += 1
		self.assemble_auto(for_.end)
		self.lines.append("storeAtVar")
		self.lines.append(end_var)
		if for_.step is not None:
			self.assemble_auto(for_.step)
		else:
			self.lines.append("ldi")
			self.lines.append("1")
		self.lines.append("storeAtVar")
		self.lines.append(step_var)

		# condition
		self.labels[label_start] = len(self.lines)
		# check if we need to increment
		self.lines.append("loadAtVar")
		self.lines.append(var)
		self.lines.append("subWithVar")
		self.lines.append(end_var)
		self.lines.append("jt")
		self.lines.append(label_end)

		self.assemble_container(for_)

		self.lines.append("loadAtVar")
		self.lines.append(var)
		self.lines.append("addWithVar")
		self.lines.append(step_var)
		self.lines.append("storeAtVar")
		self.lines.append(var)

		self.lines.append("jmp")
		self.lines.append(label_start)

		self.labels[label_end] = len(self.lines)
		self.op_depth -= 1

	@auto(AST.VarSet)
	def assemble_var_set(self, var_set: AST.VarSet) -> None:
		if isinstance(var_set.l_value, AST.Var) and var_set.offset is None:
			if var_set.modifier == "++":
				self.lines.append("inc")
				self.lines.append(self.var(var_set.l_value.name))
				return
			elif var_set.modifier == "--":
				self.lines.append("dec")
				self.lines.append(self.var(var_set.l_value.name))
				return
		elif var_set.modifier in {"--","++"}:
			var_set.modifier = var_set.modifier[0]

		if var_set.offset is not None:
			# create offset var
			if len(self.op_vars) == self.op_depth:
				self.op_vars.append(next(self.ovg))
			offset_var = self.op_vars[self.op_depth]
			self.op_depth += 1
			self.assemble_auto(var_set.offset)
			self.lines.append("storeAtVar")
			self.lines.append(offset_var)
		else:
			offset_var = "N/A"

		self.assemble_auto(var_set.value)
		if var_set.modifier is not None:
			if self.op_depth == len(self.op_vars):
				self.op_vars.append(next(self.ovg))
			op_var = self.op_vars[self.op_depth]
			self.op_depth += 1
			self.lines.append("storeAtVar")
			self.lines.append(op_var)
			self.assemble_auto(var_set.value)
			self.lines.append(oper_map[var_set.modifier])
			self.lines.append(op_var)

		if isinstance(var_set.l_value, AST.Var):
			if var_set.offset is None:
				self.lines.append("storeAtVar")
				self.lines.append(self.var(var_set.l_value.name))
			else:
				self.lines.append("storeAtVarWithOffset")
				self.lines.append(self.var(var_set.l_value.name))
				self.lines.append(offset_var)
				self.op_depth -= 1

		else:
			info("TODO: implement l_value")

	@auto(AST.StrongArrayRef)
	def assemble_strong_array_ref(self, array_ref: AST.StrongArrayRef) -> None:
		if isinstance(array_ref.index,AST.Literal) and array_ref.index.value in {"0.0","0",".0","0."} and OPT <= 1:
			self.lines.append("loadAtVar")
			self.lines.append(self.var(array_ref.array.name))
			return
		index_var = self.op_var()
		self.assemble_auto(array_ref.index)
		self.lines.append("storeAtVar")
		self.lines.append(index_var)
		self.lines.append("loadAtVarWithOffset")
		self.lines.append(self.var(array_ref.array.name))
		self.lines.append(index_var)
		self.op_depth -= 1

	@auto(AST.WeakArrayRef)
	def assemble_weak_array_ref(self, array_ref: AST.WeakArrayRef) -> None:
		# a weak array ref is just pointer stuff :D
		# get the pointer
		self.assemble_auto(array_ref.array)

		# some opt for index values
		if isinstance(array_ref.index,AST.Literal) and OPT <= 1:
			index = int(float(array_ref.index.value))
			# we'll be nice and round the literal :)
			if index <= 6:  # it takes 6 lines do load and add a number
				for _ in range(index):
					self.lines.append("incA")
				self.lines.append("getValueAtPointerOfA")
				return


		arr = self.op_var()
		self.lines.append("storeAtVar")
		self.lines.append(arr)
		# get the index
		self.assemble_auto(array_ref.index)
		self.lines.append("addWithVar")
		self.lines.append(arr)
		self.op_depth -= 1
		self.lines.append("getValueAtPointerOfA")
		return

	@auto(AST.LiteralArray)
	def assemble_literal_array(self, literal_array: AST.LiteralArray) -> str:
		if self.arr_var is None:
			arr_var = self.op_var()
		elif not self.arr_ptr:
			arr_var = self.arr_var
			self.arr_var = None
			size_var = self.op_var()

			# define and allocate the array
			self.lines.append("ldi")
			self.lines.append(str(len(literal_array.values)))
			self.lines.append("storeAtVar")
			self.lines.append(size_var)

			self.lines.append("createArray")
			self.lines.append(arr_var)
			self.lines.append(size_var)

			self.op_depth -= 1
		else:
			arr_var = self.arr_var
			self.arr_var = None

		# store the values
		index = self.op_var()
		for i, value in enumerate(literal_array.values):
			if not self.arr_ptr:
				self.lines.append("ldi")
				self.lines.append(str(i))
				self.lines.append("storeAtVar")
				self.lines.append(index)

			if isinstance(value,AST.LiteralArray):
				ptr = self.op_var()
				ptr2 = self.op_var()
				# allocate the array
				self.lines.append("imalloc")
				self.lines.append(str(len(value.values)))
				self.lines.append("storeAtVar")
				self.lines.append(ptr)
				self.lines.append("storeAtVar")
				self.lines.append(ptr2)
				p_arr_var = self.arr_var
				p_arr_ptr = self.arr_ptr
				self.arr_var = ptr
				self.arr_ptr = True

				self.assemble_literal_array(value)

				self.arr_var = p_arr_var
				self.arr_ptr = p_arr_ptr
				self.op_depth -= 1
				self.lines.append("loadAtVar")
				self.lines.append(ptr2)
				self.op_depth -= 1
			else:
				self.assemble_auto(value)

			if self.arr_ptr:
				self.lines.append("setValueAtPointer")
				self.lines.append(arr_var)
				self.lines.append("loadAtVar")
				self.lines.append(arr_var)
				self.lines.append("incA")
				self.lines.append("storeAtVar")
				self.lines.append(arr_var)
				continue
			self.lines.append("storeAtVarWithOffset")
			self.lines.append(arr_var)
			self.lines.append(index)
		self.op_depth -= 1

		return arr_var


def compile(code: str, source: str, force: bool = False):
	"""
		Preform the full compilation procedure, unless errors arise along the way

		:param code: The code to compile
		:param source: The source file name or path
		:param force: Force the compilation to continue even if errors arise.
			Can and probably will cause undefined behaviour.
		:return: A monad containing the compiled code and diagnostics
	"""
	tokens = tokenize(code, source)
	monad = Monad(tokens, force=force)
	monad >>= parse
	monad >>= Scanner.do
	monad >>= Assembler.do
	return monad
