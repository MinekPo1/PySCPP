from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeAlias, TypeVar
from re import compile as rec

Pos: TypeAlias = tuple[int, int, str]


@dataclass
class Error:
	message: str
	pos: Pos


class CodePointer:
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


class TokenType(Enum):
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
	ARROW = auto()
	MODIFIER = auto()


class RegexBank:
	number = rec(r'-?([0-9]+(\.[0-9]*)?|\.[0-9]+)')
	identifier = rec(r'[a-zA-Z_][a-zA-Z0-9_]*')


SingleCharacterTokens = {
	"+": TokenType.OPERATOR,
	"-": TokenType.OPERATOR,
	"*": TokenType.OPERATOR,
	"/": TokenType.OPERATOR,
	"%": TokenType.OPERATOR,
	"^": TokenType.OPERATOR,
	"=": TokenType.EQUALS_SIGN,
	"<": TokenType.OPERATOR,
	">": TokenType.OPERATOR,
	",": TokenType.COMMA,
	";": TokenType.SEMICOLON,
	"|": TokenType.OPERATOR,
	"&": TokenType.OPERATOR,
	"(": TokenType.PAREN_OPEN,
	")": TokenType.PAREN_CLOSE,
	"[": TokenType.SQ_BRACKET_OPEN,
	"]": TokenType.SQ_BRACKET_CLOSE,
	"{": TokenType.BRACKET_OPEN,
	"}": TokenType.BRACKET_CLOSE,
	":": TokenType.COLON,
}


CompoundTokens = {
	"->": TokenType.ARROW,
	"<=": TokenType.OPERATOR,
	">=": TokenType.OPERATOR,
	"!=": TokenType.OPERATOR,
	"==": TokenType.OPERATOR,
	"&&": TokenType.OPERATOR,
	"||": TokenType.OPERATOR,
	"++": TokenType.MODIFIER,
	"--": TokenType.MODIFIER,
	"::": TokenType.MEMBER_SELECT,
	"//": TokenType.UNKNOWN
}

Keywords = {
	"if": TokenType.KEYWORD,
	"while": TokenType.KEYWORD,
	"for": TokenType.KEYWORD,
	"return": TokenType.KEYWORD,
	"var": TokenType.KEYWORD,
	"namespace": TokenType.KEYWORD,
	"from": TokenType.KEYWORD,
	"to": TokenType.KEYWORD,
	"by": TokenType.KEYWORD,
	"func": TokenType.KEYWORD,
	"public": TokenType.KEYWORD,
	"private": TokenType.KEYWORD,
	"inline": TokenType.KEYWORD,
	"_asm_": TokenType.KEYWORD,
	"_valueOfA_": TokenType.KEYWORD,
}


@dataclass
class Token:
	type: TokenType
	value: str
	pos: Pos


def tokenize(code: str, source: str) -> list[Token]:
	code += " "
	pointer = CodePointer(code, source)
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
		if tokens != [] and c == "/" and current == "" and tokens[-1].value == "*":
			while True:
				c = next(pointer)
				if c is None:
					break
				if c == "*" and next(pointer) == "/":
					break
			del tokens[-1]
			continue

		if current in CompoundTokens:
			tokens.append(Token(CompoundTokens[current], current, pointer.pos))
			continue

		if tokens != [] and tokens[-1].value+c in CompoundTokens:
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
				tokens.append(Token(TokenType.NUMBER, current, pointer.pos))
			elif current in Keywords:
				tokens.append(Token(Keywords[current], current, pointer.pos))
			elif RegexBank.identifier.fullmatch(current):
				tokens.append(Token(TokenType.IDENTIFIER, current, pointer.pos))
			else:
				tokens.append(Token(TokenType.UNKNOWN, current, pointer.pos))
			current = ""
			if token_shelf is not None:
				tokens.append(token_shelf)
				token_shelf = None
			continue

		if current == "\"":
			# look for end of string
			while True:
				current += c
				c = next(pointer)
				if c is None:
					tokens.append(Token(TokenType.UNKNOWN, current, pointer.pos))
					current = ""
					break
				if c == "\"" and current[-1] != "\\":
					current += c
					tokens.append(Token(TokenType.STRING, current, pointer.pos))
					current = ""
					break
				if c == "\n":
					tokens.append(Token(TokenType.UNKNOWN, current, pointer.pos))
					current = ""
					break
			continue

		if current == "#":
			# look for newline
			while True:
				current += c
				c = next(pointer)
				if c is None:
					tokens.append(Token(TokenType.UNKNOWN, current, pointer.pos))
					current = ""
					break
				if c == "\n":
					tokens.append(Token(TokenType.PREPROCESSOR, current, pointer.pos))
					current = ""
					break
			continue

		current += c

	return tokens


@dataclass
class ASTElement:
	pos: Pos


class ASTExpression(ASTElement):
	pass


@dataclass
class Container(ASTElement):
	children: list[ASTElement]


@dataclass
class Accessible(ASTElement):
	name: str
	private: bool | None = field(default=None, init=False)

@dataclass
class Root(Container):
	includes: list[str]
	definitions: dict[str,str]


@dataclass
class Preprocessor(ASTElement):
	value: str


@dataclass
class TokenElement(ASTElement):
	token: Token


@dataclass
class Namespace(Accessible,Container):
	default_private: bool | None = None


@dataclass
class If(Container):
	condition: ASTExpression


@dataclass
class While(Container):
	condition: ASTExpression


@dataclass
class For(Container):
	var: str
	start: ASTExpression
	end: ASTExpression
	step: ASTExpression | None


@dataclass
class VarDef(ASTElement):
	var: Var
	value: ASTExpression | None
	private: bool | None = None
	offset: ASTExpression | None = None

@dataclass
class Return(ASTElement):
	value: ASTExpression | None


@dataclass
class ASM(ASTElement):
	exprs: list[ASTExpression]


@dataclass
class FuncDef(Accessible, Container):
	args: list[Var]


@dataclass
class Var(ASTElement):
	name: str
	memory_specifier: str | None = None


@dataclass
class VarSet(ASTElement):
	var: Var
	value: ASTExpression
	offset: ASTExpression | None = None
	modifier: str | None = None


@dataclass
class FuncCall(ASTElement):
	name: str
	args: list[ASTExpression]

T = TypeVar('T')


def _wrap(func: Callable[[Parser], T]) -> Callable[[Parser], T]:
	def wrapper(parser: Parser) -> T:
		try:
			return func(parser)
		except AssertionError as e:
			try:
				parser.errors.append(Error(str(e.args[0]), parser.token.pos))
			except IndexError:
				parser.errors.append(Error(str(e.args[0]), parser.tokens[-1].pos))
			parser.consume_token()
			return None  # type:ignore
	return wrapper


class Parser:
	class _TokenViewType:
		def __init__(self, parser: Parser):
			self.parser = parser

		def __getitem__(self, index: int) -> Token:
			return self.parser.tokens[index+self.parser.tokens_i]

		def __delitem__(self, index: int) -> None:
			del self.parser.tokens[index+self.parser.tokens_i]

	tokens: list[Token]
	tokens_i = 0
	errors: list[Error]
	stack: list[Container]
	root: Root

	def __init__(
			self,
			tokens: list[Token], source: str, root: Root | None = None
		):
		self.tokens = tokens
		self.errors = []
		if root is None:
			self.root = Root(
				includes=[], definitions={}, pos=(0, 0, source), children=[]
			)
		else:
			self.root = root

		self.stack = [self.root]
		self.container = self.root
		self.token_view = self._TokenViewType(self)

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
		"""
		if not(condition):
			self.errors.append(Error(message, self.token.pos))
		return condition

	@property
	def token(self):
		return self.tokens[self.tokens_i]

	@property
	def head(self):
		return self.stack[-1]

	def parse(self):
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
			if self.token.type == TokenType.PREPROCESSOR:
				self.parse_preprocessor()
				continue

			if self.token.type == TokenType.KEYWORD:

				if self.token.value == "namespace":
					self.parse_namespace()
					continue

				if self.token.value == 'func':
					self.parse_func()
					continue

				if self.token.value == 'struct':
					self.parse_struct()
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
			assert isinstance(self.container, Root), "#include must be in root"
			self.root.includes.append(self.token.value.removeprefix("#include "))
			return

		if self.token.value.startswith("#define"):
			# place the define in the root
			assert isinstance(self.container, Root), "#define must be in root"
			self.root.definitions[self.token.value.removeprefix("#define ")] = ""
			return

		# if we get here, we have a preprocessor
		# that we don't know how to handle
		raise AssertionError(f"Unknown preprocessor: {self.token.value[1:].split()[0]}")

	@_wrap
	def parse_namespace(self):
		# we expect a `namespace` keyword
		# then optionally a `public` or `private` keyword
		# then a name
		# and then the body

		# create the namespace
		# we will update it as we go
		namespace = Namespace(
			pos=self.token.pos,
			name="<UNNAMED>",
			children=[],
		)
		# check if we have a access specifier
		if self.token_view[-1].type == TokenType.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			namespace.private = self.token_view[-1].value == 'private'

		self.consume_token()

		# check if there is a default access specifier

		if self.token.type == TokenType.KEYWORD \
			and self.token.value in {'public','private'}:
			namespace.default_private = self.token.value == 'private'
			self.consume_token()

		assert self.token.type == TokenType.IDENTIFIER, "Expected namespace name"

		namespace.name = self.token.value

		self.consume_token()

		assert self.token.type == TokenType.BRACKET_OPEN, "Expected namespace body"
		self.consume_token()

		self.stack.append(namespace)

		consumer = self.consumer()
		while next(consumer):
			if self.token.type == TokenType.BRACKET_CLOSE:
				self.consume_token()
				break
			if self.token.type == TokenType.KEYWORD:
				if self.token.value == "namespace":
					self.parse_namespace()
					continue
				if self.token.value == "func":
					self.parse_func()
					continue
				if self.token.value in {'private','public'}:
					# this should be accessed later
					continue

			self.parse_statement()
		else:
			assert False, "Expected closing bracket"

		self.stack.pop()

		self.head.children.append(namespace)

	@_wrap
	def parse_func(self):

		# construct the function
		# we will update it as we go
		func = FuncDef(
			pos=self.token.pos,
			name="<UNNAMED>",
			children=[],
			args=[],
		)

		# check if we have a access specifier
		if self.token_view[-1].type == TokenType.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			func.private = self.token_view[-1].value == 'private'

		self.consume_token()

		assert self.token.type == TokenType.IDENTIFIER, "Expected function name"
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

			if self.token.type == TokenType.OPERATOR:
				if self.assert_(self.token.value in "&*", "unknown "):
					memory_specifier += self.token.value
				continue

			if self.token.type == TokenType.IDENTIFIER:
				func.args.append(Var(
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
					break
				continue
			if self.token.type == TokenType.COMMA:
				self.assert_(True, "Expected identifier")
				continue
			raise AssertionError(f"Unexpected token: {self.token.value}")
		else:
			assert False, "Expected closing bracket"

		# self.consume_token()
		assert self.token.type == TokenType.BRACKET_OPEN, "Expected function body"
		self.consume_token()

		self.stack.append(func)

		consumer = self.consumer()

		while next(consumer):
			if self.token.type == TokenType.BRACKET_CLOSE:
				# self.consume_token()
				break
			self.parse_statement()
		else:
			assert False, "Expected closing bracket"

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

		if self.token.type == TokenType.KEYWORD:
			if self.token.value == "return":
				self.parse_return()
				# expect a semi-colon
				self.consume_token()
				self.assert_(self.token.value == ';', "Expected semi-colon")
				return
			if self.token.value == "if":
				self.parse_if()
				return
			if self.token.value == "while":
				self.parse_while()
				return
			if self.token.value == "for":
				self.parse_for()
				return

			if self.token.value == "_asm_":
				self.parse_raw_asm()
				return
			if self.token.value == "var":
				self.parse_var_def()
				return

		# if we get here, we have a function call or a variable assignment
		# we need to figure out which
		if self.token_view[1].type == TokenType.MEMBER_SELECT:
			self.parse_member_select()
			return
		if self.token_view[1].type == TokenType.PAREN_OPEN:
			self.parse_func_call()
			return
		# must be a variable assignment
		self.parse_var_assignment()

	@_wrap
	def parse_var_def(self):
		# construct the variable
		# we will update it as we go
		var = VarDef(
			pos=self.token.pos,
			var=Var(
				pos=self.token.pos,
				name="<UNNAMED>"
			),
			value=None,
		)

		# check if we have a access specifier
		if self.token_view[-1].type == TokenType.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			var.private = self.token_view[-1].value == 'private'

		self.consume_token()

		if self.token.type == TokenType.OPERATOR and self.token.value in "&*":
			var.var.memory_specifier = self.token.value
			self.consume_token()

		assert self.token.type == TokenType.IDENTIFIER, "Expected variable name"
		var.var.name = self.token.value
		self.consume_token()

		if self.token.type == TokenType.SQ_BRACKET_OPEN:
			self.consume_token()
			var.offset = self.parse_expression()
			self.consume_token()
			self.assert_(self.token.value == ']', "Expected closing bracket")
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
		var = VarSet(
			pos=self.token.pos,
			var=Var(
				pos=self.token.pos,
				name="<UNNAMED>"
			),
			value=None,
		)
		if self.token.type == TokenType.OPERATOR and self.token.value in "&*":
			var.var.memory_specifier = self.token.value
			self.consume_token()

		assert self.token.type == TokenType.IDENTIFIER, "Syntax error"
		# the error is generic because parse_var_assignment is used
		# as a guard clause for parse_statement, which itself is used
		# as a guard clause for parse_namespace

		var.var.name = self.token.value
		self.consume_token()

		if self.token.type == TokenType.SQ_BRACKET_OPEN:
			self.consume_token()
			var.offset = self.parse_expression()
			self.consume_token()
			self.assert_(self.token.value == ']', "Expected closing bracket")
			self.consume_token()

		if self.token.type == TokenType.OPERATOR:
			var.modifier = self.token.value
			self.consume_token()

		if self.token.type == TokenType.MODIFIER:
			var.modifier = self.token.value
			self.consume_token()
			# TODO: set `value` to 1

		else:
			self.assert_(self.token.value == '=', "Expected equals")
			self.consume_token()
			var.value = self.parse_expression()

		self.consume_token()
		self.assert_(self.token.value == ';', "Expected semicolon")

		self.head.children.append(var)

	@_wrap
	def parse_func_call(self):
		func = FuncCall(
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
			while self.token.value == ',':
				self.consume_token()
				func.args.append(self.parse_expression())

		self.assert_(self.token.value == ')', "Expected closing parenthesis")
		self.consume_token()

		self.head.children.append(func)

	@_wrap
	def parse_expression(self) -> ASTExpression:
		assert False, "Expressions are not yet implemented"

	@_wrap
	def parse_member_select(self):
		assert self.token.type == TokenType.IDENTIFIER, "Expected namespace name"
		while self.token_view[2].type == TokenType.MEMBER_SELECT:
			assert self.token_view[2].type == TokenType.IDENTIFIER,\
				"Expected member name"
			self.token.value += f"::{self.token_view[2].value}"
			del self.token_view[1]
			del self.token_view[2]


# wrapper around Parser.parse()
def parse(tokens: list[Token]) -> tuple[Root,list[Error]]:
	parser = Parser(tokens, tokens[0].pos[2])
	parser.parse()

	return parser.root, parser.errors
