from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Type, TypeAlias, TypeVar
from re import compile as rec
from pathlib import Path

Pos: TypeAlias = tuple[int, int, str]


INCLUDE_PATH = Path(__file__).parent.parent / "lib"


@dataclass
class Error:
	message: str
	pos: Pos
	stack: list[Pos] = field(default_factory=list)


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
	MEMORY_MODIFIER = auto()
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
	"~": TokenType.MEMORY_MODIFIER,
	"$": TokenType.MEMORY_MODIFIER,
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

		if current == "" and c == "\"":
			print("q")
			# look for end of string
			while True:
				current += c
				c = next(pointer)
				if c is None:
					tokens.append(Token(TokenType.UNKNOWN, current, pointer.pos))
					break
				if c == "\"" and current[-1] != "\\":
					current += c
					tokens.append(Token(TokenType.STRING, current, pointer.pos))
					break
				if c == "\n":
					tokens.append(Token(TokenType.UNKNOWN, current, pointer.pos))
					break
			current = ""
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


class AST:
	@dataclass
	class ASTElement:
		pos: Pos

	class ASTExpression(ASTElement):
		pass

	@dataclass
	class Container(ASTElement):
		children: list[AST.ASTElement]

	@dataclass
	class Accessible(ASTElement):
		name: str
		private: bool | None = field(default=None, init=False)

	@dataclass
	class Root(Container):
		includes: list[AST.Include]
		definitions: list[AST.Definition]

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
		condition: AST.ASTExpression

	@dataclass
	class While(Container):
		condition: AST.ASTExpression

	@dataclass
	class For(Container):
		var: str
		start: AST.ASTExpression
		end: AST.ASTExpression
		step: AST.ASTExpression | None

	@dataclass
	class VarDef(ASTElement):
		var: AST.Var
		value: AST.ASTExpression | None
		private: bool | None = None
		offset: AST.ASTExpression | None = None

	@dataclass
	class Return(ASTElement):
		value: AST.ASTExpression | None

	@dataclass
	class ASM(ASTElement):
		exprs: list[AST.ASTExpression]

	@dataclass
	class FuncDef(Accessible, Container):
		args: list[AST.Var]

	@dataclass
	class Var(ASTExpression):
		name: str
		memory_specifier: str | None = None

	@dataclass
	class VarSet(ASTElement):
		l_value: AST.Var | AST.ASTExpression
		value: AST.ASTExpression
		offset: AST.ASTExpression | None = None
		modifier: str | None = None

	@dataclass
	class MemoryModifier(ASTExpression):
		modifier: str
		value: AST.ASTExpression

	@dataclass
	class FuncCall(ASTExpression):
		name: str
		args: list[AST.ASTExpression]

	@dataclass
	class Operation(ASTExpression):
		op: str
		left: AST.ASTExpression
		right: AST.ASTExpression

	class GetValueOfA(ASTExpression):
		pass

	@dataclass
	class RawASM(ASTExpression):
		arguments: list[AST.ASTExpression]

	@dataclass
	class Literal(ASTExpression):
		value: str | int

	@dataclass
	class LiteralArray(ASTExpression):
		values: list[AST.ASTExpression]

	@dataclass
	class Include(ASTElement):
		value: str

	@dataclass
	class Definition(ASTElement):
		name: str
		value: str


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


class Parser:
	class _TokenViewType:
		def __init__(self, parser: Parser):
			self.parser = parser

		def __getitem__(self, index: int) -> Token:
			if index+self.parser.tokens_i >= len(self.parser.tokens):
				return Token(TokenType.UNKNOWN, "<EOF>", self.parser.tokens[-1].pos)
			return self.parser.tokens[index+self.parser.tokens_i]

		def __delitem__(self, index: int) -> None:
			del self.parser.tokens[index+self.parser.tokens_i]

	tokens: list[Token]
	tokens_i = 0
	errors: list[Error]
	stack: list[AST.Container]
	root: AST.Root

	def __init__(
			self,
			tokens: list[Token], source: str, root: AST.Root | None = None
		):
		self.tokens = tokens
		self.errors = []
		if root is None:
			self.root = AST.Root(
				includes=[], definitions=[], pos=(0, 0, source), children=[]
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
			stack = [c.pos for c in self.stack[1:]]

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
			self.root.definitions.append(
				AST.Definition(
					self.token.pos,self.token.value.split()[1],
					" ".join(self.token.value.split()[2:])
				)
			)

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
		# and then the body

		# create the namespace
		# we will update it as we go
		namespace = AST.Namespace(
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

			if self.token.type == TokenType.MEMORY_MODIFIER:
				memory_specifier += self.token.value
				continue

			if self.token.type == TokenType.IDENTIFIER:
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
			if self.token.type == TokenType.COMMA:
				self.assert_(True, "Expected identifier")
				continue
			self.assert_(False,f"Unexpected token: {self.token.value}")
		else:
			assert False, "Expected closing parenthesis"

		# self.consume_token()
		print(">",self.token.value)
		assert self.token.type == TokenType.BRACKET_OPEN, "Expected function body"
		self.consume_token()

		self.stack.append(func)

		consumer = self.consumer()

		while next(consumer):
			if self.token.type == TokenType.BRACKET_CLOSE:
				print("<",self.token.value, func.name)
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

		if self.token.type == TokenType.KEYWORD:
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
		if self.token_view[1].type == TokenType.MEMBER_SELECT:
			self.parse_member_select()
		print(self.token_view[1])
		if self.token_view[1].type == TokenType.PAREN_OPEN and\
			self.token.type == TokenType.IDENTIFIER:
			self.parse_func_call()
			# self.consume_token()
			assert self.token.value == ';', "Expected semi-colon"
			return
		# must be a variable assignment
		self.parse_var_assignment()
		self.consume_token()

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
		if self.token_view[-1].type == TokenType.KEYWORD \
			and self.token_view[-1].value in {'public','private'}:
			var.private = self.token_view[-1].value == 'private'

		self.consume_token()

		if self.token.type == TokenType.MEMORY_MODIFIER:
			var.var.memory_specifier = self.token.value
			self.consume_token()

		assert self.token.type == TokenType.IDENTIFIER, "Expected variable name"
		var.var.name = self.token.value
		self.consume_token()

		if self.token.type == TokenType.SQ_BRACKET_OPEN:
			self.consume_token()
			var.offset = self.parse_expression()
			# self.consume_token()
			self.assert_(self.token.value == ']', "Expected closing square bracket")
			self.consume_token()

		self.assert_(self.token.value in {'=',';'}, "Expected equals or semicolon")

		if self.token.value == '=':
			self.consume_token()
			var.value = self.parse_expression()

		# self.consume_token()
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
		if self.token.type == TokenType.MEMORY_MODIFIER:
			var.l_value.memory_specifier = self.token.value  # type:ignore
			self.consume_token()

		if self.token.type == TokenType.PAREN_OPEN:
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
			assert self.token.type == TokenType.IDENTIFIER, "Syntax error: " \
				f"unexpected token `{self.token.value}`"
			# the error is generic because parse_var_assignment is used
			# as a guard clause for parse_statement, which itself is used
			# as a guard clause for parse_namespace

			var.l_value.name = self.token.value  # type:ignore
			self.consume_token()

		if self.token.type == TokenType.SQ_BRACKET_OPEN:
			self.consume_token()
			var.offset = self.parse_expression()
			# self.consume_token()
			self.assert_(self.token.type == TokenType.SQ_BRACKET_CLOSE,
				"Expected closing square bracket")
			self.consume_token()

		if self.token.type == TokenType.MEMORY_MODIFIER:
			var.modifier = self.token.value
			self.consume_token()

		if self.token.type == TokenType.MODIFIER:
			var.modifier = self.token.value
			self.consume_token()
			var.value = AST.Literal(var.pos, 1)

		else:
			self.assert_(self.token.value == '=', "Expected equals")
			self.consume_token()
			var.value = self.parse_expression()

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
			print(self.token)
			# self.consume_token()
			while self.token.value == ',':
				self.consume_token()
				func.args.append(self.parse_expression())

		self.assert_(self.token.value == ')', "Expected closing parenthesis")
		self.consume_token()

		self.head.children.append(func)

	@_wrap
	def parse_expression(self) -> AST.ASTExpression:
		# This can be:
		# - a variable reference
		# - a literal
		# - a function call
		# - a operation
		# - a parenthesized expression
		# - raw assembly
		# - `A` register query
		# - a literal array

		if self.token_view[1].type == TokenType.MEMBER_SELECT:
			print("???")
			self.parse_member_select()
			print(self.token)

		a: AST.ASTExpression | None = None

		if self.token_view[1].type == TokenType.PAREN_OPEN:
			self.parse_func_call()
			a = self.head.children.pop()  # type:ignore

		if self.token.type in {TokenType.IDENTIFIER, TokenType.MEMORY_MODIFIER}:
			a = self.parse_var_ref()

		if self.token.type == TokenType.BRACKET_OPEN:
			a = self.parse_literal_array()

		if self.token.type == TokenType.NUMBER:
			a = self.parse_number()

		if self.token.type == TokenType.STRING:
			a = self.parse_string()

		if self.token.type == TokenType.KEYWORD \
			and self.token.value == '_getValueOfA_':
			a = self.parse_getValueOfA()

		if self.token.type == TokenType.KEYWORD and self.token.value == '_asm_':
			self.parse_raw_asm()
			a = self.head.children.pop()  # type:ignore

		# if self.token.type == TokenType.PAREN_CLOSE:
		# 	assert False, "Unexpected closing parenthesis"

		if self.token.type == TokenType.PAREN_OPEN:
			self.consume_token()
			a = self.parse_expression()
			if a is None:
				# enter a replacement for the empty expression
				a = AST.ASTExpression(pos=self.token.pos)
			if self.assert_(self.token.value == ')', "Expected closing parenthesis"):
				self.consume_token()

		if a is None:
			assert False, "Expected expression"

		# print(self.token,self.token_view[1])

		while self.token.type == TokenType.OPERATOR:
			print("+-")
			op = self.token.value
			self.consume_token()
			b = self.parse_expression()
			a = AST.Operation(pos=a.pos, op=op, left=a, right=b)

		return a

	@_wrap
	def parse_getValueOfA(self):
		self.consume_token()
		if self.token.type == TokenType.PAREN_OPEN:
			self.consume_token()
			self.assert_(self.token.value == ')', "Expected closing parenthesis")
			self.consume_token()
		return AST.GetValueOfA(pos=self.token.pos)

	@_wrap
	def parse_raw_asm(self):
		self.consume_token()
		if paren := self.token.type == TokenType.PAREN_OPEN:
			self.consume_token()

		# we expect expressions separated by commas
		asm = AST.RawASM(self.token.pos,[])

		consumer = self.consumer()
		while next(consumer):
			if paren and self.token.value == ')':
				self.consume_token()
				break
			asm.arguments.append(self.parse_expression())
			if self.token.value != ',':
				if paren and self.token.value == ')':
					self.consume_token()
					break
				self.assert_(not paren, "Expected closing parenthesis or comma")
				break
		else:
			self.assert_(not paren, "Expected closing parenthesis")

		self.head.children.append(asm)

	@_wrap
	def parse_var_ref(self):
		memory_descriptor = ""
		while self.token.type == TokenType.OPERATOR:
			memory_descriptor = self.token.value
			self.consume_token()
		var = AST.Var(pos=self.token.pos, name=self.token.value,
			memory_specifier=memory_descriptor)
		self.consume_token()
		return var

	@_wrap
	def parse_number(self):
		num = AST.Literal(pos=self.token.pos, value=self.token.value)
		self.consume_token()
		return num

	@_wrap
	def parse_string(self):
		string = AST.Literal(pos=self.token.pos, value=self.token.value)
		self.consume_token()
		return string

	@_wrap
	def parse_member_select(self):
		assert self.token.type == TokenType.IDENTIFIER, "Expected namespace name"
		while self.token_view[1].type == TokenType.MEMBER_SELECT:
			assert self.token_view[2].type == TokenType.IDENTIFIER,\
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
			if self.token.value != ',':
				self.assert_(self.token.value == '}', "Expected closing bracket")
				break
			self.consume_token()
		self.consume_token()
		return array

	@_wrap
	def parse_return(self):
		self.consume_token()
		if self.token.type == TokenType.SEMICOLON:
			self.head.children.append(AST.Return(pos=self.token.pos,
				value=None))
			return
		self.head.children.append(AST.Return(pos=self.token.pos,
			value=self.parse_expression()))

	@_wrap
	def parse_if(self):
		self.consume_token()
		if_ = AST.If(pos=self.token.pos, condition=None, children=[])  # type:ignore
		assert self.token.value == '(', "Expected opening parenthesis"
		self.consume_token()
		if_.condition = self.parse_expression()
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
		assert self.token.type == TokenType.IDENTIFIER, "Expected identifier"
		for_.var = self.token.value
		self.consume_token()
		assert self.token.value == 'from', "Expected 'from'"
		self.consume_token()
		for_.start = self.parse_expression()
		# self.consume_token()
		assert self.token.value == 'to', "Expected 'to'"
		self.consume_token()
		for_.end = self.parse_expression()
		# self.consume_token()
		if self.token.value == 'by':
			self.consume_token()
			for_.step = self.parse_expression()
			self.parse_statement()
		assert self.token.value == ')', "Expected closing parenthesis"
		self.consume_token()
		self.stack.append(for_)

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


# wrapper around Parser.parse()
def parse(tokens: list[Token]) -> tuple[AST.Root,list[Error]]:
	parser = Parser(tokens, tokens[0].pos[2])
	parser.parse()

	return parser.root, parser.errors


@dataclass
class ScannedFunction:
	implementations: dict[int, AST.FuncDef]


TE = TypeVar('TE', bound=AST.ASTElement)


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


def _add(type: Type[TE])\
	-> Callable[[Callable[[Scanner, TE],T]],Callable[[Scanner, TE],T]]:
	def wrapper(func: Callable[[Scanner, TE],T])\
		-> Callable[[Scanner, TE],T]:
		Scanner.auto[type] = func
		return func
	return wrapper


class Scanner:
	private_ns: dict[str,dict[str, AST.VarDef | ScannedFunction | None]]
	namespaces: dict[str,tuple[dict[str, AST.VarDef | ScannedFunction | None],Pos]]
	to_be_checked: dict[str,AST.FuncDef]
	remote: bool
	errors: list[Error]
	namespace_stack: list[tuple[str,Pos]]
	definitions: dict[str, str]
	accessed: dict[str, int]

	auto: dict[Type[AST.ASTElement], Callable[[Scanner, Any], Any]]

	class Namespace:
		def __init__(self, scanner: Scanner, elm_or_pos: AST.Container, name: str,\
			override: bool = False):
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
		self.namespaces = {"":({},tree.pos)}
		self.namespace_stack = []
		self.to_be_checked = {}
		self.remote = False
		self.namespace_pos_cache = {}

	@property
	def namespace(self) -> str:
		return "::".join([i[0] for i in self.namespace_stack])

	def scan(self):
		self.scan_root(self.tree)
		return self.errors

	def get(self, name: str) -> AST.VarDef | ScannedFunction | dict:
		if name.split("::")[0] in self.private_ns[self.namespace] or \:
			# get from private namespace
			c = self.private_ns[self.namespace]
			ns = self.namespace
			for i in name.split("::")[1:]:
				assert not isinstance(c, AST.VarDef), "Cannot access member of a variable"
				assert not isinstance(c, ScannedFunction), "Cannot access member of a Function"
				assert i in c, f"Cannot find member '{i}' in namespace"





	@_add(AST.Root)
	@_wrapS
	def scan_root(self, root: AST.Root):
		remote = self.remote
		self.remote = False
		for include in root.includes:
			self.scan_include(include)
		for definition in root.definitions:
			self.scan_definition(definition)
		self.remote = remote
		self.scan_container(root)

	@_wrapS
	def scan_include(self, include: AST.Include):
		# locate the file
		include_val = include.value
		if include_val[0] == "<" and include_val[-1] == ">":
			include_val = include_val[1:-1]
			path = INCLUDE_PATH / f"{include_val}.sc"
			assert path.exists(), f"Unable to locate library <{include_val}>"
		elif include_val[0] == "\"" and include_val[-1] == "\"":
			include_val = include_val[1:-1]
			path = Path() / include_val
			assert path.exists(), f"Unable to locate file \"{include_val}\""
		else:
			raise AssertionError(f"Invalid include: {include_val}")

		# parse the file
		with path.open() as f:
			tokens = tokenize(f.read(),str(path))
			root, errors = parse(tokens)
			for error in errors:
				error.stack.insert(0,include.pos)
			self.errors.extend(errors)
			if root is not None:
				self.scan_root(root)

	@_wrapS
	def scan_definition(self, definition: AST.Definition):
		assert definition.name not in self.definitions,\
			f"Duplicate definition of {definition.name}"

	def scan_auto(self, elm: AST.ASTElement):
		assert elm.__class__ in self.auto, f"No auto-scanner for {elm.__class__}"
		return self.auto[elm.__class__](self, elm)

	def scan_container(self, elm: AST.Container):
		for child in elm.children:
			self.scan_auto(child)

	@_add(AST.FuncDef)
	@_wrapS
	def scan_function(self, func: AST.FuncDef):
		if func.name in self.namespaces[self.namespace]:
			func_obj = self.namespaces[self.namespace][func.name]
			assert isinstance(func_obj, ScannedFunction),\
				f"Namespace collision: {func.name}"
		else:
			func_obj = ScannedFunction(implementations={})
			self.namespaces[self.namespace][0][func.name] = func_obj

		assert len(func.args) not in func_obj.implementations, \
			f"Duplicate implementation for {func.name}(\
				{', '.join(arg.name for arg in func.args)})"

		self.namespaces[f"{self.namespace}::Func:{func.name}:{len(func.args)}"] = ({},func.pos)

		func_obj.implementations[len(func.args)] = func
		if self.remote:
			self.to_be_checked[f"{func.name}:{len(func.args)}"] = func
		else:
			with self.Namespace(self, func, f"Func:{func.name}:{len(func.args)}"):
				self.scan_container(func)

	@_add(Namespace)  # type: ignore
	@_wrapS  # type: ignore
	def scan_namespace(self, namespace: Namespace):
		if namespace.name in self.namespaces:
			raise AssertionError(f"Duplicate namespace {namespace.name}")
		self.namespaces[f"{self.namespace}::{namespace.name}"] = ({},namespace.pos)
		with self.Namespace(self, namespace, namespace.name):  # type: ignore
			self.scan_container(namespace)  # type: ignore

	@_add(AST.VarDef)
	@_wrapS
	def scan_variable(self, var: AST.VarDef):
		if var.var.name in self.namespaces[self.namespace]:
			raise AssertionError(f"Duplicate variable {var.var.name}")
		self.namespaces[self.namespace][0][var.var.name] = var

	@_add(AST.FuncCall)
	@_wrapS
	def scan_call(self, call: AST.FuncCall):
		id_str = f"{self.namespace}::{call.name}({len(call.args)})"
		if id_str in self.accessed:
			self.accessed[id_str] += 1
		else:
			self.accessed[id_str] = 1
		if id_str in self.to_be_checked:
			# set namespace to the function's namespace
			with self.Namespace(self, self.namespaces[id_str], id_str, False):
				self.scan_container(self.to_be_checked[id_str])
		call.name = id_str

	@_add(AST.VarSet)
	@_wrapS
	def scan_set(self, set: AST.VarSet):
		self.scan_auto(set.l_value)
		if set.offset is not None:
			self.scan_auto(set.offset)
		self.scan_auto(set.value)

	@_add(AST.Var)
	@_wrapS
	def scan_var(self, var: AST.Var):
		if "::" not in var.name:
			var.name = f"{self.namespace}::{var.name}"
		if var.name in self.accessed:
			self.accessed[var.name] += 1
		else:
			self.accessed[var.name] = 1

	@_add(AST.If)
	@_wrapS
	def scan_if(self, if_: AST.If):
		self.scan_auto(if_.condition)
		self.scan_container(if_)

	@_add(AST.While)
	@_wrapS
	def scan_while(self, while_: AST.While):
		self.scan_auto(while_.condition)
		self.scan_container(while_)

	@_add(AST.For)
	@_wrapS
	def scan_for(self, for_: AST.For):
		self.scan_auto(for_.start)
		self.scan_auto(for_.end)
		if for_.step is not None:
			self.scan_auto(for_.step)
		# define the variable

		self.scan_container(for_)
