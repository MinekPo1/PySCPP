from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias, cast
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
	COMPILER_FUNCTION = auto()
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
		if tokens != [] and c == "/" and current == "" and tokens[-1].value == "/":
			while True:
				c = next(pointer)
				if c is None:
					break
				if c == "\n":
					break
			del tokens[-1]
			current = ""
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
					tokens.append(Token(TokenType.COMPILER_FUNCTION, current, pointer.pos))
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
class Root(Container):
	includes: list[str]
	definitions: dict[str,str]


@dataclass
class CompilerFunction(ASTElement):
	value: str


@dataclass
class TokenElement(ASTElement):
	token: Token


@dataclass
class Namespace(Container):
	name: str


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
	name: str
	value: ASTExpression | None


@dataclass
class ArrayDef(ASTElement):
	name: str
	value: list[ASTExpression] | None
	length: int


@dataclass
class Return(ASTElement):
	value: ASTExpression | None


def _parse_expr(elements: list[ASTElement], errors: list[Error])\
		-> ASTExpression:
	return NotImplemented


def _parse(container: Container, errors: list[Error]) -> None:
	queue = container.children.copy()
	stack: list[ASTElement] = []
	new_element: ASTElement
	# we will be moving elements from the queue to the stack
	# trying to find patterns
	while queue != []:
		element = queue.pop(0)
		if isinstance(element, TokenElement):
			if element.token.type == TokenType.COMPILER_FUNCTION:
				new_element = CompilerFunction(element.token.pos, element.token.value)
				# try to recognize the compiler function
				if element.token.value.split()[0] == "include":
					if not isinstance(container, Root):
						errors.append(Error(
							"include directive can only be used in the root namespace",
							element.token.pos
						))
						# ignore the include directive
						continue
					if len(element.token.value.split()) != 2:
						errors.append(Error(
							"include directive must have exactly one argument",
							element.token.pos
						))
						# ignore the include directive
						continue
					container.includes.append(element.token.value.split()[1])
					continue
				if element.token.value.split()[0] == "define":
					if not isinstance(container, Root):
						errors.append(Error(
							"define directive can only be used in the root namespace",
							element.token.pos
						))
						# ignore the define directive
						continue
					if len(element.token.value.split()) < 3:
						errors.append(Error(
							"define directive must have two arguments",
							element.token.pos
						))
						# ignore the define directive
						continue
					container.definitions[element.token.value.split()[1]] = \
						" ".join(element.token.value.split()[2:])
					continue
				# else it's a compiler function
				# we don't know what it is, either just now
				# or we will never know
				stack.append(new_element)
				continue
			if element.token.type == TokenType.KEYWORD:
				if element.token.value == "namespace":
					# we expect a identifier followed by a container
					if len(queue) < 2:
						errors.append(Error("Expected a name", element.token.pos))
						continue
					if not isinstance(queue[0], TokenElement):
						errors.append(Error("Expected a name", element.token.pos))
						continue
					if queue[0].token.type != TokenType.IDENTIFIER:
						errors.append(Error("Expected a name", element.token.pos))
						continue
					if not isinstance(queue[1], Container):
						errors.append(Error("Expected a namespace body", element.token.pos))
						continue
					new_element = Namespace(
						queue[0].token.pos, queue[1].children, queue[0].token.value
					)
					stack.append(new_element)
					# remove the namespace body and name from the queue
					queue.pop(0)
					queue.pop(0)
					continue
				if element.token.value == "if":
					# we expect an expression in parenthesis followed by a container
					if len(queue) < 1:
						errors.append(Error(
							"Expected an expression in parenthesis and a body",
							element.token.pos
						))
						continue
					if not isinstance(queue[0], TokenElement):
						errors.append(Error(
							"Expected an expression in parenthesis",
							element.token.pos
						))
						continue
					if queue[0].token.type != TokenType.PAREN_OPEN:
						errors.append(Error(
							"Expected an expression in parenthesis",
							element.token.pos
						))
						continue
					# look for the closing parenthesis
					paren_count = 1
					for i,j in enumerate(queue):
						if isinstance(j, TokenElement)\
							and j.token.type == TokenType.PAREN_OPEN:
							paren_count += 1
						if isinstance(j, TokenElement)\
							and j.token.type == TokenType.PAREN_CLOSE:
							paren_count -= 1
						if paren_count == 0:
							break
					else:
						errors.append(Error("Expected a closing parenthesis", element.token.pos))
						continue
					# extract the expression
					expression = queue[1:i]
					# remove the expression and the parenthesis from the queue
					queue = queue[i+1:]
					n_expression = _parse_expr(expression, errors)
					# check if the body is next
					if len(queue) < 1:
						errors.append(Error("Expected a body", element.token.pos))
						continue
					if not isinstance(queue[0], Container):
						errors.append(Error("Expected a body", element.token.pos))
						continue
					stack.append(If(element.token.pos, queue[0].children, n_expression))
				if element.token.value == "while":
					# we expect an expression in parenthesis followed by a container
					if len(queue) < 1:
						errors.append(Error(
							"Expected an expression in parenthesis and a body",
							element.token.pos
						))
						continue
					if not isinstance(queue[0], TokenElement):
						errors.append(Error(
							"Expected an expression in parenthesis",
							element.token.pos
						))
						continue
					if queue[0].token.type != TokenType.PAREN_OPEN:
						errors.append(Error(
							"Expected an expression in parenthesis",
							element.token.pos
						))
						continue
					# look for the closing parenthesis
					paren_count = 1
					for i,j in enumerate(queue):
						if isinstance(j, TokenElement)\
							and j.token.type == TokenType.PAREN_OPEN:
							paren_count += 1
						if isinstance(j, TokenElement)\
							and j.token.type == TokenType.PAREN_CLOSE:
							paren_count -= 1
						if paren_count == 0:
							break
					else:
						errors.append(Error("Expected a closing parenthesis", element.token.pos))
						continue
					# extract the expression
					expression = queue[1:i]
					# remove the expression and the parenthesis from the queue
					queue = queue[i+1:]
					n_expression = _parse_expr(expression, errors)
					# check if the body is next
					if len(queue) < 1:
						errors.append(Error("Expected a body", element.token.pos))
						continue
					if not isinstance(queue[0], Container):
						errors.append(Error("Expected a body", element.token.pos))
						continue
					stack.append(While(element.token.pos, queue[0].children, n_expression))
				if element.token.value == "for":
					# we expect:
					# (id "from" expr "to" expr)
					# or
					# (id "from" expr "to" expr "by" expr)
					if len(queue) < 1:
						errors.append(Error(
							"Expected an expression in parenthesis and a body",
							element.token.pos
						))
						continue
					if not isinstance(queue[0], TokenElement):
						errors.append(Error(
							"Expected an expression in parenthesis",
							element.token.pos
						))
						continue
					if queue[0].token.type != TokenType.PAREN_OPEN:
						errors.append(Error(
							"Expected an expression in parenthesis",
							element.token.pos
						))
						continue
					# look for the closing parenthesis
					paren_count = 1
					for i,j in enumerate(queue):
						if isinstance(j, TokenElement)\
							and j.token.type == TokenType.PAREN_OPEN:
							paren_count += 1
						if isinstance(j, TokenElement)\
							and j.token.type == TokenType.PAREN_CLOSE:
							paren_count -= 1
						if paren_count == 0:
							break
					else:
						errors.append(Error("Expected a closing parenthesis", element.token.pos))
						continue
					# extract the expression
					expression = queue[1:i]

					# var is always the first element and is always a single token
					if not isinstance(expression[0], TokenElement):
						errors.append(Error("Expected a variable", element.token.pos))
						continue
					if expression[0].token.type != TokenType.IDENTIFIER:
						errors.append(Error("Expected a variable", element.token.pos))
						continue
					var = expression[0].token.value
					# check if "from" is next
					if len(expression) < 2:
						errors.append(Error("Expected 'from'", element.token.pos))
						continue
					if not isinstance(expression[1], TokenElement):
						errors.append(Error("Expected 'from'", element.token.pos))
						continue
					if expression[1].token.value != "from":
						errors.append(Error("Expected 'from'", element.token.pos))
						continue
					# look for the "to"
					for i,j in enumerate(expression[2:]):
						if isinstance(j, TokenElement)\
							and j.token.value == "to":
							break
					else:
						errors.append(Error("Expected 'to'", element.token.pos))
						continue
					# extract the expression
					from_ = _parse_expr(expression[2:i+2], errors)
					# check if "by" is ahead
					i2 = -1
					for i2,j2 in enumerate(expression[i+2:]):
						if isinstance(j2, TokenElement)\
							and j2.token.value == "by":
							by = _parse_expr(expression[i+3+i2:], errors)
							break
					else:
						by = None
					to = _parse_expr(expression[i+2+i2:], errors)
					# check if the body is next
					if len(queue) < 1:
						errors.append(Error("Expected a body", element.token.pos))
						continue
					if not isinstance(queue[0], Container):
						errors.append(Error("Expected a body", element.token.pos))
						continue
					stack.append(For(
						element.token.pos, queue[0].children, var, from_, to, by
					))
				if element.token.value == "var":
					# we expect:
					# id;
					# or
					# id = expr;
					# or
					# id[num];
					# or
					# id = {expr, expr, ...};

					# either way we expect an id token
					if len(queue) < 1:
						errors.append(Error("Expected an identifier", element.token.pos))
						continue
					if not isinstance(queue[0], TokenElement):
						errors.append(Error("Expected an identifier", element.token.pos))
						continue
					if queue[0].token.type != TokenType.IDENTIFIER:
						errors.append(Error("Expected an identifier", element.token.pos))
						continue
					id = queue[0].token.value

					# the next token is either a semicolon, an equal sign or a square bracket
					if len(queue) < 1:
						errors.append(Error(
							"Unexpected end of container. Did you forget a semicolon?",
							element.token.pos)
						)
						continue
					if not isinstance(queue[0], TokenElement):
						errors.append(Error("Invalid syntax", element.token.pos))
						continue

					if cast(TokenElement,queue[0]).token.type == TokenType.SEMICOLON:
						# we can exit here
						stack.append(VarDef(element.token.pos, id, None))
						# clean up the queue
						queue = queue[2:]
						continue
					elif cast(TokenElement,queue[0]).token.type == TokenType.EQUALS_SIGN:
						# we expect an expression or a curly bracket
						if len(queue) < 2:
							errors.append(Error("Expected a value", element.token.pos))
							continue
						if isinstance(queue[1], Container):
							items = []
							# we expect expressions split by commas
							s = 0
							for i,j in enumerate(queue[1].children):
								if isinstance(j, TokenElement)\
									and j.token.value == ",":
									items.append(_parse_expr(queue[1].children[s:i], errors))
									s = i+1
							if s < len(queue[1].children):
								items.append(_parse_expr(queue[1].children[s:], errors))
							stack.append(ArrayDef(element.token.pos, id, items, len(items)))
							# we expect a semicolon
							if len(queue) < 2:
								errors.append(Error("Expected a semicolon", element.token.pos))
								continue
							if not isinstance(queue[2], TokenElement):
								errors.append(Error("Expected a semicolon", element.token.pos))
								continue
							if queue[2].token.value != ";":
								errors.append(Error("Expected a semicolon", element.token.pos))
								continue
							# clean up the queue
							queue = queue[3:]
							continue
						# we expect an expression
						# lookahead for a semicolon
						for i,j in enumerate(queue[1:]):
							if isinstance(j, TokenElement)\
								and j.token.value == ";":
								expr = _parse_expr(queue[1:i+1], errors)
								# clean up the queue
								queue = queue[i+2:]
								stack.append(VarDef(element.token.pos, id, expr))
								break
						else:
							errors.append(Error("Expected a semicolon", element.token.pos))
							continue
						continue
					elif cast(TokenElement,queue[0]).token.type == TokenType.SQ_BRACKET_OPEN:
						# we expect a number
						if len(queue) < 2:
							errors.append(Error("Expected a number", element.token.pos))
							continue
						if not isinstance(queue[1], TokenElement):
							errors.append(Error("Expected a number", element.token.pos))
							continue
						if queue[1].token.type != TokenType.NUMBER:
							errors.append(Error("Expected a number", element.token.pos))
							continue
						try:
							num = int(queue[1].token.value)
						except ValueError:
							num = int(float(queue[1].token.value))

						# we expect a square bracket close
						if len(queue) < 3:
							errors.append(Error(
								"Expected a square bracket close", element.token.pos
							))
							continue
						if not isinstance(queue[2], TokenElement):
							errors.append(Error(
								"Expected a square bracket close", element.token.pos
							))
							continue
						if queue[2].token.type != TokenType.SQ_BRACKET_CLOSE:
							errors.append(Error(
								"Expected a square bracket close", element.token.pos
							))
							continue
						# we expect a semicolon
						if len(queue) < 3:
							errors.append(Error("Expected a semicolon", element.token.pos))
							continue
						if not isinstance(queue[3], TokenElement):
							errors.append(Error("Expected a semicolon", element.token.pos))
							continue
						if queue[3].token.value != ";":
							errors.append(Error("Expected a semicolon", element.token.pos))
							continue

						# clean up the queue
						queue = queue[4:]
						stack.append(ArrayDef(element.token.pos, id, None, num))
						continue
				if element.token.value == "return":
					# we expect an expression and a semicolon
					# lookahead for a semicolon
					for i,j in enumerate(queue[1:]):
						if isinstance(j, TokenElement)\
							and j.token.value == ";":
							expr = _parse_expr(queue[1:i+1], errors)
							# clean up the queue
							queue = queue[i+2:]
							stack.append(Return(element.token.pos, expr))
							break
					else:
						errors.append(Error("Expected a semicolon", element.token.pos))
						continue
					continue


def parse(tokens: list[Token]) -> tuple[Root,list[Error]]:
	errors = []

	root = Root((0,0,tokens[0].pos[2]), [], [], {})

	root.children = [
		TokenElement(token=token, pos=token.pos)
		for token in tokens
	]

	# find bracket pairs
	pairs: list[tuple[int,int]] = []
	bracket_stack: list[int] = []
	for i, token in enumerate(tokens):
		if token.type == TokenType.BRACKET_OPEN:
			bracket_stack.append(i)
		elif token.type == TokenType.BRACKET_CLOSE:
			if not bracket_stack:
				errors.append(Error("Unmatched bracket", token.pos))
				continue
			pairs.append((bracket_stack.pop(), i))
	if bracket_stack:
		errors.extend(
			Error("Unmatched bracket",tokens[i].pos)
			for i in bracket_stack
		)

	# wrap tokens in containers
	for i, j in pairs:
		container = Container(pos=root.children[i].pos, children=[])
		container.children = root.children[i+1:j]
		root.children[i] = container

	# delete stale children
	offset = 0
	for i, j in pairs:
		root.children[i+1-offset:j+1-offset] = []
		offset += j-i-1

	# do shit antlr would do for me but I'm a masochist
	_parse(root,errors)

	return root, errors
