from __future__ import annotations
import contextlib
from typing import IO, Callable, Generic, Literal, TypeVar, overload
from io import StringIO
import math
import time


GRAPHICS = False
PYGAME_INIT = False


def init_pygame():
	global pygame, PYGAME_INIT
	if not PYGAME_INIT:
		import pygame
		pygame.init()
		PYGAME_INIT = True


_WT = TypeVar('_WT', float, str)


class _Wrapper:
	instructions: dict[str, Callable]

	def __init__(self):
		self.instructions = {}

	def __call__(self,instruction: str) -> Callable[[Callable], Callable]:
		def wrapper(func: Callable) -> Callable:
			self.instructions[instruction] = func
			return func
		return wrapper


class SLVM:
	"""
		This class implements the vm.

		Use by iterating over the instance or with the :py:meth:`run` method.

		:param code: The code to run. Can be a string or a list of strings.
			If a string, it will be split on newlines.


	"""
	class _Memory:
		class _MemWrapper(Generic[_WT]):
			mem: SLVM._Memory
			is_float: bool

			@overload
			@classmethod
			def __new__(cls, mem: SLVM._Memory, is_float: Literal[True]) \
				-> SLVM._Memory._MemWrapper[float]:
				pass

			@overload
			@classmethod
			def __new__(cls, mem: SLVM._Memory, is_float: Literal[False]) \
				-> SLVM._Memory._MemWrapper[str]:
				pass

			@classmethod  # type: ignore
			def __new__(cls, mem: SLVM._Memory, is_float: bool, *args) \
				-> SLVM._Memory._MemWrapper[_WT]:
				# print(args)
				return super().__new__(cls)

			def __init__(self, mem: SLVM._Memory, is_float: bool) -> None:
				self.mem = mem
				self.is_float = is_float

			def __getitem__(self, index: int) -> _WT:
				if self.is_float:
					try:
						return float(self.mem._raw[index])  # type:ignore
					except ValueError:
						return 0.0  # type:ignore
				return str(self.mem._raw[index])  # type:ignore

			def __setitem__(self, index: int, value: _WT) -> None:
				self.mem._raw[index] = value

		_raw: list[str | float]
		floats: _MemWrapper[float]
		strings: _MemWrapper[str]

		def __init__(self) -> None:
			self._raw = []
			self.floats = self._MemWrapper(self, True)
			self.strings = self._MemWrapper(self, False)

		def __getitem__(self, index: int) -> str | float:
			return self._raw[index]

		def __setitem__(self, index: int, value: str | float) -> None:
			self._raw[index] = value

		def append(self, value: str | float) -> None:
			self._raw.append(value)

		def __len__(self) -> int:
			return len(self._raw)

	_a_reg: float | str
	_memory: _Memory
	_var_lookup: dict[str, int]
	_free_chunks: list[tuple[int,int]]
	_first_unused: int
	_code: list[str]
	_code_ptr: int
	_running: bool = True
	_stack: list[int]
	_graphic_buffer: list[str]
	_sleep_to: float = 0
	_array_sizes: list[int]
	_start_time: float | None = None

	console: IO[str]
	"The io object the VM writes/reads to/from. Default is a StringIO object."

	_wrap = _Wrapper()

	MAX_MEMORY_SIZE = 65536
	"How large the memory is. Override this to change the size of the memory."

	def __init__(self, code: list[str] | str):
		if isinstance(code, str):
			code = code.splitlines()
		self._code = code
		self._code_ptr = 0
		self._a_reg = 0
		self._memory = self._Memory()
		self._array_sizes = []
		self._first_unused = 0
		self._var_lookup = {}
		self._free_chunks = []
		self._stack = []
		self._graphic_buffer = []
		if GRAPHICS:
			init_pygame()
			self._screen = pygame.display.set_mode((640, 480))

		self.console = StringIO()

		try:
			open('.slvmcloud', 'r').close()
		except FileNotFoundError:
			f = open('.slvmcloud', 'w')
			for _ in range(10):
				f.write('0\n')

	def __iter__(self):
		return self

	@property
	def _str_a(self) -> str:
		return str(self._a_reg)

	@property
	def _float_a(self) -> float:
		try:
			return float(self._a_reg)
		except ValueError:
			return 0.0

	@_float_a.setter
	def _float_a(self, value: float) -> None:
		self._a_reg = value

	@property
	def _current(self):
		return self._code[self._code_ptr - 1]

	def _get_next(self):
		return None if self._code_ptr >= len(self._code) else self._get_next_unsafe()

	def _get_next_safe(self):
		if self._code_ptr >= len(self._code):
			raise StopIteration()
		return self._get_next_unsafe()

	def _get_next_unsafe(self):
		i = self._code[self._code_ptr]
		self._code_ptr += 1
		return i

	def __next__(self):
		if self._start_time is None:
			self._start_time = time.time()
		if not self._running:
			raise StopIteration()
		if self._sleep_to > time.time():
			return
		i = self._get_next()
		if i is None:
			raise StopIteration()
		if i not in self._wrap.instructions:
			self._running = False
			raise ValueError(f"Unknown instruction: {i}")
		self._wrap.instructions[i](self)

	def run(self) -> None:
		"""
			Exhausts the object by repeatedly calling ``next`` on it.
		"""
		with contextlib.suppress(StopIteration):
			while self._running:
				next(self)

	def _allocate(self, size: int) -> int:
		if self._free_chunks:
			for i,chunk in enumerate(self._free_chunks):
				if chunk[1] > size:
					addr = chunk[0]
					self._free_chunks[i] = (
						self._free_chunks[i][0] + size, self._free_chunks[i][1] - size
					)
					return addr
				elif chunk[1] == size:
					addr = chunk[0]
					self._free_chunks.pop(i)
					return addr

		if len(self._memory) + size >= self.MAX_MEMORY_SIZE:
			self._running = False
			raise ValueError("Memory overflow")
		self._memory._raw.extend([0] * (size + 1))
		self._array_sizes.extend([1] * (size + 1))
		addr = self._first_unused
		self._first_unused += size
		return addr

	def _dealocate(self, size: int, ptr: int):
		if self._free_chunks:
			for i,chunk in enumerate(self._free_chunks):
				if chunk[0] + chunk[1] == ptr:
					self._free_chunks[i] = (
						chunk[0], chunk[1] + size
					)
					return None
		self._free_chunks.append((ptr, size))

	def _prep_var(self, name: str):
		if name in self._var_lookup:
			return
		self._var_lookup[name] = self._allocate(1)

	@_wrap("ldi")
	def _ldi(self):
		self._a_reg = self._get_next_safe()

	@_wrap("loadAtVar")
	def _loadAtVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = self._memory[self._var_lookup[var]]

	@_wrap("storeAtVar")
	def _storeAtVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._memory[self._var_lookup[var]] = self._a_reg

	@_wrap("jts")
	def _jts(self):
		self._stack.append(self._code_ptr + 1)
		try:
			self._code_ptr = int(self._get_next_safe())
		except ValueError as e:
			self._running = False
			raise ValueError(f"Invalid jump target: {self._current}") from e

	@_wrap("ret")
	def _ret(self):
		if not self._stack:
			self._running = False
			return
		self._code_ptr = self._stack.pop()

	@_wrap("addWithVar")
	def _addWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a += self._memory.floats[self._var_lookup[var]]

	@_wrap("subWithVar")
	def _subWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a -= self._memory.floats[self._var_lookup[var]]

	@_wrap("mulWithVar")
	def _mulWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a *= self._memory.floats[self._var_lookup[var]]

	@_wrap("divWithVar")
	def _divWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a /= self._memory.floats[self._var_lookup[var]]

	@_wrap("modWithVar")
	def _modWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a %= self._memory.floats[self._var_lookup[var]]

	@_wrap("bitwiseLsfWithVar")
	def _bitwiseLsfWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a = float(
			int(self._float_a) << int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("bitwiseRsfWithVar")
	def _bitwiseRsfWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a = float(
			int(self._float_a) >> int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("bitwiseAndWithVar")
	def _bitwiseAndWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a = float(
			int(self._float_a) & int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("bitwiseOrWithVar")
	def _bitwiseOrWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._float_a = float(
			int(self._float_a) | int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("print")
	def _print(self):
		self.console.write(self._str_a)

	@_wrap("println")
	def _println(self):
		self.console.write(self._str_a)
		self.console.write("\n")

	@_wrap("jmp")
	def _jmp(self):
		try:
			self._code_ptr = int(self._get_next_safe())
		except ValueError as e:
			self._running = False
			raise ValueError(f"Invalid jump target: {self._current}") from e

	@_wrap("jt")
	def _jt(self):
		if self._float_a > 0:
			try:
				self._code_ptr = int(self._get_next_safe())
			except ValueError as e:
				self._running = False
				raise ValueError(f"Invalid jump target: {self._current}") from e
		else:
			self._code_ptr += 1

	@_wrap("jf")
	def _jf(self):
		if self._float_a < 1:
			try:
				self._code_ptr = int(self._get_next_safe())
			except ValueError as e:
				self._running = False
				raise ValueError(f"Invalid jump target: {self._current}") from e
		else:
			self._code_ptr += 1

	@_wrap("boolAndWithVar")
	def _boolAndWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) and int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("boolOrWithVar")
	def _boolOrWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) or int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("boolEqualsWithVar")
	def _boolEqualsWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) == int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("largerOrEqualsWithVar")
	def _largerOrEqualsWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) >= int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("smallerOrEqualsWithVar")
	def _smallerOrEqualsWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) <= int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("boolNotEqualsWithVar")
	def _boolNotEqualsWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) != int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("largerWithVar")
	def _largerWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) > int(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("smallerWithVar")
	def _smallerWithVar(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(
			int(self._float_a) < int(self._memory.floats[self._var_lookup[var]])
		)

	# graphics commands

	@_wrap("putPixel")
	def _putPixel(self):
		var_x = self._get_next_safe()
		var_y = self._get_next_safe()
		self._prep_var(var_x)
		self._prep_var(var_y)
		self._graphic_buffer.append(
			"putPixel "
			f"{self._memory.floats[self._var_lookup[var_x]]} "
			f"{self._memory.floats[self._var_lookup[var_y]]}"
		)

	@_wrap("putLine")
	def _putLine(self):
		var_x1 = self._get_next_safe()
		var_y1 = self._get_next_safe()
		var_x2 = self._get_next_safe()
		var_y2 = self._get_next_safe()
		self._prep_var(var_x1)
		self._prep_var(var_y1)
		self._prep_var(var_x2)
		self._prep_var(var_y2)
		self._graphic_buffer.append(
			"putLine "
			f"{self._memory.floats[self._var_lookup[var_x1]]} "
			f"{self._memory.floats[self._var_lookup[var_y1]]} "
			f"{self._memory.floats[self._var_lookup[var_x2]]} "
			f"{self._memory.floats[self._var_lookup[var_y2]]}"
		)

	@_wrap("putRect")
	def _putRect(self):
		var_x = self._get_next_safe()
		var_y = self._get_next_safe()
		var_w = self._get_next_safe()
		var_h = self._get_next_safe()
		self._prep_var(var_x)
		self._prep_var(var_y)
		self._prep_var(var_w)
		self._prep_var(var_h)
		self._graphic_buffer.append(
			"putRect "
			f"{self._memory.floats[self._var_lookup[var_x]]} "
			f"{self._memory.floats[self._var_lookup[var_y]]} "
			f"{self._memory.floats[self._var_lookup[var_w]]} "
			f"{self._memory.floats[self._var_lookup[var_h]]}"
		)

	@_wrap("setColor")
	def _setColor(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._graphic_buffer.append(
			"setColor "
			f"{self._memory.floats[self._var_lookup[var]]}"
		)

	@_wrap("clg")
	def _clg(self):
		self._graphic_buffer.clear()

	@_wrap("done")
	def _done(self):
		self._running = False

	@_wrap("malloc")
	def _malloc(self):
		var = self._get_next_safe()
		self._prep_var(var)
		size = self._memory.floats[self._var_lookup[var]]
		self._a_reg = float(self._allocate(int(size)))

	@_wrap("round")
	def _round(self):
		var = self._get_next_safe()
		var_p = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_p)
		self._memory.floats[self._var_lookup[var]] = float(
			round(
				self._memory.floats[self._var_lookup[var]],
				int(self._memory.floats[self._var_lookup[var_p]])
			)
		)

	@_wrap("celi")
	def _celi(self):
		var = self._get_next_safe()
		var_p = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_p)
		self._memory.floats[self._var_lookup[var]] = float(
			round(
				self._memory.floats[self._var_lookup[var]]
				+ 0.5 * 10 ** -int(self._memory.floats[self._var_lookup[var_p]]),
				int(self._memory.floats[self._var_lookup[var_p]])
			)
		)

	@_wrap("floor")
	def _floor(self):
		var = self._get_next_safe()
		var_p = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_p)
		self._memory.floats[self._var_lookup[var]] = float(
			round(
				self._memory.floats[self._var_lookup[var]]
				- 0.5 * 10 ** -int(self._memory.floats[self._var_lookup[var_p]]),
				int(self._memory.floats[self._var_lookup[var_p]])
			)
		)

	@_wrap("sin")
	def _sin(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(math.sin(self._memory.floats[self._var_lookup[var]]))

	@_wrap("cos")
	def _cos(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(math.cos(self._memory.floats[self._var_lookup[var]]))

	@_wrap("sqrt")
	def _sqrt(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(math.sqrt(self._memory.floats[self._var_lookup[var]]))

	@_wrap("atan2")
	def _atan2(self):
		var_y = self._get_next_safe()
		var_x = self._get_next_safe()
		self._prep_var(var_y)
		self._prep_var(var_x)
		self._a_reg = float(math.atan2(
			self._memory.floats[self._var_lookup[var_y]],
			self._memory.floats[self._var_lookup[var_x]]
		))

	@_wrap("mouseDown")
	def _mouseDown(self):
		self._a_reg = float(pygame.mouse.get_pressed()[0])

	@_wrap("mouseX")
	def _mouseX(self):
		self._a_reg = float(pygame.mouse.get_pos()[0] - 240)

	@_wrap("mouseY")
	def _mouseY(self):
		self._a_reg = float(- pygame.mouse.get_pos()[1] + 180)

	@_wrap("sleep")
	def _sleep(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._sleep_to = self._memory.floats[self._var_lookup[var]]/1000 + time.time()

	@_wrap("drawText")
	def _drawText(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._graphic_buffer.append(
			"drawText "
			+ self._memory.strings[self._var_lookup[var]]
		)

	@_wrap("loadAtVarWithOffset")
	def _loadAtVarWithOffset(self):
		var = self._get_next_safe()
		var_offset = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_offset)
		self._a_reg = float(
			self._memory[
				self._var_lookup[var]
				+ int(self._memory.floats[self._var_lookup[var_offset]])
			]
		)

	@_wrap("storeAtVarWithOffset")
	def _storeAtVarWithOffset(self):
		var = self._get_next_safe()
		var_offset = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_offset)
		self._memory[
			self._var_lookup[var]
			+ int(self._memory.floats[self._var_lookup[var_offset]])
		] = self._a_reg

	@_wrap("isKeyPressed")
	def _isKeyPressed(self):
		var = self._get_next_safe()
		self._prep_var(var)
		key_code = getattr(pygame, f"K_{self._memory.strings[self._var_lookup[var]]}")
		self._a_reg = float(pygame.key.get_pressed()[key_code])

	@_wrap("createArray")
	def _createArray(self):
		var = self._get_next_safe()
		var_size = self._get_next_safe()
		self._prep_var(var_size)
		self._var_lookup[var] = self._allocate(
			int(self._memory.floats[self._var_lookup[var_size]])
		)
		self._array_sizes[self._var_lookup[var]] = int(
			self._memory.floats[self._var_lookup[var_size]]
		)

	@_wrap("createColor")
	def _createColor(self):
		r = self._get_next_safe()
		g = self._get_next_safe()
		b = self._get_next_safe()
		self._prep_var(r)
		self._prep_var(g)
		self._prep_var(b)
		self._a_reg = float(
			int(self._memory.floats[self._var_lookup[r]]) << 16
			| int(self._memory.floats[self._var_lookup[g]]) << 8
			| int(self._memory.floats[self._var_lookup[b]])
		)

	@_wrap("charAt")
	def _charAt(self):
		var = self._get_next_safe()
		var_index = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_index)
		self._a_reg = self._memory.strings[self._var_lookup[var]][
			int(self._memory.floats[self._var_lookup[var_index]])
		]

	@_wrap("contains")
	def _contains(self):
		var = self._get_next_safe()
		var_substring = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(var_substring)
		self._a_reg = float(
			self._memory.strings[self._var_lookup[var_substring]]
			in self._memory.strings[self._var_lookup[var]]
		)

	@_wrap("join")
	def _join(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = self._str_a + self._memory.strings[self._var_lookup[var]]

	@_wrap("setStrokeWidth")
	def _setStrokeWidth(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._graphic_buffer.append(
			"setStrokeWidth "
			+ str(self._memory.floats[self._var_lookup[var]])
		)

	@_wrap("inc")
	def _inc(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._memory.floats[self._var_lookup[var]] += 1

	@_wrap("dec")
	def _dec(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._memory.floats[self._var_lookup[var]] -= 1

	@_wrap("arraySize")
	def _arraySize(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = float(self._array_sizes[self._var_lookup[var]])

	@_wrap("graphicsFlip")
	def _graphicsFlip(self):
		# TODO
		raise NotImplementedError

	@_wrap("newLine")
	def _newLine(self):
		self._graphic_buffer.append("newLine")

	@_wrap("ask")
	def _ask(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self.console.write(self._memory.strings[self._var_lookup[var]])
		self._a_reg = self.console.read()

	@_wrap("setCloudVar")
	def _setCloudVar(self):
		cid = self._get_next_safe()
		var = self._get_next_safe()
		self._prep_var(cid)
		self._prep_var(var)
		with open('.slvmcloud', "r") as f:
			cloud_vars = [float(i) for i in f.readlines()]

		cloud_vars[int(self._memory.floats[self._var_lookup[cid]])] = self._memory.floats[self._var_lookup[var]]

		with open('.slvmcloud', "w") as f:
			f.write("\n".join([str(i) for i in cloud_vars]))

	@_wrap("getCloudVar")
	def _getCloudVar(self):
		cid = self._get_next_safe()
		self._prep_var(cid)
		with open('.slvmcloud', "r") as f:
			cloud_vars = [float(i) for i in f.readlines()]

		self._a_reg = cloud_vars[int(self._memory.floats[self._var_lookup[cid]])]

	@_wrap("indexOfChar")
	def _indexOfChar(self):
		var = self._get_next_safe()
		char = self._get_next_safe()
		self._prep_var(var)
		self._prep_var(char)
		self._a_reg = float(self._memory.strings[self._var_lookup[var]].index(self._memory.strings[self._var_lookup[char]]))

	@_wrap("goto")
	def _goto(self):
		x = self._get_next_safe()
		y = self._get_next_safe()
		self._prep_var(x)
		self._prep_var(y)
		self._graphic_buffer.append(
			"goto "
			+ str(self._memory.floats[self._var_lookup[x]]) + " "
			+ str(self._memory.floats[self._var_lookup[y]])
		)

	@_wrap("imalloc")
	def _imalloc(self):
		size = int(self._get_next_safe())
		self._a_reg = float(self._allocate(size))

	@_wrap("getValueAtPointer")
	def _getValueAtPointer(self):
		pointer = self._get_next_safe()
		self._prep_var(pointer)
		self._a_reg = self._memory[int(self._memory.floats[self._var_lookup[pointer]])]

	@_wrap("setValueAtPointer")
	def _setValueAtPointer(self):
		pointer = self._get_next_safe()
		self._prep_var(pointer)
		self._memory[int(self._memory.floats[self._var_lookup[pointer]])] = self._a_reg

	@_wrap("typeOf")
	def _typeOf(self):
		var = self._get_next_safe()
		self._prep_var(var)
		if isinstance(self._a_reg, str):
			self._memory[self._var_lookup[var]] = "string"
			return
		if self._a_reg.is_integer():
			self._memory[self._var_lookup[var]] = "int"
			return
		self._memory[self._var_lookup[var]] = "float"

	@_wrap("runtimeMillis")
	def _runtimeMillis(self):
		self._a_reg = float(time.time() * 1000)

	@_wrap("free")
	def _free(self):
		addr = self._get_next_safe()
		self._prep_var(addr)
		size = self._get_next_safe()
		self._prep_var(size)

		self._dealocate(
			int(self._memory.floats[self._var_lookup[size]]),
			int(self._memory.floats[self._var_lookup[addr]])
		)

	@_wrap("getVarAddress")
	def _getVarAddress(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._a_reg = self._var_lookup[var]

	@_wrap("setVarAddress")
	def _setVarAddress(self):
		var = self._get_next_safe()
		self._prep_var(var)
		self._var_lookup[var] = int(self._float_a)

	@_wrap("copyVar")
	def _copyVar(self):
		old = self._get_next_safe()
		self._prep_var(old)
		var = self._get_next_safe()
		self._prep_var(var)
		self._var_lookup[var] = self._var_lookup[old]

	@_wrap("incA")
	def _incA(self):
		self._float_a += 1

	@_wrap("decA")
	def _decA(self):
		self._float_a -= 1

	@_wrap("arrayBoundsCheck")
	def _arrayBoundsCheck(self):
		array = self._get_next_safe()
		index = self._get_next_safe()
		self._prep_var(array)
		self._prep_var(index)
		if int(self._memory.floats[self._var_lookup[index]]) >= int(self._memory.floats[self._var_lookup[array]]):
			self._a_reg = 1
		else:
			self._a_reg = 0

	@_wrap("getValueAtPointerOfA")
	def _getValueAtPointerOfA(self):
		self._a_reg = self._memory[int(self._float_a)]
