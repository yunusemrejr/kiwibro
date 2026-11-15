# Python Reference

## Python basic types: int, float, str, bool, None
Python has the basic types `int` (whole numbers of any size), `float` (decimal numbers), `str` (text), `bool` (`True` or `False`) and `NoneType` (the single value `None`, meaning "no value"). Check a type with `type(x)` or `isinstance(x, int)`. Convert with `int("42")`, `float("3.5")`, `str(7)`.

## Python indentation and blocks
Python marks code blocks by indentation, not braces. Every line in the block must be indented by the same amount, and 4 spaces is the standard. A line ending in a colon (`if x:`, `for i in y:`, `def f():`) starts a block. Mixing tabs and spaces causes an `IndentationError`.

## Python comments and docstrings
A comment starts with `#` and runs to the end of the line. A docstring is a string in triple quotes as the first statement of a function, class or module; it documents it and can be read with `help(f)` or `f.__doc__`. Example: `def f():` followed by an indented `"""Return one."""`.

## Python division: /, //, %, ** and divmod
In Python 3, `/` always returns a float (`7 / 2` is `3.5`). `//` is floor division (`7 // 2` is `3`, and `-7 // 2` is `-4`). `%` is the remainder (`7 % 2` is `1`), `**` is power (`2 ** 10` is `1024`), and `divmod(7, 2)` returns `(3, 1)`.

## Python big integers
A Python `int` has no fixed size limit, so `2 ** 200` works and gives the exact result. It only grows as far as memory allows. A `float` does not behave this way: it is a 64-bit number with limited precision.

## Python round() and rounding
`round(x)` returns an int and `round(x, 2)` rounds to 2 decimal places. Exact halves go to the nearest even number, so `round(2.5)` is `2` and `round(3.5)` is `4`. Because floats are stored in binary, results such as `round(2.675, 2)` can look surprising.

## Python decimal module for exact decimals
The `decimal` module does exact decimal arithmetic, which is useful for money. Build values from strings: `from decimal import Decimal` then `Decimal("0.1") + Decimal("0.2") == Decimal("0.3")` is `True`. Building from a float, `Decimal(0.1)`, keeps the float's error.

## Python truthy and falsy values
In an `if` or `while`, these values count as false: `False`, `None`, `0`, `0.0`, `""`, `[]`, `()`, `{}` and `set()`. Almost everything else counts as true. So `if items:` means "the list is not empty".

## Python == vs is, and None checks
`==` compares values and `is` compares whether two names point to the very same object. Use `is None` and `is not None` to test for `None`, and `==` for everything else. `[1] == [1]` is `True` but `[1] is [1]` is `False`.

## Python if, elif, else and conditional expression
Use `if cond:`, then optional `elif other:`, then optional `else:`. A one-line form is the conditional expression: `label = "adult" if age >= 18 else "minor"`. Combine tests with `and`, `or` and `not`, and chain comparisons like `0 < x < 10`.

## Python for loop, while loop, break, continue, else
`for item in items:` loops over any iterable, `while cond:` loops while the test is true. `break` leaves the loop, `continue` skips to the next round. A loop's `else:` block runs only if the loop finished without `break`.

## Python range() function
`range(5)` gives 0, 1, 2, 3, 4. `range(2, 10, 3)` gives 2, 5, 8 (start, stop, step). The stop value is never included. `list(range(3))` makes a real list.

## Python enumerate() and zip()
`enumerate(items)` yields `(index, item)` pairs: `for i, x in enumerate(items, start=1):`. `zip(a, b)` pairs items from two sequences and stops at the shorter one: `for x, y in zip(a, b):`. `dict(zip(keys, values))` builds a dict.

## Python strings are immutable
A `str` cannot be changed in place; methods like `s.upper()` return a new string. `s[0] = "x"` raises `TypeError`. Index with `s[0]` or `s[-1]` and get the length with `len(s)`. Test for a part with `"ell" in "hello"`.

## Python str.split and str.join
`"a,b,c".split(",")` returns `["a", "b", "c"]`. `"a  b c".split()` with no argument splits on any run of whitespace. `",".join(["a", "b"])` returns `"a,b"`. `join` needs a list of strings, so convert numbers first: `",".join(str(n) for n in nums)`.

## Python str.strip, lstrip, rstrip
`s.strip()` removes whitespace (spaces, tabs, newlines) from both ends, `lstrip()` from the left, `rstrip()` from the right. Giving an argument removes those characters instead: `"xxhixx".strip("x")` is `"hi"`.

## Python str.replace
`"a-b-c".replace("-", "+")` returns `"a+b+c"`. It replaces every match unless you pass a count: `s.replace("-", "+", 1)`. It returns a new string and does not change the original.

## Python str.find vs str.index
`s.find("x")` returns the position of the first match, or `-1` if there is none. `s.index("x")` does the same but raises `ValueError` when not found. `s.count("x")` counts non-overlapping matches.

## Python str.startswith, endswith, upper, lower, title
`s.startswith("http")` and `s.endswith(".txt")` return `True` or `False`; both accept a tuple of choices such as `s.endswith((".png", ".jpg"))`. `upper()`, `lower()`, `title()` and `capitalize()` change letter case and return new strings.

## Python str.isdigit, isalpha, isalnum
`"123".isdigit()` is `True` only if every character is a digit and the string is not empty. `isalpha()` tests letters only, `isalnum()` tests letters and digits. `isdigit()` is `False` for `"-5"` and `"3.5"`, so use `int()` in a `try` to validate numbers.

## Python f-strings: formatted string literals
Put `f` before the quote and write expressions in braces: `f"{name} is {age}"`. Format after a colon: `f"{x:.2f}"` (2 decimals), `f"{n:,}"` (thousands commas), `f"{n:05d}"` (zero-padded), `f"{s:>10}"` (right-aligned). `f"{x=}"` prints the name and value (Python 3.8 and later).

## Python raw strings and multi-line strings
Triple quotes make a string that spans lines: `"""line one` newline `line two"""`. A raw string, `r"C:\new"`, treats backslashes as plain characters; it is the usual way to write regex patterns and Windows paths.

## Python bytes vs str, encode and decode
`str` holds text and `bytes` holds raw bytes. `"é".encode("utf-8")` returns `b'\xc3\xa9'` and `b'\xc3\xa9'.decode("utf-8")` returns `"é"`. Always name the encoding, for example `open(path, encoding="utf-8")`.

## Python list basics and indexing
A list is an ordered, changeable collection: `nums = [10, 20, 30]`. `nums[0]` is 10, `nums[-1]` is the last item, `len(nums)` is the count, `20 in nums` tests membership. Indexing past the end raises `IndexError`.

## Python slicing: a[start:stop:step]
`a[1:3]` takes items 1 and 2 (stop is excluded). `a[:2]` is the first two, `a[-2:]` the last two, `a[::2]` every second item, `a[::-1]` the reversed sequence. Slicing works on lists, strings and tuples and never raises `IndexError`.

## Python list.append, extend, insert
`a.append(x)` adds one item at the end. `a.extend([x, y])` adds each item of another iterable. `a.insert(0, x)` puts an item at a position. `a.append([1, 2])` adds the list as a single nested item, which is different from `extend`.

## Python list.remove, pop, del, clear
`a.remove(x)` deletes the first item equal to `x` and raises `ValueError` if absent. `a.pop()` removes and returns the last item, `a.pop(0)` the first. `del a[1]` deletes by index, `del a[1:3]` deletes a slice, `a.clear()` empties the list.

## Python list.sort vs sorted()
`a.sort()` sorts the list in place and returns `None`. `sorted(a)` returns a new sorted list from any iterable. Both accept `reverse=True` and `key=`: `sorted(words, key=len)` sorts by length. The sort is stable.

## Python list.index, count, reverse
`a.index(x)` returns the position of the first `x` (error if missing). `a.count(x)` counts how many equal `x`. `a.reverse()` reverses in place, while `reversed(a)` gives an iterator and `a[::-1]` gives a new reversed list.

## Python copying lists: shallow vs deep copy
`b = a` makes a second name for the same list. `a.copy()`, `a[:]` and `list(a)` make a shallow copy: a new outer list that shares any nested objects. For a fully independent copy use `import copy` then `copy.deepcopy(a)`.

## Python repeated list pitfall: [[0]*3]*3
`[[0] * 3] * 3` makes three references to the same inner list, so changing one row changes all rows. Build independent rows with a comprehension: `[[0] * 3 for _ in range(3)]`.

## Python list comprehension
`[x * x for x in nums if x > 0]` builds a new list: it takes each `x` from `nums`, keeps those that pass the `if`, and collects `x * x`. Nested loops are allowed: `[(a, b) for a in A for b in B]`.

## Python dict, set and generator comprehensions
Dict: `{w: len(w) for w in words}`. Set: `{x % 3 for x in nums}`. Generator expression: `(x * x for x in nums)`, which uses round brackets and makes values one at a time instead of building a list.

## Python dict basics
A dict maps keys to values: `d = {"a": 1, "b": 2}`. Read with `d["a"]` (raises `KeyError` if missing), set with `d["c"] = 3`, test with `"a" in d`, count with `len(d)`. Keys must be hashable (strings, numbers, tuples). Dicts keep insertion order.

## Python dict.get and setdefault
`d.get("k")` returns `None` when the key is missing and `d.get("k", 0)` returns the default you give. `d.setdefault("k", [])` returns the value for `k`, first inserting `[]` if the key was absent. Example: `groups.setdefault(key, []).append(item)`.

## Python looping over a dict: keys, values, items
`for k in d:` loops over keys. `d.values()` gives values and `d.items()` gives `(key, value)` pairs: `for k, v in d.items():`. Do not add or remove keys while looping over the same dict; loop over `list(d)` instead.

## Python dict update, merge, pop, del
`d.update(other)` copies keys from `other` into `d`, overwriting duplicates. `d1 | d2` makes a merged new dict (Python 3.9 and later). `d.pop("k")` removes a key and returns its value, `d.pop("k", None)` avoids `KeyError`, and `del d["k"]` removes it.

## Python sorting a dict by value
Sort the items and pass a key function: `sorted(d.items(), key=lambda kv: kv[1])`. Add `reverse=True` for largest first. To get a dict again, wrap it: `dict(sorted(d.items(), key=lambda kv: kv[1]))`. The largest key alone: `max(d, key=d.get)`.

## Python set operations
A set holds unique, unordered items: `s = {1, 2, 3}`; `{}` is an empty dict, so use `set()` for an empty set. `add(x)`, `discard(x)` (no error if absent), `remove(x)` (error if absent). `a | b` union, `a & b` intersection, `a - b` difference, `a ^ b` items in only one. `set(items)` removes duplicates.

## Python tuple basics
A tuple is an ordered collection that cannot be changed: `t = (1, 2, 3)`. A one-item tuple needs a comma: `(1,)`. Tuples can be dict keys and set members when their contents are hashable. A function returning `a, b` returns a tuple.

## Python unpacking and swapping values
`x, y = (1, 2)` assigns by position. Swap with `a, b = b, a`. A starred name collects the rest: `first, *rest = [1, 2, 3]` gives `first = 1` and `rest = [2, 3]`. Use `_` for a value you ignore.

## Python min, max, sum, len, abs, any, all
`min(a)`, `max(a)`, `sum(a)` and `len(a)` work on iterables; `max(words, key=len)` picks the longest. `any(iterable)` is `True` if at least one item is true, `all(iterable)` if every item is true. `abs(-3)` is `3`.

## Python map() and filter()
`map(f, items)` applies `f` to each item and `filter(f, items)` keeps items where `f(item)` is true. Both return lazy iterators, so wrap in `list(...)`: `list(map(str, [1, 2]))` is `["1", "2"]`. A list comprehension is usually clearer.

## Python functions: def, return, default arguments
`def add(a, b=0): return a + b` defines a function with a default for `b`. Call with positions `add(1, 2)` or keywords `add(a=1, b=2)`. A function without `return` returns `None`. A function can return several values as a tuple.

## Python mutable default argument pitfall
A default such as `def f(x, items=[])` is created once and shared by every call, so appends accumulate between calls. Use `None` as the default and make the list inside: `if items is None: items = []`.

## Python *args and **kwargs
In a definition, `*args` collects extra positional arguments into a tuple and `**kwargs` collects extra keyword arguments into a dict. In a call, `f(*lst)` unpacks a list into positions and `f(**d)` unpacks a dict into keywords.

## Python lambda
A lambda is a small unnamed function with one expression: `square = lambda x: x * x`. It is mostly used as a short argument: `sorted(people, key=lambda p: p["age"])`. For anything longer than one expression, use `def`.

## Python variable scope: local, global, nonlocal
Python looks up a name in the local scope, then enclosing functions, then global, then built-ins. Assigning to a name inside a function makes it local. Use `global x` to assign a module-level variable and `nonlocal x` to assign one from an enclosing function.

## Python classes: class, __init__, self
`class Dog:` defines a class. `__init__` runs when you create an object, and `self` is the object itself: `def __init__(self, name): self.name = name`. Create with `d = Dog("Rex")` and read `d.name`. Methods take `self` as the first parameter.

## Python inheritance and super()
`class Puppy(Dog):` makes `Puppy` a subclass of `Dog`, inheriting its methods. A subclass can override a method and call the parent version with `super().method()`. In `__init__` use `super().__init__(name)` to run the parent's setup.

## Python __str__ and __repr__
`__str__` returns the readable text used by `print(obj)` and `str(obj)`. `__repr__` returns an unambiguous text for developers, shown in the REPL and in lists. If only `__repr__` exists, it is also used for printing.

## Python @property, @staticmethod, @classmethod
`@property` lets a method be read like an attribute (`obj.area`). `@staticmethod` is a plain function placed in a class, without `self`. `@classmethod` receives the class as `cls` and is often used for alternative constructors: `Date.from_string(s)`.

## Python dataclasses
`@dataclass` (from the `dataclasses` module) writes `__init__`, `__repr__` and `__eq__` for you from annotated fields: `@dataclass class P:` with `x: int` and `y: int = 0`. For a mutable default use `field(default_factory=list)`. `@dataclass(frozen=True)` makes instances read-only.

## Python try, except, else, finally
`try:` runs code that may fail. `except ValueError as e:` handles that error. `else:` runs only if no error happened. `finally:` always runs, even after `return` or an error, and is used for cleanup. Catch specific errors; avoid a bare `except:`.

## Python raise and custom exceptions
`raise ValueError("bad input")` throws an error. A bare `raise` inside `except` re-throws the current one. Make your own with `class AppError(Exception): pass`. Catch several types with `except (ValueError, TypeError):`.

## Python common exceptions: ValueError, TypeError, KeyError, IndexError
`ValueError`: right type, bad value (`int("abc")`). `TypeError`: wrong type (`"a" + 1`). `KeyError`: missing dict key. `IndexError`: list index out of range. `FileNotFoundError`: file missing. `ZeroDivisionError`, `AttributeError` and `ImportError` are also common.

## Python assert statement
`assert x > 0, "x must be positive"` raises `AssertionError` with that message when the test is false. It is meant for catching programmer mistakes and tests, not for checking user input, because assertions can be switched off with `python3 -O`.

## Python with statement and context managers
`with open("a.txt") as f:` runs a block and then always closes the file, even after an error. Any object that defines `__enter__` and `__exit__` works in `with`. Several can be combined: `with open(a) as f, open(b, "w") as g:`.

## Python reading and writing files
`with open("a.txt", encoding="utf-8") as f:` then `f.read()` (all text), `f.readline()` (one line) or `for line in f:` (line by line). Modes: `"r"` read, `"w"` write (erases the file), `"a"` append, `"rb"` and `"wb"` for bytes. Write with `f.write("text\n")`.

## Python generators and yield
A function with `yield` returns a generator that produces values one at a time and remembers where it stopped. `def count(n):` with `for i in range(n): yield i` makes values lazily. A generator can be used only once, and `next(g)` takes the next value.

## Python decorators
A decorator is a function that takes a function and returns a replacement. `@timer` above `def work():` means `work = timer(work)`. A typical decorator defines an inner `wrapper(*args, **kwargs)` that calls the original function and returns the wrapper.

## Python functools.wraps
Inside a decorator, put `@functools.wraps(func)` on the wrapper function. It copies the original function's name and docstring onto the wrapper, so `help()` and error messages still show the real function.

## Python modules and import
Any `.py` file is a module. `import math` then `math.sqrt(9)`; `from math import sqrt` then `sqrt(9)`; `import numpy as np` renames it. Python searches the script's folder first, so a file named `random.py` or `json.py` of your own hides the standard module.

## Python packages and __init__.py
A package is a folder of modules, imported with dots: `from mypkg.tools import helper`. A folder with an `__init__.py` file is a regular package, and that file runs when the package is imported. Relative imports inside a package look like `from . import helper`.

## Python if __name__ == "__main__"
When a file is run directly, `__name__` is `"__main__"`; when it is imported, `__name__` is the module name. Code under `if __name__ == "__main__":` therefore runs only when the file is run as a script, not when it is imported.

## Python running scripts: python3 script.py and -m
Run a file with `python3 script.py`. Run a module by name with `python3 -m module`, for example `python3 -m venv .venv`, `python3 -m http.server` or `python3 -m unittest`. `python3 -c "print(1)"` runs a one-line command. `python3` alone opens the interactive prompt.

## Python virtual environment: venv
A virtual environment keeps one project's packages separate. Create it with `python3 -m venv .venv`, activate on Linux or macOS with `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`), leave with `deactivate`. Packages installed while it is active go into `.venv` only.

## Python pip: install, uninstall, list, freeze
`pip install requests` installs a package, `pip uninstall requests` removes it, `pip list` shows installed packages, `pip install -r requirements.txt` installs a list. `pip freeze > requirements.txt` saves the exact versions. Safer form: `python3 -m pip ...`, which uses the same Python you run.

## Python input() and print()
`input("Name: ")` waits for a line and always returns a `str`; convert with `int(input())`. `print("a", "b", sep="-", end="")` prints `a-b` without a newline. `print(..., file=sys.stderr)` writes to the error stream.

## Python isinstance and type checks
`isinstance(x, int)` is the usual type test and it also accepts subclasses; `isinstance(x, (int, float))` accepts several types. `type(x) is int` is exact. Python is dynamically typed, so a name can point to any type.

## Python type hints with typing
Hints document types but are not enforced when the program runs: `def f(x: int, name: str = "a") -> bool:`. Built-in generics are written `list[int]` and `dict[str, int]` (Python 3.9 and later). `int | None` (3.10 and later) or `Optional[int]` means int or None. Tools such as mypy check them.

## Python pathlib: Path objects
`from pathlib import Path` then `p = Path("data") / "a.txt"` joins paths. Useful: `p.exists()`, `p.is_file()`, `p.name`, `p.stem`, `p.suffix`, `p.parent`, `p.read_text()`, `p.write_text("hi")`, `Path.home()`, `p.mkdir(parents=True, exist_ok=True)`, `Path(".").glob("*.py")`, `p.iterdir()`.

## Python os module: environment, folders, paths
`os.getcwd()` gives the current folder, `os.listdir(".")` lists names, `os.makedirs(path, exist_ok=True)` creates folders, `os.remove(path)` deletes a file, `os.path.join("a", "b")` joins a path, `os.environ.get("HOME")` reads an environment variable.

## Python sys module: argv, exit, path
`sys.argv` is the list of command-line arguments; `sys.argv[0]` is the script name. `sys.exit(1)` stops with an exit code (0 means success). `sys.path` is the list of folders searched for imports. `sys.stdin` and `sys.stdout` are the standard streams.

## Python json module
`json.loads(text)` turns a JSON string into Python objects, `json.dumps(obj, indent=2)` turns objects into a JSON string. `json.load(f)` and `json.dump(obj, f)` do the same with open files. JSON `null` becomes `None`; add `ensure_ascii=False` to keep non-English letters readable.

## Python csv module
`with open("a.csv", newline="", encoding="utf-8") as f:` then `csv.reader(f)` yields rows as lists, and `csv.DictReader(f)` yields rows as dicts keyed by the header. For output use `csv.writer(f).writerow([...])` or `csv.DictWriter`. Always open CSV files with `newline=""`.

## Python re module: regular expressions
`re.search(p, s)` finds the first match anywhere, `re.match(p, s)` only at the start, `re.fullmatch(p, s)` needs the whole string. `re.findall(p, s)` returns all matches, `re.sub(p, repl, s)` replaces. `m.group(1)` reads a group. Use raw strings: `re.search(r"\d+", s)`. Flag: `re.IGNORECASE`.

## Python datetime module
`from datetime import datetime, date, timedelta`. `datetime.now()` is the current local time, `date.today()` today's date. `datetime.strptime("2024-01-31", "%Y-%m-%d")` parses text; `d.strftime("%d/%m/%Y")` formats it; `d.isoformat()` gives ISO text. `d + timedelta(days=7)` adds a week.

## Python math module
`math.sqrt(16)` is `4.0`, `math.floor(2.7)` is `2`, `math.ceil(2.1)` is `3`, `math.pi`, `math.log(x)`, `math.log10(x)`, `math.pow`, `math.gcd(a, b)`. To compare floats use `math.isclose(0.1 + 0.2, 0.3)`, which is `True`, instead of `==`.

## Python random module
`random.random()` is a float from 0 up to 1, `random.randint(1, 6)` includes both ends, `random.choice(seq)` picks one, `random.sample(seq, 3)` picks 3 distinct, `random.shuffle(lst)` shuffles in place. `random.seed(1)` makes results repeatable. For passwords and tokens use the `secrets` module instead.

## Python collections.Counter
`Counter` counts items: `from collections import Counter` then `c = Counter("banana")` gives `{'a': 3, 'n': 2, 'b': 1}`. `c.most_common(2)` returns the 2 most frequent as `(item, count)` pairs. Missing items count as 0.

## Python collections.defaultdict
`defaultdict(list)` is a dict that creates a missing value on first access, so `groups[key].append(x)` works without checking the key. Other factories: `defaultdict(int)` for counters, `defaultdict(set)`. Import with `from collections import defaultdict`.

## Python collections.deque
`deque` is a double-ended queue with fast adds and removes at both ends: `append`, `appendleft`, `pop`, `popleft`. `deque(maxlen=3)` keeps only the last 3 items. Use it instead of a list when taking items from the front, since `list.pop(0)` is slow on large lists.

## Python collections.namedtuple
`namedtuple` makes a tuple whose fields have names: `Point = namedtuple("Point", ["x", "y"])` then `p = Point(1, 2)`, `p.x` and `p[0]` both give 1. It is read-only. For a changeable version, use a dataclass.

## Python itertools
`itertools` builds iterators: `chain(a, b)` joins sequences, `product(a, b)` gives all pairs, `permutations(s, 2)` ordered picks, `combinations(s, 2)` unordered picks, `islice(it, 5)` takes the first 5, `count()` counts forever, `cycle(s)` repeats forever. `groupby(data, key)` groups only neighbours, so sort first.

## Python functools: lru_cache, partial, reduce
`@lru_cache(maxsize=None)` caches results of a function by its arguments, which speeds up recursion like Fibonacci. `partial(f, 2)` fixes the first argument of `f`. `reduce(f, items)` folds a list into one value: `reduce(lambda a, b: a * b, [1, 2, 3, 4])` is `24`.

## Python subprocess.run: running commands
`subprocess.run(["ls", "-l"], capture_output=True, text=True, check=True)` runs a command and returns an object with `.stdout`, `.stderr` and `.returncode`. `check=True` raises an error on a non-zero exit. Pass the command as a list; avoid `shell=True` with text a user typed, since it allows command injection.

## Python argparse
`argparse` reads command-line options: `p = argparse.ArgumentParser(description="...")`, `p.add_argument("path")`, `p.add_argument("--count", type=int, default=1)`, `p.add_argument("-v", action="store_true")`, then `args = p.parse_args()` and read `args.count`. `-h` prints help automatically.

## Python logging module
`logging.basicConfig(level=logging.INFO)` then `logging.info("started")`, `logging.warning(...)`, `logging.error(...)`. Levels from low to high: DEBUG, INFO, WARNING, ERROR, CRITICAL. The default level is WARNING, so debug and info messages are hidden until you lower it. `logging.getLogger(__name__)` gives a named logger.

## Python unittest
`class TestAdd(unittest.TestCase):` with methods named `test_something` that call `self.assertEqual(add(1, 2), 3)`, `assertTrue`, `assertRaises(ValueError)`. Run with `python3 -m unittest` (it finds files named `test*.py`) or `python3 -m unittest -v` for names.

## Python sqlite3 module
`conn = sqlite3.connect("app.db")` opens or creates a database file. `conn.execute("SELECT * FROM t WHERE id = ?", (5,))` runs a query with a safe parameter; `.fetchone()` and `.fetchall()` read rows. After inserts or updates call `conn.commit()`. Finish with `conn.close()`.

## Python urllib: fetching a URL
`from urllib.request import urlopen` then `with urlopen("https://example.com") as r: body = r.read().decode("utf-8")`. `urllib.parse.urlparse(url)` splits a URL and `urllib.parse.urlencode({"q": "a b"})` builds a query string. Errors raise `urllib.error.URLError`.

## Python slicing and indexing strings: examples
`s = "hello"`: `s[0]` is `"h"`, `s[-1]` is `"o"`, `s[1:4]` is `"ell"`, `s[::-1]` is `"olleh"`. A string is an immutable sequence, so build changes as new strings.

## Python sorted() with key and multiple fields
`sorted(people, key=lambda p: (p["age"], p["name"]))` sorts by age then name. `key=str.lower` sorts text ignoring case. Add `reverse=True` for descending. `operator.itemgetter("age")` is a faster alternative to a lambda.

## Python pass, continue and empty blocks
`pass` does nothing and fills a block that Python needs syntactically: `def todo(): pass` or `except KeyError: pass`. Ignoring errors silently with `pass` hides bugs, so log or comment why.

## Python list membership speed: list vs set
`x in my_list` checks items one by one, so it slows down as the list grows. `x in my_set` and `x in my_dict` use hashing and are fast on average. Convert once with `lookup = set(items)` when you test many times.

## Python string formatting with format() and %
`"{} is {}".format(a, b)` and `"{0:.2f}".format(x)` are the older form of f-strings. `"%s is %d" % (name, n)` is the oldest form. Prefer f-strings in new code.
