# General programming concepts

## Variable (named storage for a value)
A variable is a named place in memory that holds a value, which the program can read and change later. After `count = 3`, the name count holds 3, and after `count = 4` it holds 4.

## Constant (fixed value, unchangeable)
A constant is a named value that is set once and cannot be changed while the program runs. A program might define `PI` once and use it everywhere, so the value cannot be changed by accident; some languages enforce this strictly and others only by convention.

## Data types (integer, float, string, boolean)
A data type says what kind of value something holds and which operations make sense for it. The common basic types are integer for whole numbers such as `42`, float for decimal numbers such as `3.14`, string for text such as `"hello"`, and boolean for true or false.

## Operators (arithmetic, comparison, logical operators)
An operator is a symbol that combines or compares values, such as `+` for adding, `*` for multiplying and `==` for checking whether two values are equal. Operators follow precedence rules, so in `2 + 3 * 4` the multiplication happens first and the result is 14.

## Expression vs statement (code that yields a value)
An expression is code that produces a value, such as `2 + 3`. A statement is code that does something, such as assigning a value or printing text, and it may or may not produce a value.

## Condition (if statement, boolean test)
A condition is a test that evaluates to true or false and decides which code runs. In `if age >= 18`, the test after `if` is the condition, and the block under it runs only when the test is true.

## Loop (for loop, while loop)
A loop repeats a block of code while a condition holds, or once for each item in a collection. A loop can print the numbers from 1 to 10 with one line of code that runs ten times.

## Function (method, subroutine)
A function is a named block of code that takes inputs, does a job and can return a result. Calling its name runs the job again, so code written once can be reused many times.

## Parameter vs argument (function inputs)
A parameter is a variable named in a function's definition, while an argument is the actual value passed in when the function is called. In a definition like `greet(name)`, name is a parameter; in the call `greet("Ada")`, the text "Ada" is the argument.

## Return value (function output)
A return value is the result a function hands back to the code that called it. A function `add(a, b)` that returns `a + b` gives its caller the sum, which the caller can store or use.

## Scope (variable visibility, local and global variables)
Scope is the part of a program where a variable can be seen and used. A variable created inside a function is usually local to it and cannot be read outside, while a global variable can be used anywhere in the program.

## Recursion (recursive function)
Recursion is when a function calls itself to solve a smaller version of the same problem. It needs a base case that stops the calls, such as factorial, where 0! is 1, because otherwise the calls never end.

## Array (fixed-size list, index)
An array is an ordered collection of values stored together, where each value is found by its index number, starting at 0. Reading by index is fast, and in many languages an array has a fixed size set when it is created.

## List (ordered collection, dynamic array)
A list is an ordered collection of items that can grow and shrink, and whose items can be read, added or removed by position. Many languages provide lists that resize automatically, unlike a fixed-size array.

## Dictionary (map, key-value pairs)
A dictionary, also called a map, stores pairs of a key and a value and finds a value quickly from its key. For example, `{"apple": 3}` links the key apple to the value 3.

## Set (unique values, no duplicates)
A set is a collection that stores each value at most once and usually has no meaningful order. It is useful for checking whether an item is present, such as whether a username is already taken.

## Stack (LIFO, last in first out)
A stack is a collection where the last item added is the first one removed, like a pile of plates. Programs use stacks for undo history, for tracking function calls and for checking matching brackets.

## Queue (FIFO, first in first out)
A queue is a collection where the first item added is the first one removed, like people waiting in a line. Queues handle print jobs, task scheduling and requests that must be served in order.

## Linked list (nodes and pointers)
A linked list is a chain of nodes, where each node holds a value and a reference to the next node. Adding or removing a node is quick once you have its position, but finding the nth item means walking down the chain from the start.

## Tree (hierarchy, root node)
A tree is a hierarchy of nodes joined by links, with one top node called the root and each other node having exactly one parent. Folders on a computer are often organised as a tree.

## Binary search tree (BST)
A binary search tree is a tree where each node has at most two children, with smaller values in the left subtree and larger values in the right. When the tree is balanced, each comparison lets a search skip about half of the remaining nodes.

## Heap (priority queue, min-heap, max-heap)
A heap is a tree-shaped structure where every parent is no larger than its children (a min-heap) or no smaller than them (a max-heap). The smallest or largest item is therefore always at the top, which makes heaps good for priority queues.

## Graph (vertices and edges, network)
A graph is a set of nodes, called vertices, joined by edges, which can be one-way or two-way and can form loops. Graphs model road maps, networks of computers and links between web pages.

## Hash table (hash map, lookup by key)
A hash table stores key-value pairs in slots chosen by a hash function, which turns a key into a number that points to a slot. Lookups, inserts and deletes take about the same time whatever the table size, unless many keys land in the same slot.

## O(1) (constant time)
O(1) means an operation takes the same time however much data there is. Reading an array element by its index, such as `numbers[5]`, takes the same time whether the array holds ten items or ten million.

## O(log n) (logarithmic time)
O(log n) means the time grows by one step each time the data doubles, so very large inputs stay fast. Binary search over a sorted list of a million items needs about 20 comparisons, because 2 to the power of 20 is about a million.

## O(n) (linear time)
O(n) means the time grows in step with the number of items n, so twice the data takes twice the work. Looping once through a list to find its largest value takes O(n) time.

## O(n log n) (linearithmic time)
O(n log n) is the time that the best general comparison sorts, such as merge sort, need to sort n items. It grows a little faster than linear, and it is far faster than O(n^2) for large inputs.

## O(n^2) (quadratic time)
O(n^2) means the work grows with the square of the number of items, so twice the data takes about four times the work. A double loop that compares every item with every other item, such as `for a in items: for b in items`, is O(n^2).

## Linear search (sequential search)
Linear search finds an item by checking each element in order from the start until it finds a match or reaches the end. It works on any list, sorted or not, and takes O(n) time in the worst case.

## Binary search (search sorted list)
Binary search finds an item in a sorted list by checking the middle element and discarding the half that cannot hold the target, then repeating on the other half. It takes O(log n) time, far faster than linear search on large sorted data.

## Bubble sort (simple sorting algorithm)
Bubble sort repeatedly walks through a list, swapping neighbouring items that are out of order, until a full pass makes no swaps. It is easy to understand but takes O(n^2) time on average, so it suits only small lists and teaching.

## Insertion sort (sorting algorithm)
Insertion sort grows a sorted part at the front of a list by taking the next item and inserting it into its place among the sorted ones. It takes O(n^2) time in the worst case but is fast on small or nearly sorted lists.

## Merge sort (divide and conquer sort)
Merge sort splits a list in half, sorts each half the same way, and merges the two sorted halves into one. It always takes O(n log n) time, but it needs extra memory for the merged result.

## Quicksort (pivot sort)
Quicksort picks a pivot value, moves smaller items before it and larger items after it, then sorts both sides the same way. It is usually fast, at O(n log n) on average, but its worst case is O(n^2) when the pivots are chosen badly.

## Breadth-first search (BFS, graph search)
Breadth-first search explores a graph level by level, visiting all the neighbours of a node before going further, and it uses a queue to do so. On a graph where all edges have equal length, it finds the path with the fewest edges from the start.

## Depth-first search (DFS, graph search)
Depth-first search follows one path as far as it can go before backtracking, usually with recursion or a stack. It is used to check whether a path exists, to find connected parts of a graph and to order tasks that depend on each other.

## Dijkstra's algorithm (shortest path)
Dijkstra's algorithm finds the shortest route from one starting node to every other node in a graph whose edges have no negative lengths. It always works on the closest node not yet settled, and updates the distances to that node's neighbours as it goes.

## Dynamic programming (memoization, overlapping subproblems)
Dynamic programming solves a problem by breaking it into smaller subproblems that overlap, and storing each answer so it is never computed twice. Computing Fibonacci numbers with a stored table takes linear time, while plain recursion takes exponential time.

## Greedy algorithm (greedy method)
A greedy algorithm makes the best-looking choice at each step and never goes back to change it. It is fast and gives the best answer for some problems, but for others it can give a poor result.

## Object-oriented programming (OOP, objects and classes)
Object-oriented programming organises code into objects that bundle data with the functions that work on that data. Objects interact by calling each other's methods.

## Class vs object (blueprint and instance)
A class is a blueprint that describes the data and behaviour of a kind of thing, while an object is one actual instance made from that class. A class `Car` is like a drawing of a car, and each car built from the drawing is an object.

## Inheritance (subclass, extending a class)
Inheritance lets a new class take on the data and methods of an existing class and then add to them or change some of them. A `Dog` class can inherit from an `Animal` class and only define what is special about dogs.

## Encapsulation (data hiding)
Encapsulation keeps an object's data inside the object and lets other code use it only through defined methods. This stops outside code from putting the object into an invalid state by changing its fields directly.

## Polymorphism (same call, different behaviour)
Polymorphism lets the same method call behave differently depending on the object it is called on. A `draw()` call on a circle and on a square each draws its own shape, while the calling code stays the same.

## Interface (contract, abstract methods)
An interface is a list of methods that a class must provide, without saying how they work. Code can depend on the interface and work with any class that implements it.

## Functional programming (pure functions style)
Functional programming is a style that builds programs from functions that take inputs and return outputs, and that avoids changing shared data. Code written this way is often easier to test and to reason about, especially when parts of a program run at the same time.

## Pure function (no side effects)
A pure function always returns the same output for the same inputs and changes nothing outside itself. A function that adds two numbers is pure, while a function that writes to a file or reads the current time is not.

## Immutability (immutable data)
An immutable value cannot be changed after it is created; a change produces a new copy instead. Immutable data avoids surprises from code that alters a value elsewhere, and it is safer when several parts of a program share it.

## Higher-order function (callbacks, functions as values)
A higher-order function takes another function as an argument, returns a function, or both. For example, a `map` function applies a given function to every item in a list and returns a new list with the results.

## Static vs dynamic typing (type checking)
In static typing, the types of values are checked before the program runs, while in dynamic typing they are checked while it runs. Static checking catches many mistakes early, and dynamic typing allows more flexible code.

## Compiled vs interpreted (compiler, interpreter)
A compiled language is translated into machine code by a compiler before the program runs, while an interpreted language is run line by line by an interpreter. Many languages use both approaches, and the difference affects speed and how quickly code can be tried out.

## Pointer (memory address)
A pointer is a variable that holds the memory address of another value, so the program can reach that value through it. Pointers let code share or change data in place, and they are an easy source of bugs when used carelessly.

## Reference (alias, object reference)
A reference is a way to refer to an object or value so that code works on the same data rather than a copy. In many languages, passing a reference to a function means the function works on the original object.

## Garbage collection (automatic memory management)
Garbage collection is an automatic process that finds memory the program can no longer reach and frees it for reuse. It saves programmers from freeing memory by hand, though it can briefly pause the program while it runs.

## Stack vs heap memory (memory layout)
Stack memory holds local variables and function call data, and it is freed automatically when a function returns. Heap memory holds data of flexible size and lasts until it is freed, which makes it more flexible but also more work to manage.

## Concurrency vs parallelism (multitasking)
Concurrency means several tasks make progress by taking turns, which can happen on a single processor. Parallelism means several tasks run at the same instant on separate processors, so a program can be concurrent without being parallel.

## Thread (multithreading)
A thread is a separate path of execution inside a program that can run alongside other threads and share the same memory. Threads let a program do several things at once, but shared data then needs careful handling.

## Race condition (data race)
A race condition is a bug where the result depends on the timing of two or more threads or processes that use shared data. For example, two threads each adding 1 to the same counter at the same moment can lose one of the additions.

## Deadlock (waiting forever)
A deadlock happens when two or more threads each wait for a resource that the other holds, so none of them can continue. Acquiring locks in the same order everywhere is a common way to prevent it.

## Mutex (lock, mutual exclusion)
A mutex, short for mutual exclusion, is a lock that only one thread can hold at a time. A thread locks the mutex before touching shared data and unlocks it afterwards, so only one thread works on that data at once.

## Async (asynchronous code, async await)
Asynchronous code starts a slow task, such as reading a file or waiting for a network reply, and carries on with other work instead of waiting. When the task finishes, the program resumes the code that was waiting for its result.

## Exception handling (try catch, errors)
Exception handling deals with errors by raising an exception when something goes wrong, and catching it in code that knows how to respond. It keeps error handling separate from the normal logic, so failures are not silently ignored.

## Null (null value, null pointer)
Null is a special value that means no value is there, often used for a missing item. Using null where the code expects a real object causes a null reference error, a common source of crashes.

## Off-by-one error (fencepost error)
An off-by-one error is a mistake in which a loop or index goes one step too far or stops one step short. These errors are common with arrays, which start at index 0 and end one less than their length.

## Integer overflow (wraparound)
Integer overflow happens when a calculation produces a number too large for the integer type to store, so the result wraps around. In languages with 32-bit signed integers, 3,000,000,000 cannot be stored and wraps to a negative number.

## Floating-point error (rounding error, precision)
Floating-point error is a small inaccuracy caused because most decimal fractions cannot be stored exactly in binary. In many languages `0.1 + 0.2` gives 0.30000000000000004, so floats should be compared with a small tolerance rather than exact equality.

## Unit test (automated test)
A unit test checks one small piece of code, such as a single function, on its own by giving it inputs and comparing the output with the expected result. Unit tests run quickly and show exactly which piece broke.

## Integration test (end-to-end test)
An integration test checks that several parts of a program work together, such as a function that reads from a real database. These tests are slower than unit tests, but they catch problems that appear only where the parts meet.

## Test-driven development (TDD, red green refactor)
Test-driven development means writing a failing test for a new behaviour first, then writing just enough code to make it pass, and then tidying the code. The cycle repeats for each small change.

## Debugging (finding bugs, fixing bugs)
Debugging is finding and fixing the cause of a bug. A good method is to make the bug happen reliably, form a guess about its cause, and test that guess, using the smallest input that still shows the problem.

## Breakpoint (debugger breakpoint)
A breakpoint is a marker set on a line of code that pauses a debugger when the program reaches it. While paused, you can inspect the variables and step through the code one line at a time.

## Logging (log messages, log file)
Logging records messages from a running program, such as events, warnings and errors, to a file or console so that you can see later what happened. Useful log messages say what happened and when, and they never include passwords.

## Refactoring (restructuring code)
Refactoring changes the structure of code without changing what it does, such as renaming a function or splitting a long one into smaller ones. Tests confirm that the behaviour is unchanged after each step.

## Code review (peer review)
Code review is the practice of having another person read a change before it is merged, to find bugs, unclear code and design problems. It also spreads knowledge of the codebase across a team.

## Linter (static analysis, lint)
A linter is a tool that reads source code without running it and reports likely bugs, risky patterns and style problems. For example, it can flag a variable that is declared but never used.

## Formatter (code formatting, auto format)
A formatter rewrites source code into a consistent layout, handling spacing, line breaks and indentation automatically. Because the output is the same for everyone who runs it, it ends arguments about style.

## Semantic versioning (SemVer, version numbers)
Semantic versioning labels a release as major.minor.patch, such as 2.4.1. The major number goes up for changes that break existing users, the minor number for new features that stay compatible, and the patch number for compatible bug fixes.

## Dependency (library, external code)
A dependency is external code, such as a library, that a program uses but does not write itself. Each dependency adds maintenance work and a possible source of bugs, so it is wise to include only the ones you need.

## Package manager (install libraries)
A package manager is a tool that downloads, installs, updates and removes dependencies, and keeps track of the versions a project uses. Many record the exact versions in a lock file so that the same versions are installed again later.

## Build (compile, build step)
A build is the step that turns source code and its dependencies into a program that can run. It can include compiling, packaging files and running checks, and it is usually scripted so that it can be repeated.

## CI (continuous integration)
Continuous integration automatically builds and tests every change that is pushed to a shared code project. A CI server runs the same steps each time, so a change that breaks the build is reported straight away.

## API (application programming interface)
An API is a defined set of functions, messages or web addresses that one program offers so that other programs can use it without knowing how it works inside. A weather API, for example, might let a program ask for today's temperature in a single request.

## REST (REST API, RESTful)
REST is a style for web APIs in which each resource has its own URL, and standard HTTP methods act on it: GET reads, POST creates, PUT or PATCH changes and DELETE removes. Responses are usually in JSON.

## JSON data format (JSON file, text data)
JSON is a text format for structured data, built from objects with named keys, arrays, strings, numbers, true, false and null. For example, `{"name": "Ada", "age": 36}` is valid JSON, and nearly every programming language can read and write it.

## HTTP request (web request, HTTP methods)
An HTTP request is a message a client sends to a server to ask for something. It contains a method such as GET, a URL and optional headers and body, and the server replies with a status code, such as 200 for success or 404 for not found.

## Environment variable (env var, environment setting)
An environment variable is a named setting that the operating system or shell provides to a program when it starts. Programs read them for values that differ between machines, such as the address of a database.

## Configuration (config, settings)
Configuration is the set of settings that control how a program behaves without changing its code, such as a port number, a file path or a log level. Keeping settings apart from the code lets the same program run in different places.

## Singleton pattern (design pattern, one instance)
The singleton pattern makes sure a class has only one instance, shared wherever it is needed. It suits things like a single configuration object, though it is often overused.

## Factory pattern (design pattern, object creation)
The factory pattern moves object creation into a function or class, so that calling code asks for an object by type without knowing how it is built. It keeps the creation logic in one place.

## Observer pattern (design pattern, events)
The observer pattern lets objects subscribe to another object and be notified automatically when it changes. It is the idea behind many event systems, such as a button telling its listeners that it was clicked.

## Strategy pattern (design pattern, interchangeable algorithms)
The strategy pattern puts each algorithm behind a common interface so that it can be swapped at run time. A program can choose between several sorting or pricing methods without changing the code that uses them.

## DRY (don't repeat yourself)
DRY is the principle that each piece of knowledge should live in one clear place in the code, so a change needs to be made only once. Copying the same logic into five places means five places to fix when it changes.

## KISS (keep it simple)
KISS is the principle that the simplest design that solves the problem is usually the best one. Simple code is easier to read, test and change than clever code.

## YAGNI (you aren't going to need it)
YAGNI is the principle of not adding a feature or flexibility until it is actually needed. Code written for imagined future needs is often never used and still has to be maintained.

## SOLID (SOLID principles, object-oriented design)
SOLID is a set of five object-oriented design principles: each class has one responsibility, code is open to extension but closed to change, subtypes can replace their base types, interfaces are small and focused, and code depends on abstractions rather than concrete details. Together they aim at code that can change without breaking other parts.

## Pseudocode (plain-language algorithm)
Pseudocode is a plain-language description of an algorithm's steps, written loosely and without the rules of a particular language. Writing it first helps plan the code, and a line such as "for each item, if it is larger than the largest so far, remember it" is pseudocode.

## Comment (code comments)
A comment is text in source code that the computer ignores and that explains something to the human reader. A good comment says why something is done, while one that repeats what the code plainly says only adds noise.

## Naming conventions (camelCase, snake_case, PascalCase, kebab-case)
Naming conventions are agreed styles for writing names in code: camelCase as `userName`, snake_case as `user_name`, PascalCase as `UserName` and kebab-case as `user-name`. Use the style that your language or project already follows, and keep it consistent, because kebab-case is not allowed in names in most languages.
