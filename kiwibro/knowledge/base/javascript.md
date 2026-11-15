# JavaScript Reference

## JavaScript let, const and var
`let` declares a variable that can be reassigned, `const` declares one that cannot be reassigned, and both are limited to the block `{ }` they are in. `var` is the old form: it is limited to the function, not the block, and is hoisted. Prefer `const`, and use `let` when the value changes.

## JavaScript const does not freeze objects
`const` only stops the name from being assigned again. An object or array held by a `const` can still change: `const a = []; a.push(1);` works, while `a = [];` throws `TypeError`. `Object.freeze(obj)` blocks changes to the top level of an object.

## JavaScript data types
JavaScript has seven primitive types: `number`, `string`, `boolean`, `undefined`, `null`, `bigint` and `symbol`, plus `object` (which includes arrays and functions). There is one `number` type for both integers and decimals. Check with `typeof x`.

## JavaScript typeof operator
`typeof 5` is `"number"`, `typeof "a"` is `"string"`, `typeof undefined` is `"undefined"`, `typeof (() => 1)` is `"function"`. Quirks: `typeof null` is `"object"` and `typeof []` is `"object"`. Test for an array with `Array.isArray(x)`.

## JavaScript null vs undefined
`undefined` means a variable has no value yet or a property is missing. `null` is a value you set to mean "nothing on purpose". `null == undefined` is `true`, but `null === undefined` is `false`.

## JavaScript == vs ===
`===` compares value and type and is the safe choice: `1 === "1"` is `false`. `==` converts types first, with surprising results: `"1" == 1` is `true` and `0 == ""` is `true`. Use `===` and `!==` unless you deliberately want `x == null` to catch both `null` and `undefined`.

## JavaScript truthy and falsy values
These are falsy: `false`, `0`, `-0`, `0n`, `""`, `null`, `undefined` and `NaN`. Everything else is truthy, including `"0"`, `"false"`, `[]` and `{}`. So `if (arr.length)` tests for a non-empty array, but `if (arr)` is always true.

## JavaScript NaN and Number.isNaN
`NaN` means "not a number" and appears for results such as `Number("abc")` or `0 / 0`. It is not equal to itself: `NaN === NaN` is `false`. Test with `Number.isNaN(x)`.

## JavaScript number precision and 0.1 + 0.2
JavaScript numbers are 64-bit floating point, so `0.1 + 0.2` gives `0.30000000000000004`. For money, work in whole cents or round at the end. Integers are exact only up to `Number.MAX_SAFE_INTEGER`, which is `9007199254740991`; use `BigInt` (`123n`) beyond that.

## JavaScript converting to numbers: Number, parseInt, parseFloat
`Number("42")` is `42` and `Number("42px")` is `NaN`. `parseInt("42px", 10)` is `42` because it reads digits until the first non-digit. `parseFloat("3.5abc")` is `3.5`. `+"7"` also converts. Pass the base 10 to `parseInt`.

## JavaScript toFixed and rounding numbers
`(3.14159).toFixed(2)` returns the string `"3.14"`, not a number. `Math.round(2.5)` is `3`, `Math.floor(2.7)` is `2`, `Math.ceil(2.1)` is `3`, `Math.trunc(-2.7)` is `-2`. `Math.round(-2.5)` is `-2`, because halves round toward positive infinity.

## JavaScript Math functions
`Math.max(1, 5, 3)` is `5`, `Math.min(...arr)` spreads an array into arguments, `Math.abs(-4)` is `4`, `Math.sqrt(9)` is `3`, `Math.pow(2, 3)` or `2 ** 3` is `8`. `Math.random()` returns a number from 0 up to but not including 1. A random integer from 1 to 6 is `Math.floor(Math.random() * 6) + 1`.

## JavaScript string methods
Common ones: `s.length`, `s.includes("a")`, `s.startsWith("a")`, `s.endsWith("a")`, `s.indexOf("a")` (`-1` if absent), `s.slice(1, 4)`, `s.split(",")`, `s.trim()`, `s.toUpperCase()`, `s.padStart(3, "0")`, `s.at(-1)`. Strings are immutable: each method returns a new string.

## JavaScript replace and replaceAll
`"a-b-c".replace("-", "+")` replaces only the first match: `"a+b-c"`. `replaceAll("-", "+")` replaces all of them. `replace(/-/g, "+")` also replaces all, using a regular expression with the `g` flag. The original string is not changed.

## JavaScript template literals
Backticks make a template literal: `` `Hello ${name}, ${a + b}` `` inserts values and expressions in `${ }`. Template literals can span several lines without `\n`. Single and double quotes cannot do either.

## JavaScript function declaration vs expression
A declaration, `function add(a, b) { return a + b; }`, is hoisted, so it can be called before its line. An expression, `const add = function (a, b) { return a + b; };`, is not available until that line runs. A function with no `return` gives `undefined`.

## JavaScript arrow functions
`const add = (a, b) => a + b;` is a short function. With one parameter the brackets are optional, and a body in braces needs `return`. To return an object literal, wrap it in parentheses: `() => ({ a: 1 })`. Arrow functions have no `this` of their own and cannot be used with `new`.

## JavaScript default and rest parameters
`function f(a, b = 10) {}` uses 10 when `b` is `undefined`. `function f(...nums) {}` collects all arguments into a real array called `nums`. The rest parameter must be last.

## JavaScript closures
A closure is a function that keeps access to the variables of the place where it was created, even after that place has finished. Example: `function counter() { let n = 0; return () => ++n; }` and `const next = counter();` makes `next()` return 1, 2, 3 and so on.

## JavaScript this keyword
`this` depends on how a function is called. In `obj.method()` it is `obj`. In a plain call it is `undefined` in strict mode and modules (the global object otherwise). In `new F()` it is the new object. Arrow functions take `this` from the surrounding code, which is why they are used for callbacks inside methods.

## JavaScript call, apply and bind
`f.call(obj, a, b)` calls `f` with `this` set to `obj`. `f.apply(obj, [a, b])` does the same with an array of arguments. `f.bind(obj)` returns a new function with `this` fixed to `obj` and does not call it.

## JavaScript classes
`class Dog { constructor(name) { this.name = name; } bark() { return this.name; } static create() {} }`. Make objects with `new Dog("Rex")`. Fields starting with `#`, like `#age`, are private to the class. `get area() {}` defines a getter.

## JavaScript class inheritance: extends and super
`class Puppy extends Dog {}` inherits from `Dog`. In the subclass constructor, call `super(name)` before using `this`. `super.method()` calls the parent's version of a method. Under the hood JavaScript uses prototypes, and `class` is syntax for them.

## JavaScript objects: properties and access
`const user = { name: "Ann", age: 30 };`. Read with `user.name` or `user["name"]` (the second form allows variable keys). Add with `user.city = "Rome"`, remove with `delete user.age`, test with `"name" in user` or `Object.hasOwn(user, "name")`. Shorthand: `{ name }` means `{ name: name }`.

## JavaScript Object.keys, values and entries
`Object.keys(o)` returns the property names, `Object.values(o)` the values, `Object.entries(o)` an array of `[key, value]` pairs. `Object.fromEntries(pairs)` turns pairs back into an object. Example: `for (const [k, v] of Object.entries(o)) {}`.

## JavaScript destructuring objects
`const { name, age } = user;` pulls properties into variables. Rename with `{ name: userName }`, set a default with `{ age = 18 }`, take the rest with `{ name, ...others }`. It also works in parameters: `function f({ a, b }) {}`.

## JavaScript destructuring arrays
`const [first, second] = [1, 2, 3];` assigns by position. Skip with a gap: `[, , third]`. Take the rest: `[head, ...tail]`. Swap two variables: `[a, b] = [b, a];`.

## JavaScript spread operator
`...` expands an array or object in place. `[...a, ...b]` joins arrays, `{ ...obj, age: 31 }` copies an object and overrides one property, `Math.max(...nums)` passes items as arguments. Spread copies only the first level (shallow copy).

## JavaScript copying objects: shallow vs deep
`{ ...obj }`, `Object.assign({}, obj)` and `arr.slice()` make shallow copies: nested objects are still shared. `structuredClone(obj)` makes a deep copy of plain data. `JSON.parse(JSON.stringify(obj))` also works but drops functions and `undefined`, and turns dates into strings.

## JavaScript comparing objects and arrays
Objects are compared by reference: `{} === {}` is `false`, and `[1] === [1]` is `false`; only `a === a` is `true`. To compare content, compare each field, or compare `JSON.stringify` outputs for simple data.

## JavaScript array basics
`const a = [10, 20, 30];` `a.length` is 3, `a[0]` is 10, `a.at(-1)` is 30. `push(x)` adds at the end, `pop()` removes the last, `unshift(x)` adds at the start, `shift()` removes the first. `Array.isArray(a)` tests for an array.

## JavaScript array map
`[1, 2, 3].map(x => x * 2)` returns a new array `[2, 4, 6]`. The callback receives `(item, index, array)`. `map` does not change the original array and always returns an array of the same length.

## JavaScript array filter
`[1, 2, 3, 4].filter(x => x % 2 === 0)` returns `[2, 4]`: a new array with only the items for which the callback returns a truthy value. The original array is unchanged.

## JavaScript array reduce
`reduce` folds an array into one value: `[1, 2, 3].reduce((sum, x) => sum + x, 0)` is `6`. The callback gets the running result and the current item; the second argument to `reduce` is the starting value. Always give a starting value, because `reduce` on an empty array without one throws an error.

## JavaScript array find and findIndex
`arr.find(x => x.id === 3)` returns the first matching item, or `undefined`. `arr.findIndex(...)` returns its position, or `-1`. `arr.includes(5)` and `arr.indexOf(5)` test for an exact value.

## JavaScript array some and every
`arr.some(x => x > 5)` is `true` if at least one item passes the test. `arr.every(x => x > 5)` is `true` if all items pass; on an empty array `every` is `true` and `some` is `false`.

## JavaScript array sort
`sort()` changes the array in place and by default compares items as text, so `[10, 2, 1].sort()` gives `[1, 10, 2]`. For numbers pass a comparator: `arr.sort((a, b) => a - b)` for ascending, `(a, b) => b - a` for descending. Text: `names.sort((a, b) => a.localeCompare(b))`.

## JavaScript array slice vs splice
`arr.slice(1, 3)` returns a copy of items 1 and 2 and leaves `arr` unchanged. `arr.splice(1, 2)` removes 2 items from position 1 and changes `arr`. `arr.splice(1, 0, "x")` inserts `"x"` at position 1.

## JavaScript array join, concat, reverse, flat
`["a", "b"].join("-")` is `"a-b"`. `a.concat(b)` joins arrays into a new one. `a.reverse()` reverses in place. `[1, [2, [3]]].flat()` flattens one level, `.flat(Infinity)` flattens all. `flatMap(f)` maps then flattens one level.

## JavaScript Array.from and unique values
`Array.from("abc")` is `["a", "b", "c"]`, and `Array.from({ length: 3 }, (_, i) => i)` is `[0, 1, 2]`. Remove duplicates with `[...new Set(arr)]`. `Array.from(nodeList)` or `[...nodeList]` turns a DOM list into a real array.

## JavaScript for...of vs for...in vs forEach
`for (const x of arr)` loops over the items of an array, string, Set or Map and allows `break`. `for (const k in obj)` loops over the property names of an object; do not use it on arrays. `arr.forEach(fn)` calls a function per item but cannot be stopped with `break`.

## JavaScript Set and Map
`new Set([1, 1, 2])` holds unique values: `add`, `has`, `delete`, `size`. `new Map()` holds key-value pairs where keys can be any type: `set(k, v)`, `get(k)`, `has(k)`, `delete(k)`, `size`. Both keep insertion order and can be looped with `for...of`.

## JavaScript logical operators and short-circuit
`a && b` returns `a` if it is falsy, otherwise `b`. `a || b` returns `a` if it is truthy, otherwise `b`. So `name || "guest"` supplies a default and `user && user.name` avoids an error on missing `user`. `!!x` turns any value into a boolean.

## JavaScript optional chaining ?.
`user?.address?.city` returns `undefined` instead of throwing if `user` or `address` is `null` or `undefined`. It also works for arrays and calls: `arr?.[0]` and `obj.method?.()`.

## JavaScript nullish coalescing ?? and ??=
`a ?? b` returns `b` only when `a` is `null` or `undefined`. Unlike `||`, it keeps `0`, `""` and `false`: `0 ?? 5` is `0` but `0 || 5` is `5`. `x ??= 5` assigns only if `x` is `null` or `undefined`.

## JavaScript hoisting and temporal dead zone
Declarations are moved to the top of their scope before the code runs. `var x` exists early with the value `undefined`, and function declarations are fully available. `let` and `const` exist but cannot be used before their line: doing so throws `ReferenceError`.

## JavaScript strict mode
`"use strict";` at the top of a file or function turns silent mistakes into errors, such as assigning to an undeclared variable. ES modules and class bodies are always strict.

## JavaScript JSON.parse and JSON.stringify
`JSON.stringify(obj)` turns a value into a JSON string, and `JSON.stringify(obj, null, 2)` indents it. `JSON.parse(text)` turns a JSON string into a value and throws `SyntaxError` on invalid text. JSON needs double quotes and does not allow comments or trailing commas.

## JavaScript callbacks
A callback is a function passed to another function to be called later: `setTimeout(() => console.log("hi"), 1000)`. Many nested callbacks become hard to read, which is why promises and `async`/`await` were added.

## JavaScript promises
A promise stands for a value that will arrive later. It is pending, then fulfilled with a value or rejected with an error. Use `p.then(value => ...).catch(err => ...).finally(() => ...)`. Create one with `new Promise((resolve, reject) => { ... })`.

## JavaScript async and await
An `async` function always returns a promise. Inside it, `await p` pauses until the promise settles and gives its value, or throws if it was rejected: `const data = await getData();`. Use `try/catch` around `await` to handle errors. `await` also works at the top level of ES modules.

## JavaScript Promise.all, allSettled and race
`Promise.all([a, b])` waits for all and gives an array of results, but rejects as soon as one fails. `Promise.allSettled` waits for all and reports each success or failure. `Promise.race` settles with the first one to finish. Use `all` to run requests in parallel.

## JavaScript fetch
`const res = await fetch(url);` then `const data = await res.json();` (or `res.text()`). For a POST: `fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) })`. Browsers block cross-origin requests unless the server allows them (CORS).

## JavaScript fetch errors and res.ok
`fetch` rejects only on a network failure. A 404 or 500 response still resolves, so check `if (!res.ok) throw new Error(res.status)` before calling `res.json()`. `res.status` holds the HTTP code.

## JavaScript modules: import and export
In `a.js`: `export function f() {}` or `export default x`. In another file: `import { f } from "./a.js"` or `import x from "./a.js"`. Add the file extension in browsers. In HTML load them with `<script type="module" src="main.js">`. Node.js needs `.mjs` files or `"type": "module"` in `package.json`.

## JavaScript try, catch, finally and throw
`try { risky(); } catch (err) { console.error(err.message); } finally { cleanup(); }`. `finally` always runs. Throw errors with `throw new Error("message")`. `TypeError: Cannot read properties of undefined` means you used a property on a value that is `undefined`.

## JavaScript setTimeout and setInterval
`setTimeout(fn, 1000)` runs `fn` once after about 1000 ms. `setInterval(fn, 1000)` runs it repeatedly. Both return an id: `clearTimeout(id)` and `clearInterval(id)` cancel them. The delay is a minimum, not a guarantee.

## JavaScript event loop
JavaScript runs one piece of code at a time. Timers, network replies and events wait in queues until the running code finishes. Promise callbacks run before the next timer callback. A long loop therefore freezes the page, because nothing else can run.

## JavaScript Date
`new Date()` is now, `Date.now()` is milliseconds since 1 Jan 1970 UTC. `new Date("2024-03-05T10:00:00Z")` parses ISO text, `d.toISOString()` formats it, `d.getFullYear()`, `d.getDate()`, `d.getDay()` read parts. Months count from 0, so `d.getMonth()` is 0 for January.

## JavaScript toLocaleString and Intl formatting
`n.toLocaleString("en-US")` formats `1234567.5` as `"1,234,567.5"`. `new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(1234.5)` gives `"$1,234.50"`. `date.toLocaleDateString("en-GB")` formats a date for a locale.

## JavaScript regular expressions
`/\d+/g` is a pattern with the global flag. `re.test(s)` returns true or false, `s.match(re)` returns matches, `s.replace(re, "x")` replaces, `s.matchAll(re)` gives every match with groups. A regex with the `g` flag keeps a `lastIndex`, so reusing it with `test()` can give alternating results.

## JavaScript console methods
`console.log(x)` prints, `console.error(x)` and `console.warn(x)` print as error and warning, `console.table(arr)` shows an array of objects as a table, `console.dir(obj)` inspects an object. Open the browser console with F12.

## JavaScript Node.js basics
Node.js runs JavaScript outside the browser: `node app.js`. `process.argv` holds command-line arguments and `process.env.NAME` environment variables. Read files with `import fs from "node:fs"` and `fs.readFileSync("a.txt", "utf8")`, or `fs/promises` for the promise version.

## JavaScript npm basics
`npm init -y` creates `package.json`. `npm install pkg` adds a package to `node_modules` and records it in `package.json`. `npm install` installs everything listed there. `npm run build` runs the script named `build` from `package.json`. `npx tool` runs a package's command.

## DOM querySelector and querySelectorAll
`document.querySelector(".card")` returns the first element matching a CSS selector, or `null`. `document.querySelectorAll("li.item")` returns all matches as a NodeList, which supports `forEach` and `[...list]`. `document.getElementById("app")` finds one element by id. `el.closest(".box")` finds the nearest ancestor that matches.

## DOM textContent vs innerHTML
`el.textContent = "Hi"` sets plain text and is safe. `el.innerHTML = "<b>Hi</b>"` parses HTML, so putting text from users or from the network into it allows script injection (XSS). Use `textContent` unless you truly need markup.

## DOM addEventListener
`button.addEventListener("click", (event) => { ... });` runs the function when the event happens. Pass `{ once: true }` to run it only once. `removeEventListener` needs the same function object you added, so an inline arrow function cannot be removed. Common events: `click`, `input`, `change`, `submit`, `keydown`.

## DOM event object: target, preventDefault, stopPropagation
The handler receives an event. `event.target` is the element that was clicked, `event.currentTarget` is the element the listener is on, `event.preventDefault()` cancels the default action (a link jump, a form submit), `event.stopPropagation()` stops the event reaching parents. For keys use `event.key`.

## DOM event bubbling and delegation
Most events travel up from the clicked element to its parents (bubbling). Event delegation uses that: one listener on the list handles all items: `list.addEventListener("click", e => { const li = e.target.closest("li"); if (li) ... })`. It also works for items added later.

## DOM input and change events
The `input` event fires on every edit of a field, `change` fires when the value is committed (for a text box, when it loses focus). `input.value` is always a string, so convert with `Number(input.value)`. For checkboxes read `input.checked`.

## DOM form submit and FormData
`form.addEventListener("submit", e => { e.preventDefault(); const data = Object.fromEntries(new FormData(form)); })` stops the page reload and collects all named fields into an object. Only fields with a `name` attribute are included.

## DOM classList
`el.classList.add("on")`, `remove("on")`, `toggle("on")` (add if missing, remove if present), `contains("on")` and `replace("a", "b")` change classes without touching the others. Setting `el.className = "x"` replaces them all.

## DOM creating and removing elements
`const li = document.createElement("li"); li.textContent = "New"; list.append(li);` adds an element. `parent.prepend(el)` adds at the start, `el.remove()` deletes it, `el.cloneNode(true)` copies it with its children. `appendChild` is the older form that takes one node.

## DOM attributes, dataset and style
`el.getAttribute("href")` and `el.setAttribute("href", url)` read and write attributes. `data-user-id="5"` in HTML is `el.dataset.userId` in JavaScript. Set styles with `el.style.backgroundColor = "red"` (camelCase), or better, toggle a CSS class.

## DOM DOMContentLoaded and script placement
Code that touches the page must run after the elements exist. Use `<script src="app.js" defer></script>`, or `type="module"` (also deferred), or listen for `document.addEventListener("DOMContentLoaded", fn)`. A plain script in the head runs before the body is built.

## JavaScript localStorage and sessionStorage
`localStorage.setItem("k", "v")`, `getItem("k")` (null if missing), `removeItem("k")`, `clear()`. Data stays after the browser closes and is kept per site. It stores only strings, so save objects with `JSON.stringify` and read with `JSON.parse`. `sessionStorage` is the same but ends with the tab.

## JavaScript URL and URLSearchParams
`new URL("https://a.com/p?id=5")` gives `.hostname`, `.pathname`, `.searchParams`. `new URLSearchParams(location.search).get("id")` reads a query value. `encodeURIComponent("a b&c")` makes text safe to put inside a URL.

## JavaScript location and history
`location.href` is the current URL and assigning to it navigates. `location.reload()` reloads the page. `history.back()` goes to the previous page. `history.pushState(state, "", "/new")` changes the address bar without reloading.

## JavaScript requestAnimationFrame
`requestAnimationFrame(fn)` asks the browser to call `fn` just before the next screen repaint, usually 60 times a second. Call it again inside `fn` to animate. It pauses in background tabs, which makes it better than `setInterval` for animation.

## JavaScript copying to the clipboard
`await navigator.clipboard.writeText("text")` copies text, and `await navigator.clipboard.readText()` reads it. It works only on secure pages (HTTPS or localhost) and usually after a user action such as a click.
