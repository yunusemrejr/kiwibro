# Regular Expressions Reference

## Regular expression: a pattern for matching text
A regular expression, or regex, is a small pattern language for finding, checking, extracting and replacing text. `^cat$` matches the word cat and nothing else. The same pattern works in editors, the command line and most programming languages, though the details differ slightly between regex engines.

## Literal characters match themselves
A letter or digit in a pattern matches that same character: the pattern `cat` matches the c, a and t together, anywhere in the text. Most punctuation is literal too, but `. \ * + ? ( ) [ ] { } | ^ $` have special meanings and need escaping to be matched.

## Character class [abc]: one character from a set
`[abc]` matches one character that is a, b or c; `gr[ae]y` matches grey and gray. Inside a class, `-` is literal at the start or end. A class always matches exactly one character, so to repeat it add a quantifier: `[0-9]+`.

## Character class ranges [a-z]
Inside brackets, `-` between two characters means a range: `[a-z]` any lowercase letter, `[A-Za-z0-9]` any letter or digit, `[0-9a-fA-F]` one hex digit. Ranges follow character code order, so a written range like `[a-Z]` is invalid or includes unwanted characters.

## Negated character class [^...]
A caret first inside brackets reverses the set: `[^0-9]` matches any character that is not a digit, and `[^"<]` matches anything except a quote or a less-than sign. The negated class still matches exactly one character, including spaces and newlines unless they are excluded.

## \d digits and \D non-digits
`\d` matches one decimal digit, the same as `[0-9]` in most engines. `\D` is its negation, one non-digit character. In JavaScript without the `u` flag, `\d` is ASCII only; other engines and Unicode-aware modes may also match digits from other scripts.

## \w word characters and \W
`\w` matches one word character: letters, digits and the underscore in almost all engines. `\W` matches one character that is not a word character, such as a space, comma or hyphen. `\w` is ASCII only unless the engine is in a Unicode mode.

## \s whitespace and \S
`\s` matches one whitespace character: a space, tab, newline, carriage return, form feed or vertical tab. `\S` matches one non-whitespace character, which is why `\S+` is a common shorthand for a run of visible text such as one word.

## Dot . matches any character
`.` matches any single character except, in most engines, a newline. `a.c` matches abc and a c. The `s` flag makes the dot match newlines as well. To match a literal dot, as in a file name, escape it: `\.`.

## Anchors ^ and $: start and end of the text
`^` matches the start of the string and `$` matches the end: `^\d+$` means the whole text is digits. Without anchors a pattern matches anywhere inside the text. In multiline mode they match the start and end of each line instead.

## Word boundary \b
`\b` matches a position between a word character and a non-word character, not an actual character. `\bcat\b` matches cat in "a cat sat" but not inside "concatenate". `\B` matches where there is no boundary.

## Quantifiers * + ?: how many times
`*` means zero or more, `+` one or more, `?` zero or one. `colou?r` matches color and colour; `\d+` matches one or more digits; `a*` matches even an empty string, so it never fails and can match nothing at all.

## Quantifier {n}, {n,}, {n,m}: an exact count
`\d{3}` matches exactly three digits, `\d{2,}` two or more, `\d{2,4}` between two and four. Braces let you state a length instead of chaining `?`. They need escaping to be matched literally in some engines.

## Greedy and lazy quantifiers
A quantifier is greedy by default: `".*"` in `say "hi" and "bye"` matches from the first quote to the last. Adding `?` makes it lazy: `".*?"` stops at the earliest closing quote. Lazy and greedy give the same answer when no choice exists, so use lazy to take the shortest match.

## Capturing group ( ): grouping and remembering
Parentheses group part of a pattern, so a quantifier applies to the whole group and the matched text is captured: `(\d{4})-(\d{2})` captures year and month. Captures are numbered from 1 in the order their opening parenthesis appears, and are read after a match.

## Non-capturing group (?: )
`(?:...)` groups without capturing. It keeps the pattern readable and the group count low: `(?:https?|ftp)://` matches one of the schemes without adding a numbered group. Useful when a group exists only to apply `|` or a quantifier.

## Named group (?<name>...)
A named group captures under a name instead of a number: `(?<year>\d{4})-(?<month>\d{2})`, read as `match.groups.year` in JavaScript and `m.group("year")` in Python. Python also accepts the older `(?P<name>...)` form. Names make patterns with several captures far easier to read.

## Alternation |: one of several patterns
`cat|dog` matches cat or dog. Alternation takes the whole expressions on either side, so `^cat|dog$` means "starts with cat" or "ends with dog"; group it as `^(cat|dog)$` to mean either whole word. Most engines try alternatives from left to right and keep the first that matches.

## Backreference \1: matching the same text again
`(\w+)\s+\1` matches a repeated word such as "the the", because `\1` matches whatever the first group captured. Named groups use `\k<name>`. Backreferences work in most engines but not in POSIX extended regular expressions, and they can make matching slower.

## Lookahead (?= ) and (?! )
`(?=...)` checks that a pattern follows without consuming it: `\d+(?= dollars)` matches the number in "12 dollars" but not in "12 euros". `(?!...)` checks that it does not follow. Lookarounds add conditions while keeping the match text short.

## Lookbehind (?<= ) and (?<! )
`(?<=...)` checks that a pattern comes before the current position: `(?<=\$)\d+` matches the amount in "$12" without the dollar sign. `(?<!...)` requires that it does not. Lookbehind is supported in modern engines; older JavaScript versions rejected it.

## Flag i: ignore case
The `i` flag makes letter matching ignore case, so `/ab+c/i` matches ABBC and Abc. In most languages the flag is added after the closing slash or passed as an option, such as `re.IGNORECASE` in Python. It applies to the whole pattern.

## Flag g: global matching
The `g` flag makes a function find all matches instead of stopping at the first. In JavaScript `text.match(/a/g)` returns every match, while without `g` it returns the first with its groups. Other languages use separate functions such as `findall` in Python.

## Flag m: multiline anchors
The `m` flag makes `^` and `$` match the start and end of each line rather than only the start and end of the whole text. It is what you want when scanning a multi-line file line by line, and it does not change what `.` matches.

## Flag s: dot matches newline
By default `.` does not match a newline, so a pattern cannot run across lines. The `s` flag, called dotAll in JavaScript, lets `.` match a newline too. An alternative that works everywhere is `[\s\S]`, a class that matches any character.

## Escaping special characters
A backslash before a special character makes it literal: `\.` a dot, `\$` a dollar sign, `\(` a parenthesis. To match a backslash itself write `\\`. Inside a character class, `]`, `\`, `^` and sometimes `-` need escaping. Escaping a letter changes its meaning instead, as in `\d`.

## Regex dialects vary between tools
Patterns are not fully portable. JavaScript, Python, PCRE (PHP, Perl, grep -P), Java and the POSIX tools of grep and sed agree on the basics but differ in escaping, named groups, lookbehind and Unicode support. Test a pattern in the tool that will run it.

## Writing regex inside code: escaping twice
A regex written in a string passes through the language's string escaping first: `"\d"` can lose its backslash, so write `"\\d"` or use a raw string such as Python's `r"\d+"`. JavaScript regex literals `/^\d+$/` avoid the double escaping for simple patterns.

## Catastrophic backtracking: a slow regex
Some patterns make an engine try enormous numbers of ways to match: `(a+)+$` on a long run of a letters followed by one b can hang. The pattern is valid, the engine explodes. Avoid nesting quantifiers, prefer specific classes, and never run a user-supplied pattern without a timeout.

## Match, search and full match
`search` or `test` looks for the pattern anywhere; `match` often means the same; a full match requires the whole string, as if the pattern were wrapped in `^...$`. Most validation bugs come from forgetting the anchors and accepting a string that merely contains the pattern.

## Regex and Unicode text
By default a set such as `\w` or `[a-z]` covers ASCII only, so names with accents or other scripts fail. JavaScript keeps `\w` ASCII even with its `u` flag, while Python 3 makes it Unicode-aware by default. To be sure, write an explicit range such as `\p{L}`, where the engine supports Unicode properties.

## Pattern: whole number
`^\d+$` matches a run of digits such as `42`. Limits: no sign and no decimal point, so `-5` and `4.2` fail; `\d` is ASCII digits only in JavaScript unless the `u` flag is used; leading zeros are accepted; a value too large for the number type passes.

## Pattern: decimal number
`^-?\d+(\.\d+)?$` matches `42`, `-3.5` and `0.25`. Limits: it allows leading zeros and forms such as `007`, it rejects `.5` and `5.` and exponents such as `1e3`, and it does not stop a value outside the allowed range. Parse the text as a number after matching.

## Pattern: hexadecimal colour
`^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$` matches `#fff` and `#1a73e8`. Limits: it rejects the four- and eight-digit forms with an alpha channel (`#ffff`, `#1a73e8ff`); without anchors it would match a colour inside a longer string; it does not check the colour is visible or sensible.

## Pattern: ISO date YYYY-MM-DD
`^\d{4}-\d{2}-\d{2}$` matches `2026-11-15`. Limits: it accepts impossible dates such as `2026-02-31` and `2026-13-01`, because a regex cannot count days in a month. Match the shape with the pattern, then check the date with a date library.

## Pattern: 24-hour time
`^([01]\d|2[0-3]):[0-5]\d$` matches `00:00` and `23:59`. Limits: it rejects seconds and fractional seconds, so `12:30:45` fails; it accepts `9:05` only if you change the hour part to `[0-1]?\d|2[0-3]`, which also allows `9` alone; `24:00` is rejected, which fits a 00 to 23 clock.

## Pattern: simple email check
`^[^\s@]+@[^\s@]+\.[^\s@]+$` accepts text with one @ and a dot after it. Limits: it is only a rough shape check. It accepts `a@b..c` and `a@-b.c`, rejects valid addresses with unusual quoting, and cannot prove the mailbox exists. The only real check is sending a message with a confirmation link.

## Pattern: URL scheme
`^[a-z][a-z0-9+.-]*://` matches the scheme of a URL such as `https://` or `git+ssh://`. Limits: it checks only the beginning, not the host or path, and it is case-insensitive only if you add the `i` flag, although schemes are case-insensitive by definition. For a web address, `^https?://` is stricter.

## Pattern: IPv4 address
`^((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$` matches four octets from 0 to 255, such as `192.168.0.1`. Limits: it rejects octets with leading zeros such as `01`; it accepts `0.0.0.0`; it says nothing about whether the address exists or answers.

## Pattern: leading and trailing whitespace
`^\s+|\s+$` matches spaces, tabs or newlines at the start or end of the text, and with the global flag you can replace each match with nothing to trim it. Most languages already have a trim function; use that first and the pattern only when trimming needs to be selective.

## Pattern: a repeated word
`\b(\w+)\s+\1\b` finds a word repeated straight after itself, as in "this is is wrong". Limits: the match is case-sensitive unless the `i` flag is added, so "The the" is missed with `i` off; it does not treat punctuation or line breaks between the words; it will also flag intentional repetitions.

## Pattern: file extension
`\.([A-Za-z0-9]+)$` captures the extension of a file name, such as `md` in `notes.md`. Limits: it takes the last dot, so `archive.tar.gz` gives `gz`; a hidden file such as `.bashrc` matches with `bashrc` as the extension; names with trailing spaces are not handled. Add the `i` flag to compare extensions without case.

## Pattern: URL slug
`^[a-z0-9]+(?:-[a-z0-9]+)*$` matches a slug such as `how-to-write-regex`. Limits: it allows no uppercase letters and no underscores, and forbids a leading or trailing hyphen and two hyphens in a row; some sites allow those, so adjust the class if needed. It also allows a single long run of digits.
