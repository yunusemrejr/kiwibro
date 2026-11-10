/**
 * LaTeX maths to MathML, with no dependencies and no DOM. Browsers draw MathML
 * natively, so formulas need no maths library. Anything this file does not know
 * gives null, and the caller shows the original text instead.
 *
 * Output holds only the tags emitted here. User text is escaped on the way out.
 */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };

// Kept local: markdown.js imports this file, so importing back would be circular.
function escapeText(value) {
  return value.replace(/[&<>"]/g, (char) => ESCAPES[char]);
}

// Deep nesting stops here, so a hostile input cannot overflow the call stack.
const MAX_DEPTH = 100;

// One match per token: a command (name, or the one character after a backslash),
// a number, whitespace (skipped), a structural character, or any other character.
const TOKEN = /\\([a-zA-Z]+|[\s\S]?)|(\d+(?:\.\d+)?)|(\s+)|([{}^_&~])|([\s\S])/gu;
const LETTER = /^\p{L}$/u;
const STRUCTURE = { "{": "open", "}": "close", "^": "sup", _: "sub", "&": "amp", "~": "tilde" };

// Plain characters that become <mo>. Other ASCII punctuation is rejected, so
// stray symbols show as source rather than as a guessed operator.
const PLAIN_OPERATORS = "+-=<>/*()[],.!|:;?'%";
const OPERATOR_SIGNS = { "-": "−", "'": "′" };

const SYMBOLS = {
  times: "×", cdot: "⋅", div: "÷", pm: "±", mp: "∓", leq: "≤", le: "≤", geq: "≥", ge: "≥",
  neq: "≠", ne: "≠", approx: "≈", equiv: "≡", sim: "∼", propto: "∝", infty: "∞",
  to: "→", rightarrow: "→", leftarrow: "←", Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔",
  leftrightarrow: "↔", mapsto: "↦", in: "∈", notin: "∉", subset: "⊂", subseteq: "⊆", supset: "⊃",
  cup: "∪", cap: "∩", emptyset: "∅", forall: "∀", exists: "∃", neg: "¬", land: "∧", lor: "∨",
  partial: "∂", nabla: "∇", ldots: "…", dots: "…", cdots: "⋯", vdots: "⋮", angle: "∠", perp: "⊥",
  parallel: "∥", circ: "∘", bullet: "•", star: "⋆", ast: "∗", oplus: "⊕", otimes: "⊗",
  langle: "⟨", rangle: "⟩", lfloor: "⌊", rfloor: "⌋", lceil: "⌈", rceil: "⌉", prime: "′",
  hbar: "ℏ", ell: "ℓ", Re: "ℜ", Im: "ℑ", aleph: "ℵ",
  "%": "%", $: "$", "&": "&", "#": "#", _: "_", "{": "{", "}": "}", "|": "‖",
};

// Epsilon and phi follow LaTeX: \epsilon is the lunate form, \varepsilon the plain one.
const GREEK = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ϵ", varepsilon: "ε", zeta: "ζ",
  eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν",
  xi: "ξ", omicron: "ο", pi: "π", varpi: "ϖ", rho: "ρ", varrho: "ϱ", sigma: "σ", varsigma: "ς",
  tau: "τ", upsilon: "υ", phi: "ϕ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ", Upsilon: "Υ",
  Phi: "Φ", Psi: "Ψ", Omega: "Ω",
};

// Big operators take their limits above and below in display maths; integrals keep them at the side.
const BIG_OPERATORS = { sum: "∑", prod: "∏", int: "∫", iint: "∬", oint: "∮", bigcup: "⋃", bigcap: "⋂" };
const INTEGRALS = new Set(["int", "iint", "oint"]);
const FUNCTIONS = new Set([
  "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
  "log", "ln", "lg", "exp", "lim", "max", "min", "sup", "inf", "det", "gcd", "deg", "dim", "ker",
  "arg", "mod", "bmod",
]);
const LIMIT_FUNCTIONS = new Set(["lim", "max", "min", "sup", "inf"]);
const SPACING = { ",": "0.17em", ":": "0.22em", ";": "0.28em", "!": "-0.17em", " ": "0.25em", quad: "1em", qquad: "2em" };
const ACCENTS = { hat: "ˆ", bar: "¯", vec: "→", tilde: "˜", dot: "˙", ddot: "¨", overline: "‾" };
// Sizing and style commands change nothing in MathML, which sizes delimiters by itself.
const IGNORED = new Set([
  "displaystyle", "textstyle", "limits", "nolimits", "big", "Big", "bigg", "Bigg", "bigl", "bigr",
]);
const DOUBLE_STRUCK = { N: "ℕ", Z: "ℤ", Q: "ℚ", R: "ℝ", C: "ℂ" };
const STYLES = { mathrm: "normal", mathit: "italic", mathbf: "bold", mathcal: "script", mathbb: "double-struck" };

const ENVIRONMENTS = {
  aligned: { columns: "alternate" },
  align: { columns: "alternate" },
  "align*": { columns: "alternate" },
  cases: { open: "{", columns: "left" },
  matrix: { columns: "center" },
  pmatrix: { open: "(", close: ")", columns: "center" },
  bmatrix: { open: "[", close: "]", columns: "center" },
  vmatrix: { open: "|", close: "|", columns: "center" },
  array: { columns: "center" },
};

/** Thrown for input this converter does not handle. texToMathML turns it into null. */
class Unsupported extends Error {}

function isCommand(token, name) {
  return token?.type === "cmd" && token.text === name;
}

function isChar(token, text) {
  return token?.type === "char" && token.text === text;
}

function tokenize(source) {
  const tokens = [];
  for (const match of source.matchAll(TOKEN)) {
    const text = match[0];
    const start = match.index;
    const end = start + text.length;
    if (match[1] !== undefined) {
      tokens.push({ type: "cmd", text: match[1], start, end });
    } else if (match[2] !== undefined) {
      tokens.push({ type: "num", text, start, end });
    } else if (match[4] !== undefined) {
      tokens.push({ type: STRUCTURE[text], text, start, end });
    } else if (match[5] !== undefined) {
      tokens.push({ type: LETTER.test(text) ? "letter" : "char", text, start, end });
    }
  }
  return tokens;
}

/** One element for a run of parts: a single part stays bare, several are wrapped. */
function mrow(items) {
  if (items.length === 1 && items[0]) return items[0];
  return `<mrow>${items.join("")}</mrow>`;
}

function fence(sign, stretchy = true) {
  return sign ? `<mo stretchy="${stretchy}">${escapeText(sign)}</mo>` : "";
}

/** A matrix-style table. Cells are arrays of parts; `columns` is a columnalign value or "alternate". */
function table(rows, columns) {
  const width = Math.max(...rows.map((row) => row.length));
  const align =
    columns === "alternate"
      ? Array.from({ length: width }, (_, k) => (k % 2 ? "left" : "right")).join(" ")
      : columns;
  const body = rows
    .map((row) => `<mtr>${row.map((cell) => `<mtd>${mrow(cell)}</mtd>`).join("")}</mtr>`)
    .join("");
  return `<mtable columnalign="${align}">${body}</mtable>`;
}

/** Tokens that end a run of parts: a closing brace, the cell and row separators, \end and \right. */
function endsSequence(token) {
  return (
    token.type === "close" ||
    token.type === "amp" ||
    isCommand(token, "\\") ||
    isCommand(token, "end") ||
    isCommand(token, "right")
  );
}

class Reader {
  constructor(source, tokens, display) {
    this.source = source;
    this.tokens = tokens;
    this.display = display;
    this.pos = 0;
    this.depth = 0;
    // mathvariant set by \mathbf, \mathbb and the other style commands.
    this.variant = null;
  }

  peek() {
    return this.tokens[this.pos];
  }

  next() {
    const token = this.tokens[this.pos];
    if (!token) throw new Unsupported("unexpected end");
    this.pos += 1;
    return token;
  }

  expect(type) {
    if (this.next().type !== type) throw new Unsupported(`expected ${type}`);
  }

  /** The whole input: rows of cells. Outside an environment, only rows (\\) are allowed in display. */
  top() {
    const rows = this.rows(false);
    if (rows.length === 1) return mrow(rows[0][0]);
    return table(rows, "center");
  }

  rows(inEnvironment) {
    const rows = [];
    let row = [];
    for (;;) {
      row.push(this.sequence());
      const token = this.peek();
      if (!token) {
        if (inEnvironment) throw new Unsupported("unclosed environment");
        break;
      }
      if (inEnvironment && isCommand(token, "end")) break;
      this.pos += 1;
      if (token.type === "amp" && inEnvironment) {
        continue;
      } else if (isCommand(token, "\\") && (inEnvironment || this.display)) {
        this.skipSpacing();
        rows.push(row);
        row = [];
      } else {
        throw new Unsupported("unexpected token");
      }
    }
    // A \\ just before \end or the end of input does not start an empty row.
    const trailingEmpty = row.length === 1 && row[0].length === 0;
    if (!trailingEmpty || !rows.length) rows.push(row);
    return rows;
  }

  /** The optional [6pt] after \\, which is spacing only. */
  skipSpacing() {
    if (!isChar(this.peek(), "[")) return;
    while (!isChar(this.peek(), "]")) this.next();
    this.next();
  }

  sequence() {
    this.enter();
    const items = [];
    while (this.peek() && !endsSequence(this.peek())) items.push(this.scripted());
    this.depth -= 1;
    return items;
  }

  /** Every nesting passes through sequence() or atom(), so deep input fails cleanly instead of overflowing the stack. */
  enter() {
    this.depth += 1;
    if (this.depth > MAX_DEPTH) throw new Unsupported("nested too deeply");
  }

  /** A base with any ^ and _ scripts, in the layout its kind asks for. */
  scripted() {
    const base = this.atom();
    let limits = Boolean(base.limits);
    let sub;
    let sup;
    for (let token = this.peek(); token; token = this.peek()) {
      if (isCommand(token, "limits") || isCommand(token, "nolimits")) {
        this.pos += 1;
        limits = token.text === "limits";
      } else if (token.type === "sub" || token.type === "sup") {
        this.pos += 1;
        const taken = token.type === "sub" ? sub : sup;
        if (taken !== undefined) throw new Unsupported("double script");
        // Operators in a script sit close to their neighbours (i=1, x→∞), as in print.
        const script = this.argument().replaceAll("<mo>", '<mo lspace="0.1em" rspace="0.1em">');
        if (token.type === "sub") sub = script;
        else sup = script;
      } else {
        break;
      }
    }
    if (sub === undefined && sup === undefined) return base.html;

    const core = base.html || "<mrow></mrow>";
    // Limits sit above and below only for big operators in display maths.
    const above = limits && this.display;
    if (above && sub !== undefined && sup !== undefined) return `<munderover>${core}${sub}${sup}</munderover>`;
    if (above && sub !== undefined) return `<munder>${core}${sub}</munder>`;
    if (above) return `<mover>${core}${sup}</mover>`;
    if (sub !== undefined && sup !== undefined) return `<msubsup>${core}${sub}${sup}</msubsup>`;
    if (sub !== undefined) return `<msub>${core}${sub}</msub>`;
    return `<msup>${core}${sup}</msup>`;
  }

  /** A braced group, or one token: the argument form of x^2 and \frac12. */
  argument() {
    const token = this.peek();
    if (!token || token.type === "sub" || token.type === "sup") throw new Unsupported("missing argument");
    if (token.type === "open") return this.group();
    if (token.type === "num" && token.text.length > 1) {
      // Without braces only one digit is the argument: x^10 is x^1 followed by 0.
      this.tokens.splice(this.pos, 1, { ...token, text: token.text[0] }, { ...token, text: token.text.slice(1) });
    }
    return mrow([this.atom().html]);
  }

  arguments(count) {
    return Array.from({ length: count }, () => this.argument());
  }

  /** The optional [n] of \sqrt, or null when absent. */
  optionalArgument() {
    if (!isChar(this.peek(), "[")) return null;
    this.pos += 1;
    const items = [];
    while (!isChar(this.peek(), "]")) items.push(this.scripted());
    this.pos += 1;
    return mrow(items);
  }

  group() {
    this.expect("open");
    const items = this.sequence();
    this.expect("close");
    return mrow(items);
  }

  /** Source text of a braced group, for \text, \operatorname and environment names. */
  rawGroup() {
    const open = this.next();
    if (open.type !== "open") throw new Unsupported("expected {");
    let depth = 1;
    for (;;) {
      const token = this.next();
      if (token.type === "open") {
        depth += 1;
      } else if (token.type === "close") {
        depth -= 1;
        if (depth === 0) return this.source.slice(open.end, token.start);
      }
    }
  }

  atom() {
    this.enter();
    const result = this.atomToken();
    this.depth -= 1;
    return result;
  }

  atomToken() {
    const token = this.peek();
    if (!token) throw new Unsupported("unexpected end");
    // A script with no base, such as ^2 at the start, gets an empty base.
    if (token.type === "sub" || token.type === "sup") return { html: "" };
    if (token.type === "open") return { html: this.group() };
    this.pos += 1;
    switch (token.type) {
      case "num":
        return { html: `<mn${this.attr()}>${escapeText(token.text)}</mn>` };
      case "letter":
        return { html: this.letter(token.text) };
      case "char":
        return { html: this.operator(token.text) };
      case "tilde":
        return { html: "<mtext> </mtext>" };
      case "cmd":
        return this.command(token.text);
      default:
        // A stray } or & outside its construct.
        throw new Unsupported("unexpected token");
    }
  }

  operator(char) {
    if (!PLAIN_OPERATORS.includes(char) && char.charCodeAt(0) < 128) throw new Unsupported(char);
    return `<mo>${escapeText(OPERATOR_SIGNS[char] ?? char)}</mo>`;
  }

  attr(fallback = null) {
    const variant = this.variant ?? fallback;
    return variant ? ` mathvariant="${variant}"` : "";
  }

  mi(text, fallback = null) {
    return `<mi${this.attr(fallback)}>${escapeText(text)}</mi>`;
  }

  letter(text) {
    // Blackboard capitals N Z Q R C have their own Unicode letters.
    if (this.variant === "double-struck" && Object.hasOwn(DOUBLE_STRUCK, text)) return `<mi>${DOUBLE_STRUCK[text]}</mi>`;
    return this.mi(text);
  }

  /** Applies a style to the argument's letters and digits only. */
  styled(variant) {
    const saved = this.variant;
    this.variant = variant;
    const html = this.argument();
    this.variant = saved;
    return html;
  }

  command(name) {
    if (Object.hasOwn(SYMBOLS, name)) return { html: `<mo>${escapeText(SYMBOLS[name])}</mo>` };
    if (Object.hasOwn(GREEK, name)) return { html: this.mi(GREEK[name], /^[A-Z]/.test(name) ? "normal" : null) };
    if (Object.hasOwn(BIG_OPERATORS, name)) return { html: `<mo>${BIG_OPERATORS[name]}</mo>`, limits: !INTEGRALS.has(name) };
    if (FUNCTIONS.has(name)) return { html: this.mi(name), limits: LIMIT_FUNCTIONS.has(name) };
    if (Object.hasOwn(SPACING, name)) return { html: `<mspace width="${SPACING[name]}"></mspace>` };
    if (Object.hasOwn(ACCENTS, name)) return { html: `<mover accent="true">${this.argument()}<mo>${ACCENTS[name]}</mo></mover>` };
    if (IGNORED.has(name)) return { html: "" };
    if (Object.hasOwn(STYLES, name)) return { html: this.styled(STYLES[name]) };

    switch (name) {
      case "frac":
      case "dfrac":
      case "tfrac": {
        const [numerator, denominator] = this.arguments(2);
        return { html: `<mfrac>${numerator}${denominator}</mfrac>` };
      }
      case "binom": {
        const [top, bottom] = this.arguments(2);
        return { html: `<mrow><mo>(</mo><mfrac linethickness="0">${top}${bottom}</mfrac><mo>)</mo></mrow>` };
      }
      case "sqrt": {
        const index = this.optionalArgument();
        const body = this.argument();
        return { html: index === null ? `<msqrt>${body}</msqrt>` : `<mroot>${body}${index}</mroot>` };
      }
      case "underline":
        return { html: `<munder accentunder="true">${this.argument()}<mo>_</mo></munder>` };
      case "boxed":
        return { html: `<menclose notation="box">${this.argument()}</menclose>` };
      case "text":
        return { html: `<mtext>${escapeText(this.rawGroup())}</mtext>` };
      case "operatorname":
        return { html: `<mi mathvariant="normal">${escapeText(this.rawGroup())}</mi>` };
      case "left":
        return { html: this.leftRight() };
      case "begin":
        return { html: this.environment(this.rawGroup()) };
      default:
        throw new Unsupported(name);
    }
  }

  /** \left ... \right with stretchy delimiters. A lone "." means no delimiter. */
  leftRight() {
    const open = this.delimiter();
    const items = this.sequence();
    if (!isCommand(this.next(), "right")) throw new Unsupported("\\left without \\right");
    const close = this.delimiter();
    return `<mrow>${fence(open)}${items.join("")}${fence(close)}</mrow>`;
  }

  delimiter() {
    const token = this.next();
    if (isChar(token, ".")) return "";
    if (token.type === "char" && PLAIN_OPERATORS.includes(token.text)) return token.text;
    if (token.type === "cmd" && Object.hasOwn(SYMBOLS, token.text)) return SYMBOLS[token.text];
    throw new Unsupported("delimiter");
  }

  environment(name) {
    if (!Object.hasOwn(ENVIRONMENTS, name)) throw new Unsupported(name);
    const env = ENVIRONMENTS[name];
    // The column spec of array, such as {cc}, has no meaning here.
    if (name === "array") this.rawGroup();
    const rows = this.rows(true);
    if (!isCommand(this.next(), "end") || this.rawGroup() !== name) throw new Unsupported(name);
    const body = table(rows, env.columns);
    const open = fence(env.open);
    const close = fence(env.close);
    return open || close ? `<mrow>${open}${body}${close}</mrow>` : body;
  }
}

/**
 * LaTeX to MathML. `display` gives a block formula. Returns null for input that
 * is empty, unknown or unbalanced, so the caller can show the source instead.
 */
export function texToMathML(tex, { display = false } = {}) {
  const source = String(tex ?? "");
  if (!source.trim()) return null;
  try {
    const body = new Reader(source, tokenize(source), display).top();
    return display ? `<math display="block">${body}</math>` : `<math>${body}</math>`;
  } catch (error) {
    if (error instanceof Unsupported) return null;
    throw error;
  }
}
