# HTML and CSS Reference

## HTML document structure: doctype, html, head, body
A page starts with `<!DOCTYPE html>`, then `<html lang="en">` containing `<head>` (title, meta tags, stylesheets; nothing shown on the page) and `<body>` (everything visible). The `lang` attribute tells browsers and screen readers the page language.

## HTML meta charset and title
`<meta charset="utf-8">` as the first thing in `<head>` makes the browser read the file as UTF-8, so accented letters and symbols show correctly. `<title>Page name</title>` sets the text of the browser tab and the search-result headline.

## HTML meta viewport for mobile
`<meta name="viewport" content="width=device-width, initial-scale=1">` in `<head>` makes phones lay the page out at the real screen width. Without it, phones pretend the screen is about 980 px wide and shrink the page.

## HTML meta description
`<meta name="description" content="One sentence about the page">` gives search engines a summary they may show under the page title. Keep it to one or two plain sentences that match the page.

## HTML headings h1 to h6
`<h1>` is the main heading, down to `<h6>`. Use one `<h1>` per page and do not skip levels (`h2` under `h1`, then `h3` under `h2`). Headings give the page its outline for screen readers and search engines; choose size with CSS, not by heading level.

## HTML paragraphs, line breaks and horizontal rules
`<p>` is a paragraph. `<br>` forces a line break inside text. `<hr>` draws a thematic break between sections. HTML collapses several spaces and newlines in the source into a single space.

## HTML text elements: strong, em, code, pre, blockquote
`<strong>` marks important text (usually bold), `<em>` marks stress (usually italic), `<code>` marks code, `<pre>` keeps spaces and line breaks exactly as written, `<blockquote>` marks a quotation, `<mark>` highlights, `<small>` marks fine print.

## HTML links: the a element
`<a href="https://example.com">text</a>` makes a link. `href="page.html"` is a relative link, `href="#section"` jumps to the element with `id="section"`, `href="mailto:a@b.com"` opens an email, `href="tel:+123"` starts a call. `target="_blank"` opens a new tab; add `rel="noopener noreferrer"` with it.

## HTML links vs buttons
Use `<a href>` to go somewhere (a new page or a place in the page). Use `<button>` to do something on the current page (open a menu, submit, toggle). Both work with the keyboard when you use the right element, and `<div onclick>` does not.

## HTML images: img, src and alt
`<img src="cat.jpg" alt="A grey cat on a sofa" width="600" height="400">`. `alt` is the text for screen readers and for when the image fails to load. For a purely decorative image use `alt=""`. Set `width` and `height` so the page does not jump while loading; `loading="lazy"` delays offscreen images.

## HTML responsive images: srcset and picture
`<img src="small.jpg" srcset="small.jpg 480w, large.jpg 1200w" sizes="100vw" alt="...">` lets the browser pick a file size. `<picture>` with `<source media="(min-width: 800px)" srcset="wide.jpg">` and a final `<img>` chooses between different crops or formats.

## HTML lists: ul, ol, li, dl
`<ul>` is a bulleted list and `<ol>` a numbered list; each item is an `<li>`. `<ol start="5">` starts at 5 and `<ol reversed>` counts down. `<dl>` holds term and description pairs: `<dt>` for the term and `<dd>` for the description. Lists can be nested by putting a list inside an `<li>`.

## HTML tables
`<table>` contains `<thead>`, `<tbody>` and rows `<tr>`, which contain header cells `<th>` and data cells `<td>`. `colspan="2"` and `rowspan="2"` merge cells. `<th scope="col">` helps screen readers. Use tables for data, not for page layout.

## HTML forms: form, action and method
`<form action="/send" method="post">` groups fields and sends them when submitted. `method="get"` puts values in the URL, `method="post"` sends them in the request body. A field is sent only if it has a `name` attribute.

## HTML input types
`<input type="...">` can be `text`, `email`, `password`, `number`, `tel`, `url`, `search`, `date`, `time`, `checkbox`, `radio`, `file`, `range`, `color` or `hidden`. The type changes the keyboard on phones and the built-in checks. Radio buttons with the same `name` form one choice group.

## HTML label element
`<label for="email">Email</label><input id="email" type="email">` links the text to the field, so clicking the text focuses the field and screen readers read it. You can also wrap the input inside the label. Every field needs a label; a `placeholder` is not a label.

## HTML button element
`<button type="submit">Send</button>`. Inside a form, a button without a type submits the form, so write `type="button"` for buttons that should not submit. `type="reset"` clears the form. Add `disabled` to switch a button off.

## HTML select, option and textarea
`<select name="c"><option value="a">Apple</option><option value="b">Banana</option></select>` makes a dropdown; `selected` marks the default. `<textarea name="msg" rows="4"></textarea>` is a multi-line text box and its starting text goes between the tags.

## HTML form validation attributes
`required` blocks empty submission, `minlength` and `maxlength` limit text length, `min` and `max` limit numbers and dates, `pattern="[0-9]{5}"` requires a regex match, `placeholder` shows a hint, `disabled` and `readonly` stop editing. Browsers check these before submitting, but a server must check again.

## HTML semantic elements: header, nav, main, section, article, aside, footer
These describe the role of a part of the page: `<header>` the top, `<nav>` the main links, `<main>` the unique content (one per page), `<section>` a themed group, `<article>` a self-contained piece, `<aside>` side content, `<footer>` the bottom. They help screen readers and search engines.

## HTML div vs span
`<div>` is a generic block box and `<span>` is a generic inline box; neither has meaning. Use them only when no semantic element fits, usually to attach a class or to group things for layout or styling.

## HTML id vs class and data attributes
An `id` must be unique on the page and is used for links and scripts: `id="top"`. A `class` can be shared by many elements: `class="card big"`. `data-*` attributes store custom values: `data-price="9"` is read in JavaScript as `el.dataset.price`.

## HTML link element and style element for CSS
`<link rel="stylesheet" href="style.css">` in `<head>` loads an external stylesheet. `<style>` holds CSS written in the page. `<link rel="icon" href="favicon.ico">` sets the tab icon. A `style="..."` attribute applies CSS to one element only.

## HTML script element: defer and async
`<script src="app.js" defer></script>` downloads while the page is read and runs after the page is parsed, in order. `async` runs as soon as it is downloaded, in any order, so it suits independent scripts. A plain script in `<head>` blocks the page while it loads.

## HTML iframe
`<iframe src="https://example.com" title="Description"></iframe>` shows another page inside the page. Always give it a `title`. Add `sandbox` to restrict what the embedded page may do. Many sites refuse to be embedded.

## HTML video and audio
`<video src="a.mp4" controls width="640"></video>` and `<audio src="a.mp3" controls></audio>` show players. For several formats nest `<source src="a.webm" type="video/webm">` elements. Browsers block autoplay with sound, so autoplay needs `muted`. `poster="p.jpg"` sets a preview image.

## HTML canvas
`<canvas id="c" width="400" height="200"></canvas>` is a bitmap area drawn with JavaScript: `const ctx = c.getContext("2d"); ctx.fillRect(10, 10, 50, 50);`. The default size is 300 by 150. Set size with the `width` and `height` attributes, not only with CSS, or the drawing will be stretched.

## HTML inline SVG
`<svg viewBox="0 0 100 100" width="100"><circle cx="50" cy="50" r="40" fill="teal"/></svg>` draws vector shapes that stay sharp at any size. `viewBox` sets the internal coordinate system, which scales to the `width` and `height`. Inline SVG can be styled with CSS and scripted.

## HTML details and summary
`<details><summary>Question</summary>Answer text</details>` makes a section that opens and closes on click, with no JavaScript. Add the `open` attribute to start it open.

## HTML entities: &amp; &lt; &gt; &nbsp;
Characters with a special meaning in HTML are written as entities: `&lt;` for `<`, `&gt;` for `>`, `&amp;` for `&`, `&quot;` for a double quote, `&nbsp;` for a space that does not wrap. HTML comments are written `<!-- note -->`.

## CSS syntax: selector, property, value
A rule is `selector { property: value; }`, for example `p { color: navy; font-size: 18px; }`. Each declaration ends with a semicolon. Comments are written `/* note */`. Several selectors can share a rule: `h1, h2 { margin: 0; }`.

## CSS selectors: element, class, id, descendant, child
`p` matches elements, `.card` matches `class="card"`, `#top` matches `id="top"`, `nav a` matches links anywhere inside `nav`, `nav > a` only direct children, `h2 + p` the paragraph right after an `h2`, `h2 ~ p` all later sibling paragraphs, `input[type="text"]` matches by attribute, `*` matches everything.

## CSS specificity and cascade
When rules conflict, the more specific selector wins: an inline `style` beats an `#id`, which beats a `.class` (also attributes and pseudo-classes), which beats an element selector. With equal specificity the later rule wins. `!important` overrides the rest and makes later changes hard, so avoid it.

## CSS inheritance
Some properties pass from parent to children automatically, mainly text ones such as `color`, `font-family`, `font-size`, `line-height` and `text-align`. Boxes properties such as `margin`, `padding` and `border` are not inherited. Force it with `inherit`.

## CSS box model: content, padding, border, margin
Every element is a box: content, then `padding` (space inside), `border`, then `margin` (space outside). With the default `box-sizing: content-box`, `width` covers only the content, so padding and border add to the total. Set `box-sizing: border-box` and `width` includes them.

## CSS margin vs padding
Padding is space inside the border and shows the element's background. Margin is space outside the border, is transparent, and separates the element from its neighbours. Vertical margins between blocks collapse into the larger of the two. `margin: 0 auto` centers a block that has a set width.

## CSS margin and padding shorthand
`margin: 10px` sets all four sides. `margin: 10px 20px` is top and bottom 10, left and right 20. `margin: 1px 2px 3px 4px` goes clockwise from the top: top, right, bottom, left. `padding` works the same way.

## CSS display: block, inline, inline-block, none
`block` elements start on a new line and fill the width (`div`, `p`). `inline` elements flow within text and ignore `width` and `height` (`span`, `a`). `inline-block` flows inline but accepts width and height. `none` removes the element from the layout.

## CSS display none vs visibility hidden vs opacity 0
`display: none` removes the element and its space. `visibility: hidden` hides it but keeps its space. `opacity: 0` makes it invisible but it still takes space and can be clicked. For toggling, `display: none` also hides it from screen readers.

## CSS flexbox basics
`display: flex` on a container lays its children in a row. `flex-direction: column` stacks them. `justify-content` places them along the main axis (`flex-start`, `center`, `space-between`, `space-around`, `flex-end`). `align-items` places them across it (`stretch`, `center`, `flex-start`, `flex-end`).

## CSS flexbox: wrap, gap, grow, shrink, basis
`flex-wrap: wrap` lets items move to a new line. `gap: 16px` sets space between items. On an item, `flex: 1` makes it share the free space equally, `flex-grow` sets how much it grows, `flex-shrink: 0` stops it shrinking, `flex-basis: 200px` sets its starting size. `align-self` and `order` change one item.

## CSS centering with flexbox and grid
Center a child both ways: `.parent { display: flex; justify-content: center; align-items: center; }` or `.parent { display: grid; place-items: center; }`. Give the parent a height, such as `min-height: 100vh`, to center in the page. For horizontal only: `margin: 0 auto` on a block with a width.

## CSS grid basics
`display: grid; grid-template-columns: 1fr 2fr; gap: 16px;` makes two columns where the second is twice as wide. `fr` is a share of the free space. `repeat(3, 1fr)` makes three equal columns. `grid-template-rows` and `grid-auto-rows` set row heights.

## CSS responsive grid with auto-fit and minmax
`grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));` makes as many columns as fit, each at least 200 px wide, and shares the leftover space. The columns reduce on a narrow screen without any media query.

## CSS grid placement: span, areas, justify and align
`grid-column: 1 / 3` makes an item run from line 1 to line 3, `grid-column: span 2` makes it two columns wide. `grid-template-areas: "head head" "side main"` names areas that items claim with `grid-area: head`. `justify-items` and `align-items` place content in cells.

## CSS position: static, relative, absolute, fixed, sticky
`static` is the default flow. `relative` shifts the element from its normal spot and becomes the anchor for absolute children. `absolute` is placed against the nearest positioned ancestor with `top`, `right`, `bottom`, `left`. `fixed` stays in place on the screen while scrolling. `sticky` scrolls normally until it reaches its `top` value, then sticks.

## CSS z-index and stacking
`z-index` sets which overlapping element is in front; a higher number is in front. It works on positioned elements (not `static`) and on flex or grid items. A parent that creates a stacking context, for example with `position` plus `z-index`, traps its children, so they cannot go above an outside element with a higher index.

## CSS units: px, em, rem, %, vw, vh
`px` is a fixed CSS pixel. `rem` is relative to the root font size (browsers default to 16 px, so `1.5rem` is 24 px). `em` is relative to the element's own font size, or its parent's when used for `font-size`. `%` is relative to the parent. `vw` and `vh` are 1% of the viewport width and height.

## CSS calc, min, max and clamp
`calc(100% - 40px)` mixes units. `min(90%, 600px)` takes the smaller value and `max(...)` the larger. `clamp(1rem, 2.5vw, 2rem)` gives a size that scales with the screen but never goes below 1rem or above 2rem; it is common for fluid font sizes.

## CSS media queries
`@media (min-width: 768px) { .nav { display: flex; } }` applies rules only when the condition is true. Mobile-first means writing the small-screen styles first and adding `min-width` queries for larger ones. Other tests: `(orientation: landscape)`, `(hover: hover)`, `print`.

## CSS custom properties (variables)
Define with `--name` and read with `var()`: `:root { --brand: #0a7a5c; } a { color: var(--brand); }`. A fallback goes second: `var(--gap, 8px)`. Custom properties inherit, so redefining one on an element changes it for its children, and JavaScript can change them with `el.style.setProperty("--brand", "red")`.

## CSS colors: hex, rgb, hsl, rgba
`#ff0000` and `#f00` are hex red; `#ff000080` adds half opacity. `rgb(255, 0, 0)` gives red, green, blue from 0 to 255. `rgba(0, 0, 0, 0.5)` adds alpha. `hsl(210, 50%, 40%)` is hue (0 to 360 on the colour wheel), saturation, lightness, which makes shades easy to adjust. `currentColor` reuses the text colour.

## CSS fonts: font-family, size, weight, line-height
`font-family: "Georgia", serif;` lists fonts in order, ending with a generic family (`serif`, `sans-serif`, `monospace`, `system-ui`). `font-size: 1rem`, `font-weight: 700` (bold), `line-height: 1.5` (a unitless number is a multiple of the font size). `font-style: italic`.

## CSS @font-face custom fonts
`@font-face { font-family: "MyFont"; src: url("my.woff2") format("woff2"); font-display: swap; }` loads a font file, then `font-family: "MyFont", sans-serif;` uses it. `font-display: swap` shows fallback text first so the page is readable while the font loads.

## CSS text properties
`text-align: center`, `text-decoration: underline` (or `none` to remove link underlines), `text-transform: uppercase`, `letter-spacing: 0.05em`, `white-space: nowrap`. To cut long text with an ellipsis use `overflow: hidden; white-space: nowrap; text-overflow: ellipsis;`.

## CSS backgrounds and gradients
`background-color: #eee;` and `background-image: url("a.jpg");` with `background-size: cover` (fill the box, may crop) or `contain` (fit whole), `background-position: center`, `background-repeat: no-repeat`. A gradient is an image: `background: linear-gradient(to right, #fff, #000);`.

## CSS border, border-radius, box-shadow
`border: 1px solid #ccc;` sets width, style and colour. `border-radius: 8px` rounds corners and `50%` on a square makes a circle. `box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2)` is x offset, y offset, blur, colour. `outline` draws outside the border and does not change layout.

## CSS overflow
`overflow: visible` (default) lets content spill out of the box. `hidden` clips it, `scroll` always shows scrollbars, `auto` shows them only when needed. `overflow-x` and `overflow-y` set each direction. A horizontal scrollbar on a whole page usually means some element is wider than the screen.

## CSS object-fit and aspect-ratio
`object-fit` controls how an image or video fills its box: `cover` fills and crops, `contain` fits whole with empty space, `fill` stretches. Use `img { width: 100%; height: 200px; object-fit: cover; }` for equal thumbnails. `aspect-ratio: 16 / 9` keeps a box in proportion.

## CSS responsive images with max-width
`img { max-width: 100%; height: auto; }` stops images from overflowing their container and keeps their proportions. It is a common part of a basic reset.

## CSS box-sizing reset
A common starting rule is `*, *::before, *::after { box-sizing: border-box; }`, so width and height always include padding and border. Many stylesheets also add `body { margin: 0; }`.

## CSS pseudo-classes: :hover, :focus, :active, :nth-child
A pseudo-class selects an element in a state. `a:hover` (pointer over), `button:focus-visible` (keyboard focus), `a:active` (being pressed), `input:checked`, `li:first-child`, `li:last-child`, `li:nth-child(2n)` (even items), `p:not(.note)`, `input:disabled`, `input:invalid`.

## CSS focus styles and accessibility
Do not use `outline: none` without a replacement; keyboard users need to see where they are. Style `:focus-visible` instead, for example `button:focus-visible { outline: 3px solid #1a73e8; outline-offset: 2px; }`, which shows for keyboard focus and mostly not for mouse clicks.

## CSS pseudo-elements: ::before, ::after, ::placeholder, ::selection
`::before` and `::after` insert generated content inside an element and need the `content` property: `.tag::before { content: "#"; }`. `::placeholder` styles an input's hint text, `::first-line` the first line of text, `::selection` the highlighted text. Pseudo-elements use two colons.

## CSS transitions
`transition: background-color 0.3s ease;` animates a property smoothly when its value changes, such as on `:hover`. Syntax: property, duration, timing function, optional delay. `transition: all` works but animates everything. Prefer `transform` and `opacity`, which are smooth to animate.

## CSS transform
`transform: translateX(20px) rotate(45deg) scale(1.2);` moves, rotates and scales an element without changing the layout around it. `transform-origin` sets the pivot point. Transforms apply right to left in the order written.

## CSS animations and @keyframes
`@keyframes spin { from { transform: rotate(0); } to { transform: rotate(360deg); } }` defines the steps, and `animation: spin 2s linear infinite;` plays them (name, duration, timing, repeat count). Percentages such as `0%`, `50%`, `100%` allow more steps.

## CSS prefers-reduced-motion
`@media (prefers-reduced-motion: reduce) { * { animation: none; transition: none; } }` turns motion off for people who set "reduce motion" in their system. Use it for large movement, parallax and auto-playing effects.

## CSS prefers-color-scheme and dark mode
`@media (prefers-color-scheme: dark) { :root { --bg: #111; --text: #eee; } }` applies styles when the system is set to dark mode. Put colours in custom properties so one block switches the whole page. `color-scheme: light dark` also makes form controls follow the system.

## CSS cursor and pointer-events
`cursor: pointer` shows a hand over clickable things, `not-allowed` for disabled, `text` for text. `pointer-events: none` makes an element ignore mouse and touch so clicks pass through it.

## CSS scroll-behavior and sticky header
`html { scroll-behavior: smooth; }` animates jumps to `#anchor` links. A header that stays at the top: `header { position: sticky; top: 0; }`. Sticky fails if a parent has `overflow: hidden`.

## CSS float and clear
`float: left` moves an element to the left and lets text wrap around it. `clear: both` makes an element start below floats. Floats were once used for page layout; today use flexbox or grid for layout and keep float for wrapping text around an image.

## CSS filter and opacity
`opacity: 0.5` makes an element and its children half transparent, unlike `rgba`, which affects only one colour. `filter: blur(4px)`, `grayscale(1)` and `brightness(0.8)` change how an element looks. `backdrop-filter: blur(8px)` blurs what is behind it.
