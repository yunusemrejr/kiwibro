# Computing Concepts Reference

## CPU: the processor that runs instructions
The CPU (central processing unit) executes the instructions of programs one step at a time: arithmetic, comparisons, moving data and deciding what to do next. Its speed is set by a clock measured in gigahertz, and modern CPUs have several cores. It is the part that does the actual computing, while memory and disks hold the data.

## Core: one independent processing unit in a CPU
A core is a complete set of circuits that can run instructions on its own, so a 4-core CPU can work on four instruction streams at once. More cores help when a program or several programs can run in parallel, such as video encoding or many browser tabs. A single task that cannot be split gains little from extra cores.

## Thread: a sequence of instructions being run
A thread is one path of execution inside a program, with its own place in the code and its own stack. Threads in the same process share memory, which makes them fast to start but able to corrupt each other's data. A CPU core can switch between threads quickly, and with hyper-threading or SMT one core may present two threads to the system.

## RAM: working memory of a computer
RAM (random access memory) holds the programs and data in use right now, and the CPU reads it far faster than a disk. It is volatile: everything in it is lost when power goes off, which is why work must be saved to storage. When RAM is full, the system slows down and starts using swap.

## Swap: disk space used when RAM is full
Swap is a reserved area of a disk that the system uses as extra memory. Windows calls it the page file. When RAM fills, the least-used pages are moved there and brought back when needed, which is far slower than RAM. Heavy use of swap, called thrashing, makes a computer crawl; adding RAM is the real fix.

## SSD vs hard disk: two kinds of storage
An SSD stores data in flash memory chips with no moving parts, so it starts instantly, is silent and handles random reads far faster. A hard disk (HDD) stores data magnetically on spinning platters with a moving head; it is slower but cheaper per gigabyte for large archives. Systems and programs belong on an SSD, backups and media archives can sit on a hard disk.

## GPU: the chip for parallel graphics work
A GPU (graphics processing unit) has thousands of small cores built to work on many pieces of data in the same way at once, such as the pixels of a frame. It draws what is on screen and also speeds up work with the same shape, including machine learning. It is poor at long chains of different steps, which is the CPU's job.

## Bit and byte: the units of digital data
A bit is one binary digit, 0 or 1, and a byte is 8 bits, which can hold 256 different values. Text characters, numbers and instructions are all stored as bytes. File sizes are counted in bytes, and network speeds are usually counted in bits per second, so 100 Mbit/s is about 12.5 MB/s.

## Kilobyte vs kibibyte: 1000 or 1024 bytes
A kilobyte (kB) is 1000 bytes and a kibibyte (KiB) is 1024 bytes, because the binary prefix ki- means 2 to the power 10. Megabyte and mebibyte, gigabyte and gibibyte follow the same pattern. Drive makers use the 1000-based units, while many operating systems show 1024-based values under the shorter name, so the same disk can appear smaller than advertised.

## File system: how a disk organises files
A file system decides how files are named, stored in folders, given permissions and found again on a disk. Common ones are NTFS on Windows, APFS on macOS, ext4 on Linux, and FAT32 or exFAT on memory cards and drives moved between systems. Two systems without a common file system cannot read each other's disk without extra software.

## Path: absolute and relative locations of a file
A path names a file or folder. An absolute path starts from the root, such as `/home/ana/notes.txt` on Linux and macOS or `C:\Users\ana\notes.txt` on Windows. A relative path starts from the current folder, such as `notes.txt` or `../images/pic.png`, where `..` means the parent folder and `.` the current one.

## File extension: the letters after the last dot
A file extension is the part of a name after the last dot, such as `.txt` in `notes.txt`, and it hints at the file's format. Windows uses it to choose which program opens the file, and it hides extensions by default. The extension is only a convention: renaming a file does not convert it, and a program should check the content.

## Process: a running program
A process is a program that is running, with its own memory space, open files and a process ID (PID). Starting a program creates a process, and closing it ends one. Processes are kept apart by the operating system, so one crash usually does not take down the others. The task manager or system monitor lists them.

## Operating system: the software that runs the computer
An operating system (OS) manages the hardware and gives programs a standard way to use it: memory, files, screens, keyboards, networks and other programs. It decides which process runs when and keeps users and their files apart. Windows, macOS, Linux, Android and iOS are operating systems; most of the work is done by its kernel.

## Kernel: the core of an operating system
The kernel is the central part of an operating system and runs with full access to the hardware. It schedules processes, hands out memory, talks to devices through drivers and enforces permissions. Ordinary programs run in user space and must ask the kernel for anything privileged through system calls, such as opening a file or starting a process.

## Driver: software that lets the system talk to hardware
A driver is a piece of software that knows how to operate one device, such as a graphics card, printer, Wi-Fi adapter or touchpad. It translates general requests from the operating system into the signals the device understands. A missing or wrong driver leaves the device unusable or unstable; most arrive with the system and update through it.

## Firmware: software built into a device
Firmware is software stored in a device's own memory that runs on the device itself, rather than on the computer's processor. Keyboards, SSDs, routers, printers and cameras all contain it, and the BIOS or UEFI is the firmware of the motherboard. It is updated rarely; a failed or wrong update can leave the device unusable.

## BIOS and UEFI: firmware that starts the computer
BIOS and UEFI are the firmware built into the motherboard. When power comes on they test the hardware, find a boot device and start its bootloader. UEFI is the modern replacement: it has a graphical setup, supports very large disks with the GPT partition scheme, and can verify the bootloader with Secure Boot. BIOS is the older form and is mostly gone from new machines.

## Boot: starting the computer
Booting is the whole process from pressing power to a usable system: the firmware runs a self-test, finds the boot device, loads a bootloader, which starts the kernel, which starts background services and finally the login screen. A slow boot usually means slow storage, many startup programs or a service waiting for a timeout. Restarting is often called rebooting.

## Terminal: a window for typing commands
A terminal is a text interface where you type commands and read their output. On a desktop it is normally a terminal emulator, a program that imitates the old hardware terminal inside a window. The terminal itself only handles text and keys; the shell running inside it is what understands commands. It gives direct, scriptable access to the system.

## Shell: the program that reads and runs commands
A shell is a program that reads commands as text and runs them: bash, zsh and fish on Linux and macOS, PowerShell and cmd on Windows. It also offers variables, pipes that send one command's output into another, redirection to files and scripts. It is what you talk to inside a terminal.

## API: a defined way for programs to use each other
An API (application programming interface) is a documented set of functions, requests and data formats that one piece of software offers to another. A library API is called inside a program; a web API is called over the network, commonly with HTTP and JSON. It hides the internals, so the provider can change them without breaking callers as long as the API stays the same.

## Client and server: who asks and who answers
A client is a program that asks for something, and a server is a program that answers. Your browser is a client asking a web server for pages; an email app is a client asking a mail server for messages. The server usually runs all the time on a machine others can reach, and one server can serve many clients at once.

## Database: organised, searchable stored data
A database is a program or system for storing data so it can be searched, changed and kept consistent. Relational databases store rows in tables and are queried with SQL; others store documents, key-value pairs or graphs. Real ones handle many users at once, transactions and backups, which a plain file does not.

## Cache: a fast copy of data kept close by
A cache is a smaller, faster store holding copies of data that is likely to be needed again: the CPU cache, the browser cache of web files, a DNS cache of addresses, or a server cache of query results. A cache hit is fast, a miss must fetch the original. Keeping the copy in step with the original is the hard part.

## Compiler vs interpreter: two ways to run code
A compiler translates a whole program into machine code before it runs, reporting errors first and producing a fast executable. An interpreter reads and executes the source step by step, which starts quickly and is easier to debug but usually slower. Many languages mix the two: they compile to bytecode and a virtual machine or just-in-time compiler runs it.

## Source code: the text a program is written in
Source code is the instructions of a program written by people in a programming language such as Python, C or JavaScript, stored in text files. It is either compiled into machine code or run by an interpreter. Reading and changing source code is how software is maintained, and keeping it in version control records who changed what.

## Open source: software whose source anyone may use and change
Open source software is released with its source code under a licence that lets anyone use, study, modify and redistribute it, usually on conditions about credit and about keeping the licence. It does not have to be free of charge: open refers to the freedoms, not the price. Linux, Firefox and Python are open source.

## MIT licence: a short permissive open source licence
The MIT licence lets anyone use, change, publish and sell the software, provided the copyright notice and licence text stay with it, and it comes with no warranty. It is short and permissive, so modified versions can be kept closed. Many small libraries use it, often worded almost the same as the BSD licences.

## GPL: a copyleft open source licence
The GNU General Public License lets anyone use, study, change and share the software, but a copyleft rule applies: if you distribute a modified version, it must also be under the GPL and its source code must be made available. Software that merely runs next to GPL code is usually unaffected; code combined into one program is. Linux uses the GPL.

## Apache 2.0 licence: permissive, with a patent grant
The Apache License 2.0 lets anyone use, change and redistribute the software, including in closed products, as long as notices are kept and modified files are marked. It also grants users a licence to the contributor's patents, which the MIT licence does not address. It is common for larger projects and companies.

## Version control: recording the history of files
Version control records every change to a set of files, who made it and when, so any earlier version can be restored. Git is the most used system: you commit changes with a message, work on branches, and merge them back. It also lets several people work on the same project without overwriting each other.

## Bug: a defect in software
A bug is a mistake in a program that makes it do the wrong thing: crash, give a wrong result, lose data or behave differently from what is documented. Most are in the code's logic or in its handling of unusual input. Fixing one means reproducing it, finding the cause, changing the code and adding a test that would have caught it.

## Stack trace: the list of calls at the moment of an error
A stack trace is printed when a program fails and shows the chain of function calls that led to the error, with file names and line numbers. Where the error happened is in the trace, but whether it is at the top or the bottom depends on the language: Python prints the most recent call last, Java prints the failing frame first. Read the frames from your own code.

## Unit test: an automatic check of one small piece
A unit test runs one function or small part of a program with chosen inputs and checks the result against what is expected. Tests are automated and fast, so they can run after every change and catch a break before users do. A passing test suite shows that the tested cases still work, not that the whole program is correct.

## Encoding: the mapping from characters to bytes
A character encoding says how the characters of text are stored as bytes. The same bytes read with the wrong encoding produce mojibake, such as `cafÃ©` instead of `café`. The encoding must be known at both ends; UTF-8 is the safe default today, and most systems assume it when nothing is declared.

## ASCII: the original 7-bit character set
ASCII defines 128 characters with numbers from 0 to 127: the English letters in upper and lower case, the digits, punctuation, a space and control codes such as newline and tab. It needs 7 bits per character and covers no accents or other scripts. UTF-8 keeps ASCII unchanged, so old ASCII text is still valid.

## Unicode: one number for every character
Unicode is a standard that gives every character of the world's writing systems a number, called a code point and written like `U+0041` for a capital A. It covers letters, digits, punctuation, symbols and emoji. Unicode does not by itself say how those numbers are stored as bytes; UTF-8, UTF-16 and UTF-32 are encodings that do.

## UTF-8: the common encoding of Unicode text
UTF-8 stores each Unicode code point in one to four bytes, using one byte for the ASCII range and more for other scripts. That makes it compatible with old ASCII files and efficient for English, and it is the default on the web and on Linux and macOS. A byte that starts a multi-byte character is never a valid ASCII byte, which helps detect corruption.

## Hash: a fixed-size fingerprint of data
A hash function turns any data into a short value of fixed length, such as a SHA-256 hash of 64 hex characters. The same input always gives the same hash, a tiny change gives a very different one, and the original cannot be reconstructed from the hash. Hashes are used to check data, to store passwords with a salt, and to look things up quickly; different inputs can rarely share a hash, a collision.

## Checksum: a value used to detect damaged data
A checksum is a small value computed from a file or message and stored or sent with it, so the receiver can compute it again and compare. If the values differ, the data was changed or damaged in transit or on disk. It catches accidental errors; it does not prove the data is genuine, because anyone can recompute it, which is what cryptographic hashes are for.

## Encryption: symmetric vs asymmetric
Encryption scrambles data so only someone with the right key can read it. Symmetric encryption uses one secret key for both locking and unlocking, and is fast, so it carries the bulk of traffic. Asymmetric encryption uses a public key to lock and a matching private key to unlock, which solves the problem of sharing a secret without meeting.

## Public and private key: a pair for encryption and signatures
A key pair has a public key that can be given to anyone and a private key that stays secret. Data encrypted with the public key can be decrypted only with the private key; data signed with the private key can be checked with the public key. Never share the private key, and keep a backup, because losing it means losing access.

## Password hashing: storing passwords so they cannot be read
A password should never be stored as plain text. A password hash is a one-way value from a deliberately slow function, with a random salt added per user, so a stolen database cannot be turned back into passwords and identical passwords get different hashes. bcrypt, scrypt and Argon2 are built for this; a fast hash such as SHA-256 is not enough.

## Two-factor authentication: a second proof at login
Two-factor authentication (2FA) asks for two different kinds of proof: usually a password plus something you have, such as a code from an app, a hardware key or a code by SMS, or something you are, such as a fingerprint. It protects the account when the password is stolen. Codes from an app or a hardware key resist phishing better than SMS.

## Backup: a separate copy that can be restored
A backup is a copy of data kept somewhere other than the original, so it can be restored after a disk failure, a deleted file, ransomware or a disaster. A backup is only proven when a restore has been tried. A common rule is three copies, on two kinds of media, with one kept off site.

## Virtual machine: a computer simulated in software
A virtual machine (VM) is a complete computer built in software by a hypervisor: it has virtual disks, memory and network cards and runs its own operating system. Several VMs can share one physical machine, each isolated from the others. VMs are used for servers, testing other systems and running software written for another OS.

## Container: a packaged program sharing the host kernel
A container bundles a program with its libraries and settings so it runs the same anywhere, and it uses the host's kernel instead of running a whole operating system of its own. Containers therefore start in seconds and are smaller than virtual machines, but they share the kernel and are less isolated. Docker is the best-known tool for building and running them.

## Cloud: computing rented over the network
Cloud computing means using computers, storage and services that run in someone else's data centre and are reached over the internet, instead of owning the hardware. It is rented by use, can grow or shrink quickly, and comes as services from basic servers to databases and machine learning. Your data then sits on machines you do not control.

## JSON: a text format for structured data
JSON (JavaScript Object Notation) writes data as objects `{}` with quoted keys, arrays `[]`, strings in double quotes, numbers, and the values `true`, `false` and `null`. A record looks like `{"name": "Ana", "tags": ["a", "b"], "age": 30}`. It has no comments and no trailing commas, and it is the usual format for web APIs and configuration.

## CSV: tabular data as comma-separated text
CSV (comma-separated values) stores a table as plain text, one row per line, with fields separated by commas and a header row of column names usually first. A field containing a comma, quote or line break is wrapped in double quotes, and a quote inside is doubled. There are no data types and dialects vary, so spreadsheets and databases ask which separator and encoding to use.

## XML: a markup format for structured documents
XML writes data as nested tags with attributes and text, such as `<book id="1"><title>Dune</title></book>`. Tags must be closed and nested properly, and the format is strict, unlike HTML. It is self-describing and can be checked against a schema, which makes it common in older enterprise systems and document formats; it is more verbose than JSON.

## YAML: an indentation-based configuration format
YAML writes data with indentation instead of brackets: a mapping is `key: value` and a list is lines starting with `- `. It supports comments, multi-line text and references, and is common for configuration files. Indentation is meaningful and tabs must not be used for it, so a YAML file that looks right can fail on one misplaced space.

## Markdown: plain text that formats into HTML
Markdown is a light markup for writing with plain characters: `# Heading`, `**bold**`, `*italic*`, `[text](url)`, `- item` for a list and backticks for code. It stays readable as raw text and converts to HTML. It is used for README files, notes, documentation and many web pages; versions such as CommonMark settle the details.

## PDF: a document with a fixed layout
A PDF (Portable Document Format) describes each page with fixed positions for text, images, fonts and vector shapes, so it looks the same on every device and prints as designed. Fonts can be embedded, text is usually selectable in digital PDFs, and scanned pages are images unless text is added by OCR. It is made for sharing and printing, not for easy editing.

## PNG: lossless image format with transparency
PNG (Portable Network Graphics) stores raster images without losing quality, supports full transparency and is well suited to screenshots, logos, diagrams and images with sharp edges or few colours. For photographs it produces much larger files than JPEG, so it is the wrong choice for photo galleries. Saving and reopening a PNG does not degrade it.

## JPEG: lossy format for photographs
JPEG compresses images by discarding detail the eye notices less, so it makes photos small but loses quality on every save. It has no transparency and shows visible blocks and rings around sharp edges at strong compression. Use it for photographs and images with smooth colour, and keep the original file when you may need to edit again.

## GIF: old format for short animations
GIF stores images with at most 256 colours per frame and supports simple animation and one-colour transparency. That limit makes it poor for photographs and large for video, but it plays everywhere and loops automatically, so it survives for short silent clips and simple graphics. WebP and MP4 are better for anything longer.

## WebP: modern image format for photos and graphics
WebP is an image format that supports lossy and lossless compression, transparency and animation, usually producing smaller files than JPEG or PNG at similar quality. Browsers accept it widely. Because it is smaller and more flexible it often replaces both formats, though some older software cannot open it.

## SVG: images described as vector shapes
SVG (Scalable Vector Graphics) is an XML format that describes images as shapes, paths and text rather than pixels. It stays sharp at any size, its text is searchable and it can be styled with CSS and changed by scripts. It suits icons, logos, diagrams and charts but not photographs, which are made of pixels.

## MP3 vs WAV: compressed and uncompressed audio
MP3 compresses audio by removing parts the ear is unlikely to notice, giving small files that are fine for listening and sharing but lose quality. WAV stores the samples without compression, so files are much larger but exact, which is why recording and editing use it. Converting a WAV to MP3 throws detail away; converting back does not restore it.

## MP4: a video container holding streams
MP4 (MPEG-4 Part 14) is a container file for video, audio, subtitles and metadata, with information on how the streams are timed. The quality and size come from the codecs inside it, commonly H.264 video and AAC audio. The same MP4 can hold different codecs, so a player that opens one file may still fail on another.

## Codec: the algorithm that compresses audio or video
A codec (coder-decoder) is the method used to compress a media stream and expand it again for playback, such as H.264 or H.265 for video and AAC, MP3 or Opus for audio. A container format like MP4 or MKV only packages the streams; the codec decides quality, size and compatibility. Both the file and the player need to support the codec.

## Pixel: one dot of a digital image or screen
A pixel is the smallest unit of a raster image or display, holding one colour. On a screen the colour comes from tiny red, green and blue subpixels, and an image is a grid of pixels, for example 1920 by 1080. More pixels in the same physical space means a sharper picture; a photo enlarges well only up to its pixel count.

## Resolution: how many pixels, or how dense
Resolution usually means the number of pixels in an image or display, written as width by height, such as 1920 by 1080. It can also mean pixel density, pixels per inch: a phone and a large monitor can share a resolution and look very different in sharpness because their pixels are different sizes. Print resolution is given in dots per inch.

## Refresh rate: how often a screen redraws
Refresh rate is how many times per second a display redraws its picture, measured in hertz: 60 Hz means 60 times a second. A higher rate makes motion look smoother and reduces blur, which matters for games and fast video. It differs from frame rate, the number of frames per second in the video or game itself.

## Binary numbers: counting in base 2
Binary writes numbers with only the digits 0 and 1, each place worth twice the one to its right: `1011` is 8 + 2 + 1, or 11. Computers use it because a circuit can reliably tell two states, such as low and high voltage, and 8 bits form a byte. One byte holds values from 0 to 255.

## Hexadecimal numbers: base 16 shorthand for bits
Hexadecimal uses sixteen digits, 0 to 9 and A to F, where A is 10 and F is 15. One hex digit stands for exactly 4 bits, so a byte is two hex digits, which makes hex a compact way to read binary. It appears in colour codes such as `#ff8800`, memory addresses and byte dumps, often written with a `0x` prefix.

## Floating point: why 0.1 + 0.2 is not exactly 0.3
Most languages store decimals in binary floating point, which can hold fractions such as 0.5 exactly but not 0.1, in the same way that 1/3 has no exact decimal form. So `0.1 + 0.2` gives `0.30000000000000004`. Compare such results with a small tolerance, or use a decimal type for money.

## Recursion: a function that calls itself
Recursion solves a problem by having a function call itself on a smaller version of the same problem, such as a folder scan that scans each subfolder the same way. Every recursion needs a base case that stops it, or it runs until the program crashes with a stack overflow. It often replaces a loop when the data is nested.

## Algorithm: a finite recipe for solving a problem
An algorithm is a clear, finite sequence of steps that turns an input into a result, such as sorting a list or finding the shortest route. Different algorithms solve the same problem at very different costs. That cost is described by how the work grows with the input size, written with big O, such as O(n) for one pass over n items.

## URL: the address of something on the web
A URL (uniform resource locator) identifies a resource and says how to reach it: `https://example.com:443/docs/page?x=1#top`. It has a scheme (`https`), a host (`example.com`), an optional port, a path, an optional query after `?` and an optional fragment after `#`. Browsers show a shortened form and hide some parts.

## IP address: the number that identifies a device on a network
An IP address is the number a device uses on a network, like a postal address for data. IPv4 addresses are 32 bits written as four numbers, such as `192.168.0.1`; IPv6 addresses are 128 bits in hex. Some ranges are private and used only inside local networks, and `127.0.0.1` means the machine itself.

## DNS: the system that turns names into addresses
DNS (domain name system) translates human names such as `example.com` into IP addresses, so browsers do not need to remember numbers. It works as a hierarchy of servers, from the root through the top-level domain to the site's own records, and answers are cached for a time set by each record. A DNS failure makes every site unreachable by name.

## HTTP and HTTPS: how browsers talk to servers
HTTP is the request-response protocol of the web: the client sends a method and path with headers, the server answers with a status code such as 200 or 404, headers and usually a body. HTTPS is HTTP carried inside TLS, which encrypts the traffic and proves the server's identity with a certificate. Browsers mark plain HTTP as not secure.

## Cookie: a small piece of data a site stores in the browser
A cookie is a short name and value that a website asks the browser to store, and the browser sends it back with later requests to the same site. Cookies keep you logged in and remember settings, and they are also used to track visitors across sites. Flags such as HttpOnly, Secure and SameSite limit how scripts and other sites can use them.

## Port: a numbered door for a service on a machine
A port is a number from 0 to 65535 that identifies which service on a machine a connection is for, so one IP address can host a web server, a mail server and ssh at once. Well-known ports include 80 for HTTP, 443 for HTTPS, 22 for SSH and 53 for DNS. A blocked port means the service is unreachable even if the machine is up.

## Latency and bandwidth: delay versus amount
Latency is the time a message takes to reach the other side, measured in milliseconds; bandwidth is how much data can flow per second. A wide pipe with long latency still feels slow for interactive use, such as a video call or a game, because every action waits for a round trip. Downloading a large file depends mostly on bandwidth.

## Compression: making data smaller
Compression stores data in fewer bytes by finding repetition and patterns. Lossless compression, as in ZIP or PNG, restores the exact original; lossy compression, as in JPEG, MP3 and most video, throws away detail to shrink much further. Archives and downloads use lossless, while media uses lossy to fit reasonable sizes.

## Package manager: installing software and its dependencies
A package manager downloads, installs, updates and removes software and the libraries it depends on, keeping track of versions. Examples are apt and dnf on Linux, Homebrew on macOS, pip for Python, npm for JavaScript and NuGet for .NET. A lock file records the exact versions a project used, so another machine can install the same set.

## IDE and code editor: tools for writing code
A code editor is a program for writing source code, with syntax highlighting, search and often extensions; a full IDE (integrated development environment) adds a debugger, a build system, refactoring tools and project-wide understanding of the code. Small edits suit a light editor, larger projects suit an IDE. The line between the two has blurred.

## File permissions: who may read, write or run a file
File permissions say which users may read, change or run a file or folder. On Linux and macOS each file has separate rights for its owner, its group and everyone else, written like `rwxr-xr--` and changed with `chmod`. Windows uses access control lists with per-user entries. Running as an administrator bypasses many checks, which is why it is risky.

## Environment variable: a named setting passed to programs
An environment variable is a name and value that the system passes to every program it starts, such as `HOME`, `PATH` or `LANG`. Programs read them for configuration and to find tools; `PATH` lists the folders searched for commands. They are convenient for settings and secrets but visible to the processes that inherit them, so secrets still belong in a proper store.
