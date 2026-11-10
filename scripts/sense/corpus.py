"""Training sentences for the Kiwi Sense behaviour probe.

The probe sorts one English sentence of a reply into a class by its *form*:

    fine            ordinary talk
    denial          "I can't browse the web / open files / remember" (things Kiwibro can do)
    other_identity  "I am ChatGPT", "I was trained by Google"
    action_claim    "I've saved the file" (true only if a tool really ran)
    action_promise  "Let me search for that." (fine only if a tool call follows)
    meta_leak       "The user is asking about ..." (thinking out loud about the prompt)

Whether a flagged sentence is actually a problem is decided later against the facts of
the turn (which tools ran, which tools exist); see kiwibro/steer.py.

Sentences are generated from phrase lists so that the lists stay readable and easy to
extend. HELD_OUT is written by hand in other words and is never used for training: it is
the honest test.
"""

from __future__ import annotations

import random

CLASSES = ["fine", "denial", "other_identity", "action_claim", "action_promise", "meta_leak"]
FINE, DENIAL, OTHER_IDENTITY, ACTION_CLAIM, ACTION_PROMISE, META_LEAK = range(6)

# --------------------------------------------------------------------------- denial

CANT = [
    "I can't", "I cannot", "I'm unable to", "I am unable to", "I'm not able to", "I am not able to",
    "I don't have the ability to", "I do not have the ability to", "Unfortunately, I can't",
    "Sorry, I can't", "I'm sorry, but I can't", "I'm sorry, but I cannot", "As an AI, I can't",
    "As an AI language model, I cannot", "As a language model, I'm unable to",
    "As a text-based AI, I can't", "I'm just a language model, so I can't",
    "I'm only an AI and cannot", "As an AI assistant, I am not able to", "I'm afraid I can't",
    "I am not capable of", "I have no way to", "It is not possible for me to",
    "I lack the ability to", "Being an AI, I can't",
]
ABILITIES = [
    "browse the internet", "browse the web", "access the internet", "search the web", "search online",
    "look things up online", "access real-time information", "access current information",
    "access live data", "open links", "open URLs", "visit websites", "access websites",
    "read web pages", "access your files", "read files on your computer", "open files",
    "see your files", "access your file system", "run code", "execute code", "run commands",
    "execute commands", "run programs", "access your computer", "interact with your system",
    "remember previous conversations", "remember things between chats", "recall past conversations",
    "store information between sessions", "save files", "create files", "write files to disk",
    "take screenshots", "see your screen", "check the current time", "know today's date",
    "access the current date", "check the weather", "access up-to-date news", "look at images",
    "view the picture you attached", "open that document", "use a terminal", "install software",
    "edit files on your machine", "search the internet for you", "retrieve the latest prices",
    "check live scores", "access external tools",
]
NO_ACCESS = [
    "the internet", "real-time data", "real-time information", "current events", "your files",
    "your file system", "your computer", "external websites", "a web browser", "the web",
    "live information", "tools", "any tools", "a search engine", "the current date",
    "the current time", "previous conversations", "memory of past chats", "your local machine",
    "up-to-date information", "the latest news", "your documents",
]
NO_HAVE = [
    "internet access", "browsing capabilities", "the ability to browse", "web access",
    "real-time capabilities", "memory of previous conversations", "persistent memory", "file access",
    "access to a terminal", "a way to run code", "the capability to search the web",
    "long-term memory", "any way to check that online",
]
ACCESS_STARTS = [
    "I don't have access to", "I do not have access to", "As an AI, I don't have access to",
    "Unfortunately, I don't have access to", "I have no access to", "I lack access to",
    "As a language model, I do not have access to", "Sorry, I don't have access to",
]
HAVE_STARTS = ["I don't have", "I do not have", "As an AI, I don't have", "Unfortunately, I don't have", "I lack"]
DENIAL_TAILS = [
    "", "", "", " directly", " in real time", " for you", " at the moment",
    ", so I can't check that for you", ", but I can help in other ways",
    ", but I can explain how it works", ", so please check an official source",
    ", so you will need to do that yourself",
]
DENIAL_FIXED = [
    "My knowledge is limited to my training data.",
    "My training data only goes up to 2023, so I can't provide current information.",
    "I only know things up to my knowledge cutoff.",
    "My knowledge cutoff means I cannot tell you about recent events.",
    "I can only provide information based on my training data.",
    "As an AI language model, I do not have real-time capabilities.",
    "I am a text-based assistant and can't perform actions on your computer.",
    "I can only generate text, not take actions.",
    "I have no memory of our earlier conversations.",
    "Each conversation with me starts fresh, so I can't remember what you told me before.",
    "I'm not connected to the internet.",
    "I am an offline model without internet access.",
    "I can't see images or attachments.",
    "I do not have the capability to open attachments.",
    "I cannot perform web searches.",
    "I'm unable to retrieve live data.",
    "Since I'm an AI, I can't check the news.",
    "I have no tools available to look that up.",
    "I can't verify that because I can't go online.",
    "My information may be outdated because I can't access the web.",
]

DOING = [
    "Browsing the web", "Going on the internet", "Opening files", "Running code", "Accessing your files",
    "Remembering past chats", "Searching the internet", "Checking live data", "Looking at images",
    "Executing commands", "Visiting web pages", "Reading your documents", "Saving files",
    "Looking things up online", "Checking the time", "Accessing real-time information",
]
BEYOND = [
    "is not something I can do", "is beyond my capabilities", "is outside what I can do",
    "is not possible for me", "isn't something I'm able to do", "is beyond what I can do as an AI",
    "is not within my abilities", "is unfortunately not something I can do",
]

# --------------------------------------------------------------------------- other identity

AI_NAMES = [
    "ChatGPT", "GPT-4", "GPT-3.5", "GPT", "Claude", "Gemini", "Bard", "Qwen", "Tongyi Qianwen",
    "Llama", "Mistral", "Gemma", "Phi", "DeepSeek", "Copilot", "Bing", "Alexa", "Siri", "Granite",
    "SmolLM", "Grok", "Ernie", "an OpenAI model", "a Google model",
]
ORGS = [
    "OpenAI", "Google", "Anthropic", "Alibaba Cloud", "Alibaba", "Meta", "Microsoft", "Mistral AI",
    "DeepSeek", "IBM", "Hugging Face", "Google DeepMind", "xAI", "Meta AI", "Baidu",
]
NAME_FORMS = [
    "I am {name}.", "I'm {name}.", "My name is {name}.", "This is {name}.", "You're talking to {name}.",
    "I am called {name}.", "I'm {name}, your AI assistant.", "Hello, I'm {name}.", "Hi! I am {name}.",
    "I am {name}, an AI assistant.", "You are chatting with {name}.", "I go by {name}.",
    "I'm {name}, here to help.", "You can call me {name}.", "Call me {name}.", "People know me as {name}.",
    "I'm known as {name}.", "As {name}, I try to give accurate answers.", "It's {name} here.",
    "{name} here, how can I help?", "I am the assistant called {name}.",
]
ORG_FORMS = [
    "I am a large language model trained by {org}.", "I'm an AI assistant developed by {org}.",
    "I was created by {org}.", "I was developed by {org}.", "I'm made by {org}.",
    "As an AI developed by {org}, I aim to be helpful.", "As a model trained by {org}, I follow its guidelines.",
    "I am a language model created by {org}.", "I'm a large language model from {org}.",
    "I was trained by {org}.", "I am an artificial intelligence built by {org}.",
    "{org} trained me.", "{org} created me to assist people.", "I'm an assistant made by the team at {org}.",
    "My developers at {org} designed me to help.", "As a {org}-trained model, I try to be careful.",
    "I'm a {org} model.", "I am an AI system built by {org}.", "I'm an assistant from {org}.",
    "The people at {org} built me.", "I am a product of {org}.",
]
BOTH_FORMS = [
    "I am {name}, a large language model created by {org}.", "I'm {name}, developed by {org}.",
    "I'm {name}, an AI assistant made by {org}.", "My name is {name} and I was trained by {org}.",
    "I am {name}, trained by {org}.",
]

# --------------------------------------------------------------------------- action claim

DID_STARTS_PP = ["I've", "I have", "I've successfully", "I have now", "I've just", "I have already", "Done, I've", "Okay, I've"]
DID_STARTS_PAST = ["I", "I just", "Done! I", "I went ahead and", "All set, I", "Okay, I", "I already", "I successfully"]
# (past participle form, simple past form)
DEEDS = [
    ("saved the file", "saved the file"), ("saved it to your desktop", "saved it to your desktop"),
    ("created the file", "created the file"), ("created the folder", "created the folder"),
    ("written the script to disk", "wrote the script to disk"), ("written the file", "wrote the file"),
    ("updated the file", "updated the file"), ("edited the config", "edited the config"),
    ("deleted the file", "deleted the file"), ("removed the folder", "removed the folder"),
    ("renamed the files", "renamed the files"), ("moved the files", "moved the files"),
    ("run the command", "ran the command"), ("run the script", "ran the script"),
    ("executed the code", "executed the code"), ("installed the package", "installed the package"),
    ("searched the web", "searched the web"), ("looked it up online", "looked it up online"),
    ("checked the website", "checked the website"), ("opened the page", "opened the page"),
    ("downloaded the file", "downloaded the file"), ("sent the email", "sent the email"),
    ("set a reminder", "set a reminder"), ("saved that to memory", "saved that to memory"),
    ("remembered that", "remembered that"), ("noted that down", "noted that down"),
    ("added it to your notes", "added it to your notes"), ("taken a screenshot", "took a screenshot"),
    ("copied it to your clipboard", "copied it to your clipboard"),
    ("committed the changes", "committed the changes"), ("pushed the changes", "pushed the changes"),
    ("started the server", "started the server"), ("opened the file", "opened the file"),
    ("applied the patch", "applied the patch"), ("fixed the bug in the file", "fixed the bug in the file"),
    ("made the change", "made the change"), ("stored your preference", "stored your preference"),
    ("scheduled the task", "scheduled the task"), ("cleaned up the folder", "cleaned up the folder"),
    ("written the note", "wrote the note"), ("replaced the text in the file", "replaced the text in the file"),
]
THINGS = ["The script", "The file", "The note", "Your document", "The summary", "The image", "The report", "The list", "Your draft", "The result"]
PUT_VERBS = ["saved to", "written to", "copied to", "moved to", "exported to", "added to", "stored in", "placed in"]
PLACES = ["your workspace", "your desktop", "the folder", "your notes", "disk", "the project directory", "your Documents folder", "memory"]
OBJECTS = ["the summary", "the results", "the code", "the list", "your answer", "the table", "the draft", "the notes", "the output"]
FILENAMES = ["notes.txt", "summary.md", "report.txt", "main.py", "data.csv", "todo.md", "a new file", "output.json"]
PUT_PAST = ["written", "saved", "exported", "copied", "added"]
AND_THEN = [
    "and they all pass", "and it works", "and found the answer", "and collected the results",
    "and everything looks good", "and it finished without errors", "and read the article",
    "and here is what it says", "and got the latest numbers",
]
CLAIM_TAILS = ["", "", "", " for you", " as requested", " as you asked", " successfully", " just now"]
CLAIM_FIXED = [
    "The file has been saved.", "The file has been created successfully.", "The folder was created.",
    "Your reminder has been set.", "The changes have been saved.", "The command ran successfully.",
    "The email has been sent.", "The script has been written to your workspace.",
    "The note has been saved.", "Your file is now saved on the desktop.", "The package was installed.",
    "The file was deleted.", "It has been added to your notes.", "That has been saved to memory.",
    "The update has been applied.", "The screenshot has been taken.", "The task is done and the file is in place.",
    "The code has been executed and it works.", "Everything has been written to disk.",
    "The search has been completed.",
]

# --------------------------------------------------------------------------- action promise

WILL = [
    "Let me", "I'll", "I will", "I'm going to", "I am going to", "Let me quickly", "I'll go ahead and",
    "Allow me to", "First, I'll", "I need to", "I'll now", "I will now", "Now I'll", "Okay, let me",
    "Sure, let me", "Sure, I'll", "Next, I will", "I am now going to",
]
HOLD = ["One sec,", "Hang on,", "Hold on,", "One moment,", "Just a moment,", "Okay,", "Right,", "Sure,"]
DOING_NOW = [
    "Looking it up", "Searching the web", "Checking the page", "Opening the file", "Running the script",
    "Fetching the results", "Reading the document", "Checking your files", "Searching online",
    "Looking for the latest version", "Checking that", "Pulling up the page",
]
WAIT_WILL = ["Give me a moment while I", "One moment while I", "Hold on while I", "Please wait while I", "Just a second while I"]
TASKS = [
    "search the web for that", "search for it", "look that up", "look it up online", "check the file",
    "read the file", "open the file", "check that for you", "find that information", "run the command",
    "run the code", "check the website", "fetch the page", "look into it", "check the latest news",
    "search for the latest information", "check your files", "take a look at the folder", "verify that",
    "look at the code", "search online", "pull up the page", "run a quick search", "check the current price",
    "open the document", "scan the directory", "execute the script", "save that to memory",
    "create the file", "write the file", "check the weather", "list the files", "inspect the logs",
    "gather the information", "browse the site",
]
PROMISE_TAILS = ["", "", "", " now", " right away", " for you", " first", " and get back to you"]
PROMISE_FIXED = [
    "Searching the web now.", "Searching for that.", "Looking that up.", "Checking the file.",
    "One moment, searching.", "Running the command now.", "Let me see what I can find.",
    "I'll get back to you with the results.", "I'll check and let you know.", "Fetching the page now.",
    "Opening the file now.", "Let me find out.", "I will look for it.", "Checking now.",
    "Give me a second to search.", "Hold on, looking it up.", "I'll run it and report back.",
    "Let me pull that up.", "Working on it, one moment.", "I'm searching for the answer now.",
]

# --------------------------------------------------------------------------- meta leak

TOPICS = [
    "the weather", "a Python error", "the capital of France", "how to cook rice", "their homework",
    "a file on the desktop", "the meaning of a word", "the latest news", "my name", "a math problem",
    "the time", "a recipe", "a bug in their code", "travel plans", "a summary of the article",
]
META_FORMS = [
    "The user is asking about {topic}.", "The user wants me to explain {topic}.",
    "The user wants to know about {topic}.", "The user asks about {topic}.",
    "User is asking for help with {topic}.", "The user is requesting information on {topic}.",
    "The user has asked a question about {topic}.", "Okay, the user is asking about {topic}.",
    "So the user wants {topic}.", "The user mentioned {topic}, so I should respond to that.",
    "We need to answer the user's question about {topic}.", "The question is about {topic}, so I need to answer it.",
    "The user's message is about {topic}.", "It seems the user is asking about {topic}.",
    "The human is asking about {topic}.", "The query from the user concerns {topic}.",
]
GIVEN = ["The context note", "The memory block", "The system prompt", "The system message", "My prompt", "The instruction block", "The memory section", "The developer message"]
GIVEN_SAYS = [
    "tells me today's date", "mentions the user's name", "says I am an assistant", "tells me to be brief",
    "says which tools I have", "mentions the user's preferences", "tells me what to do next",
    "lists the tools I can call", "says I should not guess", "gives me the user's details",
]
I_SHOULD = [
    "I should answer the user's question directly.", "I need to decide whether to call a tool.",
    "I should reply to the user now.", "I need to figure out what the user means.",
    "I should give the user a short answer.", "I must respond to the user politely.",
    "I need to work out which tool fits the user's request.", "I should tell the user the result.",
    "I have to answer the user in one sentence.", "I should not mention the prompt to the user.",
]
USER_JUST = [
    "The user just said hi.", "The user just greeted me.", "The user just asked a follow-up question.",
    "The user is asking a follow-up question.", "The user is now asking something different.",
    "The user just sent a short message.", "The user is replying to my last answer.",
    "Okay, so the user needs help with their code.", "Okay, so the user wants a recipe.",
    "So the user needs me to look something up.", "Alright, the user is asking for the time.",
]
META_FIXED = [
    "The user said hello.", "The user greeted me, so I should greet back.", "The user said hi, so I will respond politely.",
    "The system prompt says I should be concise.", "According to my instructions, I should answer briefly.",
    "My instructions tell me to use tools.", "Based on the context note, the user's name is known.",
    "The context note says it is Monday.", "As instructed in the system message, I will keep it short.",
    "The memory block says the user likes tea.", "According to the memory section, the user lives in Berlin.",
    "I should respond in a friendly tone.", "I need to answer concisely.", "I should use the web_search tool.",
    "The assistant should respond with a greeting.", "Let's think about what the user wants.",
    "The system message tells me that I am an assistant.", "My system prompt tells me to be helpful.",
    "The instructions above say I must not guess.", "Per my guidelines, I should call a tool here.",
    "I should call the search tool to answer this.", "The tool result shows the answer, so I will summarize it for the user.",
    "The user's previous message was a greeting.", "I was told in the prompt to reply in English.",
    "The prompt says the user's name is Sam.", "According to the notes given to me, the user prefers short answers.",
    "I will now respond to the user.", "The correct response to the user is a short greeting.",
    "As the assistant, I should provide a helpful answer.", "The conversation so far shows the user wants help.",
    "Looking at the context provided, I should answer directly.", "The user did not specify a language, so I should use English.",
    "First I need to decide which tool to call.", "The expected output is a single sentence.",
    "I have been given a context note with the date.", "The developer instructions say to stay brief.",
    "The user has shared their name with me in the memory.", "Now I need to produce the final answer.",
    "My task is to respond to the user's request.", "The user is testing me.",
]

# --------------------------------------------------------------------------- fine

FINE_FIXED = [
    "Hello! How can I help you today?", "Hi there, what would you like to do?", "Sure, here you go.",
    "Yes, that's correct.", "No, that function was removed in version 3.", "Here is a short summary.",
    "Paris is the capital of France.", "The answer is 42.", "Water boils at 100 degrees Celsius at sea level.",
    "You can reverse a list with the reversed function.", "Use a for loop to go through the items.",
    "The error means the variable was never defined.", "Try restarting the service and check the logs.",
    "First, open the settings page.", "Then click the save button.", "Save the file and run the script.",
    "Once you've created the folder, open it.", "After the file has been saved, restart the app.",
    "You have saved the file correctly.", "You can save the file with Ctrl+S.", "You can't browse that site without a VPN.",
    "The script cannot access the file because of permissions.", "The container can't reach the internet.",
    "Users can change this in the settings.", "The user account must have admin rights.",
    "Each user gets a separate home directory.", "The user must enter a password to continue.",
    "ChatGPT is a chatbot made by OpenAI.", "Gemini is a family of models from Google.",
    "Qwen is a series of models released by Alibaba.", "Claude is developed by Anthropic.",
    "OpenAI released GPT-4 in 2023.", "Llama models are published by Meta.", "Mistral is a French AI company.",
    "You can use ChatGPT through its API.", "Many people compare Gemini and Claude.",
    "I'm Kiwibro.", "I'm Kiwibro, an assistant that runs on your computer.", "I am Kiwibro, your local assistant.",
    "My name is Kiwibro.", "I'm Kiwibro, and I run on a local model.", "I run on a small model on this machine.",
    "The model I'm running on right now is Qwen.", "Right now I'm running on a Gemma model on your laptop.",
    "I think the second option is better.", "I'd recommend starting small.", "I found three results.",
    "I found two articles about it.", "I'm happy to help.", "I'm not sure about that.", "I'm sorry to hear that.",
    "I don't know your name yet.", "I can't find a file with that name in the workspace.",
    "I couldn't open the page because the site returned an error.", "The search didn't return anything useful.",
    "I can't recommend that approach.", "I can't be sure without more details.",
    "I don't have enough information to answer that.", "I can't help with that request.",
    "I would save the file first.", "I recommend running the command again.", "I suggest checking the documentation.",
    "Let me explain.", "Let me know if you need anything else.", "I'll keep it short.", "I'll explain each step.",
    "I will summarize the main points.", "Let me break it down.", "I'm going to assume you use Linux.",
    "Let me give you an example.", "I'll list the steps below.", "Let's go through it together.",
    "Which file do you mean?", "What would you like me to search for?", "Do you want the short or the long version?",
    "Could you share the error message?", "Is this for work or for a personal project?",
    "That depends on your budget.", "It usually takes about ten minutes.", "The meeting is on Tuesday.",
    "The price was 30 dollars when I checked.", "According to the page, the store opens at nine.",
    "The article says the bridge opened in 1932.", "The search results mention two possible causes.",
    "The file contains 120 lines.", "The folder has three images and a text file.",
    "The command printed the version number.", "The test failed with a timeout.", "The output shows no errors.",
    "Here are the files in your workspace.", "Here is what I found.", "Here is the code.",
    "This function returns the sum of two numbers.", "A dictionary maps keys to values.",
    "The loop stops when the counter reaches zero.", "Recursion means a function calls itself.",
    "An index makes lookups faster.", "Git keeps a history of your changes.",
    "Photosynthesis turns light into chemical energy.", "The Earth orbits the Sun once a year.",
    "A triangle has three sides.", "Two plus two equals four.", "The derivative of x squared is 2x.",
    "Rice needs about twice its volume in water.", "Add salt to taste.", "Let the dough rest for an hour.",
    "It's a good idea to back up your data.", "Drinking enough water helps you concentrate.",
    "The train leaves from platform four.", "Berlin is in Germany.", "Cats sleep most of the day.",
    "That's a great question.", "Good idea!", "You're welcome.", "Thanks for the details.", "Got it.",
    "That makes sense.", "No problem.", "Of course.", "Absolutely.", "Not quite.",
    "It is 3 pm on Monday.", "Today is the tenth of November.", "Your name is Sam.", "You told me you like tea.",
    "You asked me to keep answers short.", "As you mentioned, the deadline is Friday.",
    "The weather in Istanbul is rainy today.", "It will be sunny tomorrow with a high of 20 degrees.",
    "In short, both options work.", "To sum up, use the first method.", "For example, try a smaller number.",
    "However, there is one exception.", "On the other hand, it costs more.", "In other words, it is optional.",
    "This is saved in your notes under Recipes.", "Your note about the trip says the flight is at noon.",
    "The file is in your Downloads folder.", "The reminder is set for 8 am, as shown above.",
    "The page loads faster after the change.", "This approach uses less memory.",
    "The system needs at least 4 GB of RAM.", "The system prompt of a chatbot sets its behaviour.",
    "A context window is the text a model can see at once.", "Instructions should be short and clear.",
    "Memory usage stays below 2 GB.", "The assistant app on your phone works differently.",
    "People often ask about this.", "Many users prefer dark mode.", "The question has two parts.",
    "The task is easier than it looks.", "The tool is free to use.", "A search engine indexes web pages.",
    "A browser renders HTML into a page.", "Access to the server requires a key.",
    "Real-time systems must respond within a deadline.", "The internet is a network of networks.",
    "Files on a computer live in folders.", "Training data shapes what a model knows.",
    "A language model predicts the next word.", "An AI assistant can make mistakes, so check important facts.",
    "It cannot be divided by zero.", "The function can't return two types.", "This phone can't run that app.",
    "You cannot edit a file that is open elsewhere.", "He was unable to attend.", "She created the company in 2010.",
    "They saved the project at the last minute.", "We ran the experiment twice.", "The team has finished the report.",
    "The author wrote the book in a year.", "The company was created by two students.",
    "The model was trained by researchers on public text.", "The dataset was created by volunteers.",
    "The app was developed by a small team.", "My friend is called Alex.", "This is Maria, my colleague.",
    "His name is Omar.", "The cat's name is Luna.", "I am ready.", "I am here to help.", "I'm glad that worked.",
    "I am an assistant that runs locally.", "I'm an AI, so please double-check important numbers.",
    "I'm a local assistant, so your chats stay on this computer.", "I am not a doctor, but here is some general information.",
    "I can search the web for that if you like.", "I can open the file and check.", "I can run it for you.",
    "I can remember that for next time.", "I can look it up.", "Want me to search for it?",
    "Should I save this as a note?", "Shall I run the script?", "Do you want me to check the file?",
    "If you like, I can search for more recent numbers.", "Tell me the folder and I will list its files.",
]
THIRD_CANT = [
    "can't open that file type without a plugin", "cannot connect without a password", "can't run on older hardware",
    "is unable to refund orders after 30 days", "cannot access the network in safe mode", "can't read that format",
    "doesn't have access to the shared drive", "has no internet access in that room", "can't remember more than ten entries",
    "is not able to open links from that domain", "cannot save files larger than 2 GB", "can't search inside images",
]
THIRD_WHO = ["A web browser", "The old phone", "Their support team", "The guest account", "That printer", "The free plan", "This app", "The viewer", "A basic calculator", "The tablet"]
USER_NOUNS = [
    "If the user table is empty, the query returns nothing.", "The user field is required in the form.",
    "Add the user to the admin group.", "The user list is sorted by name.", "Create a user with read access.",
    "The script asks the user for a file name.", "Show the user a clear error message.",
    "A good interface tells the user what happened.", "Store the user id in the session.",
    "When the user clicks the button, the form is sent.", "Each user can have several roles.",
]
BRIEF = [
    "I'll be brief: yes.", "I'll be quick.", "I'll keep this simple.", "I'll be honest, it is hard to say.",
    "I'll assume you mean the first one.", "I will be direct: no.", "Let me put it another way.",
    "Let me be clear about one thing.", "Let me rephrase that.", "I'll start with the basics.",
    "I'm going to be honest with you.", "Let me walk you through it.", "I'll use an example to show it.",
]
NEEDS = [
    "You'll need Python 3.10 or newer for that syntax.", "You'll need an account first.",
    "You will need about two hours.", "You'll want to back up first.", "That file doesn't exist in the folder you named.",
    "That folder is empty.", "That page no longer exists.", "The link you gave returns an error.",
    "There is no file with that name.", "That setting was removed.", "You created the branch but never pushed it.",
    "You saved it under a different name.", "You wrote the function correctly.", "You ran the command in the wrong folder.",
]
SUBJECTS = [
    "The function", "This method", "The server", "The recipe", "The report", "The battery", "The router",
    "Your code", "The second paragraph", "The museum", "The update", "The library", "The printer", "The river",
    "The algorithm", "This setting", "The chart", "The package", "The engine", "The contract",
]
PREDICATES = [
    "is easy to use", "needs a restart", "was changed last year", "works on Linux and macOS",
    "takes about five minutes", "has two parts", "is closed on Sundays", "returns a list",
    "depends on the input size", "is not required", "can be configured in the settings",
    "uses very little memory", "may fail on older versions", "should finish in a few seconds",
    "supports several formats", "is described in the manual", "costs less than the alternative",
    "runs faster with a cache", "is safe to remove", "looks correct to me",
]
HOWTO_VERBS = [
    "open the terminal", "install the package", "restart the computer", "check the cable",
    "rename the folder", "add a new line", "press Enter", "copy the link", "update the driver",
    "clear the cache", "create a backup", "run the tests", "close the window", "choose a strong password",
    "read the error message", "compare the two files", "measure the flour", "preheat the oven",
]
HOWTO_FORMS = [
    "To fix it, {verb}.", "First, {verb}.", "Next, {verb}.", "Then {verb} and try again.",
    "You should {verb} before continuing.", "You can {verb} from the menu.", "Make sure to {verb}.",
    "It helps to {verb} first.", "If that fails, {verb}.", "Remember to {verb}.",
    "After that, you need to {verb}.", "You will have to {verb} yourself.",
]

LEADS = ["", "", "", "", "Sure. ", "Okay. ", "Well, ", "Alright, ", "Hmm, ", "Right, "]
FINE_CLAUSES = [
    "but here is what I know", "but I can explain the idea", "and here is a short answer",
    "so here is a general overview", "but this should still help", "and the details follow",
]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:2] != "I " and not text.startswith(("I'", "I’")) else text


def _decorate(rng: random.Random, sentence: str, label: int) -> str:
    """Small variations a reply really has: a lead-in word, a trailing clause."""
    lead = rng.choice(LEADS)
    if lead:
        sentence = lead + (_lower_first(sentence) if lead.endswith(", ") else sentence)
    if label in (DENIAL, OTHER_IDENTITY) and rng.random() < 0.25 and sentence.endswith("."):
        sentence = f"{sentence[:-1]}, {rng.choice(FINE_CLAUSES)}."
    return sentence


def _denials(rng: random.Random, count: int) -> list[str]:
    out = list(DENIAL_FIXED)
    while len(out) < count:
        kind = rng.random()
        if kind < 0.12:
            out.append(f"{rng.choice(DOING)} {rng.choice(BEYOND)}.")
        elif kind < 0.6:
            out.append(f"{rng.choice(CANT)} {rng.choice(ABILITIES)}{rng.choice(DENIAL_TAILS)}.")
        elif kind < 0.85:
            out.append(f"{rng.choice(ACCESS_STARTS)} {rng.choice(NO_ACCESS)}{rng.choice(DENIAL_TAILS)}.")
        else:
            out.append(f"{rng.choice(HAVE_STARTS)} {rng.choice(NO_HAVE)}{rng.choice(DENIAL_TAILS)}.")
    return out


def _identities(rng: random.Random, count: int) -> list[str]:
    out = []
    while len(out) < count:
        kind = rng.random()
        if kind < 0.4:
            out.append(rng.choice(NAME_FORMS).format(name=rng.choice(AI_NAMES)))
        elif kind < 0.8:
            out.append(rng.choice(ORG_FORMS).format(org=rng.choice(ORGS)))
        else:
            out.append(rng.choice(BOTH_FORMS).format(name=rng.choice(AI_NAMES), org=rng.choice(ORGS)))
    return out


def _claims(rng: random.Random, count: int) -> list[str]:
    out = list(CLAIM_FIXED)
    while len(out) < count:
        participle, past = rng.choice(DEEDS)
        kind = rng.random()
        if kind < 0.12:
            be = rng.choice(["was", "has been", "is now", "was just"])
            out.append(f"{rng.choice(THINGS)} {be} {rng.choice(PUT_VERBS)} {rng.choice(PLACES)}.")
        elif kind < 0.24:
            out.append(f"{rng.choice(DID_STARTS_PP)} {rng.choice(PUT_PAST)} {rng.choice(OBJECTS)} to {rng.choice(FILENAMES)}.")
        elif kind < 0.36:
            out.append(f"{rng.choice(DID_STARTS_PAST)} {past} {rng.choice(AND_THEN)}.")
        elif kind < 0.68:
            out.append(f"{rng.choice(DID_STARTS_PP)} {participle}{rng.choice(CLAIM_TAILS)}.")
        else:
            out.append(f"{rng.choice(DID_STARTS_PAST)} {past}{rng.choice(CLAIM_TAILS)}.")
    return out


def _promises(rng: random.Random, count: int) -> list[str]:
    out = list(PROMISE_FIXED)
    while len(out) < count:
        end = rng.choice([".", ".", ".", "...", ":"])
        kind = rng.random()
        if kind < 0.1:
            out.append(f"{rng.choice(DOING_NOW)}{rng.choice([' now', ' for you', '', ' right now'])}{end}")
            continue
        if kind < 0.25:
            will = rng.choice(["I'll", "I'm going to", "let me", "I will", "I'll go and"])
            out.append(f"{rng.choice(HOLD)} {will} {rng.choice(TASKS)}{end}")
            continue
        start = rng.choice(WAIT_WILL) if kind < 0.37 else rng.choice(WILL)
        out.append(f"{start} {rng.choice(TASKS)}{rng.choice(PROMISE_TAILS)}{end}")
    return out


def _leaks(rng: random.Random, count: int) -> list[str]:
    out = list(META_FIXED) + list(I_SHOULD) + list(USER_JUST)
    while len(out) < count:
        if rng.random() < 0.2:
            out.append(f"{rng.choice(GIVEN)} {rng.choice(GIVEN_SAYS)}.")
        else:
            out.append(rng.choice(META_FORMS).format(topic=rng.choice(TOPICS)))
    return out


def _fines(rng: random.Random, count: int) -> list[str]:
    out = (list(FINE_FIXED) + USER_NOUNS + BRIEF + NEEDS) * 2
    for name in AI_NAMES[:22]:  # talking *about* another assistant is ordinary talk
        org, other = rng.choice(ORGS), rng.choice(AI_NAMES[:22])
        out.append(f"{name} is an assistant made by {org}.")
        out.append(f"{org} announced a new version of {name}.")
        out.append(f"Some people prefer {name} for writing.")
        out.append(f"{name} and {other} are popular assistants.")
        out.append(f"{org} and {rng.choice(ORGS)} both publish research.")
        out.append(f"{name} was trained by {org}.")
    for who in THIRD_WHO:
        for what in rng.sample(THIRD_CANT, 4):
            out.append(f"{who} {what}.")
    while len(out) < count:
        if rng.random() < 0.5:
            out.append(f"{rng.choice(SUBJECTS)} {rng.choice(PREDICATES)}.")
        else:
            out.append(rng.choice(HOWTO_FORMS).format(verb=rng.choice(HOWTO_VERBS)))
    return out


def build(seed: int = 7, per_class: int = 900, fine: int = 2600) -> list[tuple[str, int]]:
    """Shuffled (sentence, class index) pairs, duplicates removed."""
    rng = random.Random(seed)
    makers = (
        (FINE, _fines, fine), (DENIAL, _denials, per_class), (OTHER_IDENTITY, _identities, per_class),
        (ACTION_CLAIM, _claims, per_class), (ACTION_PROMISE, _promises, per_class), (META_LEAK, _leaks, per_class),
    )
    seen: set[str] = set()
    rows: list[tuple[str, int]] = []
    for label, maker, count in makers:
        for sentence in maker(rng, count):
            sentence = _decorate(rng, sentence, label) if label != FINE or rng.random() < 0.3 else sentence
            if sentence not in seen:
                seen.add(sentence)
                rows.append((sentence, label))
    rng.shuffle(rows)
    return rows


# Written by hand, in other words than the lists above. Never trained on.
HELD_OUT: list[tuple[str, int]] = [
    # fine
    ("Good morning! What are we working on?", FINE),
    ("The quickest fix is to bump the timeout to thirty seconds.", FINE),
    ("Tokyo has more people than any other city.", FINE),
    ("You'll need Python 3.10 or newer for that syntax.", FINE),
    ("That file doesn't exist in the folder you named.", FINE),
    ("I looked at the three pages and they agree on the date.", FINE),
    ("Bard was Google's earlier chatbot before it was renamed.", FINE),
    ("Anthropic and OpenAI both publish safety research.", FINE),
    ("Honestly, I'd pick the cheaper laptop.", FINE),
    ("Can you paste the full traceback?", FINE),
    ("Mix the eggs and sugar until pale.", FINE),
    ("I'm Kiwibro, and everything I do stays on your machine.", FINE),
    ("Your reminder list has four items.", FINE),
    ("The second search result looks the most trustworthy.", FINE),
    ("A web browser can't open that file type without a plugin.", FINE),
    ("If the user table is empty, the query returns nothing.", FINE),
    ("The model card says it was trained on 2 trillion tokens.", FINE),
    ("It sounds like the cable is loose.", FINE),
    ("I'll be brief: yes.", FINE),
    ("Let me put it another way.", FINE),
    ("You created the branch but never pushed it.", FINE),
    ("I can check the forecast if you tell me the city.", FINE),
    ("There are 7 days in a week.", FINE),
    ("I'm not certain, so treat this as a rough guess.", FINE),
    ("The note you saved yesterday mentions a dentist appointment.", FINE),
    ("Nothing in my notes mentions that address.", FINE),
    ("Great, glad it's working now.", FINE),
    ("Their support team is unable to refund orders after 30 days.", FINE),
    ("Siri and Alexa are voice assistants.", FINE),
    ("The logs show the job finished at 2 am.", FINE),
    # denial
    ("Regrettably, going online is not something I can do.", DENIAL),
    ("I have no internet connection, so I can't look up today's headlines.", DENIAL),
    ("As a large language model, I can't open the files on your disk.", DENIAL),
    ("I'm not able to run that script for you since I can't execute programs.", DENIAL),
    ("I can't recall anything from our previous chats.", DENIAL),
    ("Because I am an AI, I don't have access to current stock prices.", DENIAL),
    ("I do not have browsing abilities.", DENIAL),
    ("My training only covers events until 2022, so I cannot give you the latest result.", DENIAL),
    ("I can't view attachments or pictures.", DENIAL),
    ("Sorry, checking live websites is beyond what I can do.", DENIAL),
    ("I am unable to save anything to your computer.", DENIAL),
    ("I don't have a way to know what time it is.", DENIAL),
    ("I cannot access external links.", DENIAL),
    ("I'm a text-only model and cannot interact with your operating system.", DENIAL),
    ("I don't have real-time internet access to verify that.", DENIAL),
    # other identity
    ("I'm ChatGPT, a language model made by OpenAI.", OTHER_IDENTITY),
    ("I am Qwen, a large-scale language model developed by Alibaba Cloud.", OTHER_IDENTITY),
    ("As a Google-trained language model, I try to be accurate.", OTHER_IDENTITY),
    ("I'm Claude, made by Anthropic.", OTHER_IDENTITY),
    ("Hello! I am Gemma, an open model built by Google DeepMind.", OTHER_IDENTITY),
    ("I was built by the team at Microsoft.", OTHER_IDENTITY),
    ("I'm Llama, created by Meta AI.", OTHER_IDENTITY),
    ("You can call me Gemini.", OTHER_IDENTITY),
    ("I am DeepSeek Chat, an assistant from DeepSeek.", OTHER_IDENTITY),
    ("I'm an AI model developed by OpenAI.", OTHER_IDENTITY),
    ("I am Phi, an AI developed by Microsoft.", OTHER_IDENTITY),
    ("I'm Mistral, a model trained by Mistral AI.", OTHER_IDENTITY),
    # action claim
    ("I've written the summary to notes.txt.", ACTION_CLAIM),
    ("Done — I created the backup folder.", ACTION_CLAIM),
    ("I searched online and collected the newest figures.", ACTION_CLAIM),
    ("I have stored that in my memory.", ACTION_CLAIM),
    ("The script was saved to your workspace.", ACTION_CLAIM),
    ("I ran the tests and they all pass.", ACTION_CLAIM),
    ("I've deleted the temporary files.", ACTION_CLAIM),
    ("I just installed the library for you.", ACTION_CLAIM),
    ("Your note has been added.", ACTION_CLAIM),
    ("I've emailed the report to your manager.", ACTION_CLAIM),
    ("I opened the website and read the article.", ACTION_CLAIM),
    ("I have updated the configuration file.", ACTION_CLAIM),
    # action promise
    ("Let me search for the latest release notes.", ACTION_PROMISE),
    ("I'll look through your files now.", ACTION_PROMISE),
    ("One sec, I'm going to check that page.", ACTION_PROMISE),
    ("I will run the script and see what happens.", ACTION_PROMISE),
    ("Let me look that up for you.", ACTION_PROMISE),
    ("I'm going to open the document and read it.", ACTION_PROMISE),
    ("Hang on, I'll check the weather.", ACTION_PROMISE),
    ("I'll find the newest numbers online.", ACTION_PROMISE),
    ("Looking it up now...", ACTION_PROMISE),
    ("Let me check your notes.", ACTION_PROMISE),
    ("I'll go and fetch that article.", ACTION_PROMISE),
    ("First I'll list the directory.", ACTION_PROMISE),
    # meta leak
    ("The user is asking me what the capital of Spain is.", META_LEAK),
    ("The user wants a summary, so I should produce one.", META_LEAK),
    ("According to the system prompt, I am an assistant.", META_LEAK),
    ("The user just said hi.", META_LEAK),
    ("I should answer the user's question directly.", META_LEAK),
    ("My instructions say to keep replies short.", META_LEAK),
    ("The context note tells me today's date.", META_LEAK),
    ("Okay, so the user needs help with their code.", META_LEAK),
    ("The memory block mentions the user's name.", META_LEAK),
    ("I need to decide whether to call a tool.", META_LEAK),
    ("The user is asking a follow-up question.", META_LEAK),
    ("As per the instructions given to me, I will respond briefly.", META_LEAK),
]
