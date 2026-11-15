// KIWIBRO: sample flow for a support request; replace the steps and links with the project's own.
// Each step has a unique id, its text and a kind: "start", "step" (the default), "choice" or "end".
// A link goes from one id to another and may carry a label such as "yes"; a link back up the chart becomes a loop.
const FLOW = {
  title: "{{title}}",
  steps: [
    { id: "in", text: "Support request arrives", kind: "start" },
    { id: "read", text: "Read the request and check the account" },
    { id: "known", text: "Known issue?", kind: "choice" },
    { id: "answer", text: "Send the saved answer" },
    { id: "logs", text: "Ask for logs and steps to reproduce" },
    { id: "repro", text: "Reproduce the problem" },
    { id: "fixed", text: "Fixed?", kind: "choice" },
    { id: "apply", text: "Apply the fix and tell the customer" },
    { id: "close", text: "Close the ticket", kind: "end" },
  ],
  links: [
    { from: "in", to: "read" },
    { from: "read", to: "known" },
    { from: "known", to: "answer", label: "yes" },
    { from: "known", to: "logs", label: "no" },
    { from: "logs", to: "repro" },
    { from: "repro", to: "fixed" },
    { from: "fixed", to: "apply", label: "yes" },
    { from: "fixed", to: "logs", label: "no" },
    { from: "answer", to: "close" },
    { from: "apply", to: "close" },
  ],
};
