// KIWIBRO: connect the valid branch below to the project's endpoint
const form = document.querySelector("#contact-form");
const success = document.querySelector("#form-status");

const rules = {
  name: (value) => {
    if (value === "") return "Enter your name.";
    if (value.length < 2) return "Use at least two characters.";
    return "";
  },
  email: (value) => {
    if (value === "") return "Enter your email address.";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) return "That email address does not look right.";
    return "";
  },
  message: (value) => {
    if (value === "") return "Write a message.";
    if (value.length < 10) return "Add a little more, at least ten characters.";
    return "";
  },
};

function setError(field, message) {
  const input = form.elements[field];
  const error = document.querySelector(`#${field}-error`);
  const invalid = message !== "";
  error.textContent = message;
  error.hidden = !invalid;
  if (invalid) input.setAttribute("aria-invalid", "true");
  else input.removeAttribute("aria-invalid");
  return !invalid;
}

function validate(field) {
  const input = form.elements[field];
  return setError(field, rules[field](input.value.trim()));
}

function clearSuccess() {
  success.hidden = true;
  success.textContent = "";
}

Object.keys(rules).forEach((field) => {
  const input = form.elements[field];
  input.addEventListener("blur", () => validate(field));
  input.addEventListener("input", () => {
    clearSuccess();
    if (input.getAttribute("aria-invalid") === "true") validate(field);
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  clearSuccess();

  const invalidFields = Object.keys(rules).filter((field) => !validate(field));
  if (invalidFields.length > 0) {
    form.elements[invalidFields[0]].focus();
    return;
  }

  const data = Object.fromEntries(new FormData(form).entries());
  console.info("Form is valid and ready to send:", data);
  form.reset();
  Object.keys(rules).forEach((field) => setError(field, ""));
  success.textContent = "Checked. The form is valid and ready to send.";
  success.hidden = false;
});
