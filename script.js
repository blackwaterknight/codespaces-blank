async function send() {
  const input = document.getElementById("input");
  const chatBox = document.getElementById("chatBox");

  const message = input.value;

  if (!message) return;

  chatBox.innerHTML += `<div class="message user">${message}</div>`;

  input.value = "";

  const res = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message })
  });

  const data = await res.json();

  let reply = data.output?.[0]?.content?.[0]?.text || "Sorry, no response";

  chatBox.innerHTML += `<div class="message bot">${reply}</div>`;

  chatBox.scrollTop = chatBox.scrollHeight;
}
