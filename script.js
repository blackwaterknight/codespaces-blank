async function send() {
  const input = document.getElementById("input");
  const chatBox = document.getElementById("chatBox");

  const message = input.value;

  chatBox.innerHTML += `<p><b>You:</b> ${message}</p>`;

  const res = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message })
  });

  const data = await res.json();

  let reply = "Sorry, no response";

  try {
    reply = data.output[0].content[0].text;
  } catch (e) {}

  chatBox.innerHTML += `<p><b>AI:</b> ${reply}</p>`;

  input.value = "";
}
