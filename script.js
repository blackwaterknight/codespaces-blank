async function send() {
  const input = document.getElementById("input");
  const chatBox = document.getElementById("chatBox");

  const message = input.value;
  if (!message) return;

  // User message
  chatBox.innerHTML += `<div class="message user">${message}</div>`;
  input.value = "";

  // Typing animation
  const typingId = "typing-" + Date.now();
  chatBox.innerHTML += `<div id="${typingId}" class="message bot">Typing...</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;

  const res = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message })
  });

  const data = await res.json();
  let reply = data.output?.[0]?.content?.[0]?.text || "Sorry, no response";

  // Remove typing
  document.getElementById(typingId).remove();

  // Detect hospital list (basic logic)
  if (reply.toLowerCase().includes("hospital") || reply.toLowerCase().includes("clinic")) {
    const hospitals = extractHospitals(reply);

    if (hospitals.length > 0) {
      chatBox.innerHTML += renderHospitalCards(hospitals);
    } else {
      chatBox.innerHTML += `<div class="message bot">${reply}</div>`;
    }
  } else {
    chatBox.innerHTML += `<div class="message bot">${reply}</div>`;
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

// 🔍 Extract hospital names (simple parser)
function extractHospitals(text) {
  const lines = text.split(",");
  return lines.slice(0, 3).map(name => name.trim());
}

// 🏥 Render hospital cards
function renderHospitalCards(hospitals) {
  let html = `<div class="card-container">`;

  hospitals.forEach(h => {
    const mapLink = `https://www.google.com/maps/search/${encodeURIComponent(h)}`;

    html += `
      <div class="card">
        <h3>${h}</h3>
        <p>Recommended healthcare provider</p>
        <div class="card-actions">
          <a href="${mapLink}" target="_blank">📍 View Map</a>
          <button onclick="alert('Booking simulated!')">Book</button>
        </div>
      </div>
    `;
  });

  html += `</div>`;
  return html;
}
