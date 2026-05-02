// 🎤 Voice input
function startVoice() {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-US";

  recognition.onresult = function(event) {
    const text = event.results[0][0].transcript;
    document.getElementById("input").value = text;
    send();
  };

  recognition.start();
}

// 🔊 Voice output
function speak(text) {
  const speech = new SpeechSynthesisUtterance(text.substring(0, 150));
  window.speechSynthesis.speak(speech);
}

async function send() {
  const input = document.getElementById("input");
  const chatBox = document.getElementById("chatBox");

  const message = input.value.trim();
  if (!message) return;

  chatBox.innerHTML += `<div class="message user">${message}</div>`;
  input.value = "";

  // Typing indicator
  const typing = document.createElement("div");
  typing.className = "message bot";
  typing.innerText = "Typing...";
  chatBox.appendChild(typing);

  const res = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message })
  });

  const data = await res.json();
  typing.remove();

  const reply = data.output?.[0]?.content?.[0]?.text || "Sorry, no response";

  chatBox.innerHTML += `<div class="message bot">${reply}</div>`;
  speak(reply);

  // Show hospital cards + map if detected
  if (reply.toLowerCase().includes("hospital")) {
    const hospitals = reply.split(",").slice(0, 3);

    if (hospitals.length > 0) {
      chatBox.innerHTML += renderCards(hospitals);
      showMap(hospitals[0]);
    }
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

// 🏥 Cards
function renderCards(hospitals) {
  let html = `<div class="card-container">`;

  hospitals.forEach(h => {
    const clean = h.trim();

    html += `
      <div class="card">
        <h3>${clean}</h3>
        <div class="card-actions">
          <button onclick="showMap('${clean}')">📍 Map</button>
          <button onclick="alert('Appointment booked!')">Book</button>
        </div>
      </div>
    `;
  });

  html += `</div>`;
  return html;
}

// 🗺️ Map
function showMap(place) {
  const map = document.getElementById("mapContainer");
  const url = `https://www.google.com/maps?q=${encodeURIComponent(place)}&output=embed`;
  map.innerHTML = `<iframe src="${url}"></iframe>`;
}
