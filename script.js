// 🎤 Voice Recognition
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

// 🔊 Text-to-Speech
function speak(text) {
  const speech = new SpeechSynthesisUtterance(text);
  speech.rate = 1;
  speech.pitch = 1;
  window.speechSynthesis.speak(speech);
}

async function send() {
  const input = document.getElementById("input");
  const chatBox = document.getElementById("chatBox");

  const message = input.value;
  if (!message) return;

  chatBox.innerHTML += `<div class="message user">${message}</div>`;
  input.value = "";

  // Typing indicator
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

  chatBox.innerHTML += `<div class="message bot">${reply}</div>`;

  // 🔊 Speak response (shortened)
  speak(reply.substring(0, 200));

  // 🏥 Detect hospitals and show cards + map
  if (reply.toLowerCase().includes("hospital") || reply.toLowerCase().includes("clinic")) {
    const hospitals = extractHospitals(reply);

    if (hospitals.length > 0) {
      chatBox.innerHTML += renderHospitalCards(hospitals);
      showMap(hospitals[0]); // show first hospital on map
    }
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

// Extract hospital names
function extractHospitals(text) {
  const lines = text.split(",");
  return lines.slice(0, 3).map(name => name.trim());
}

// Render cards
function renderHospitalCards(hospitals) {
  let html = `<div class="card-container">`;

  hospitals.forEach(h => {
    html += `
      <div class="card">
        <h3>${h}</h3>
        <p>Recommended healthcare provider</p>
        <div class="card-actions">
          <button onclick="showMap('${h}')">📍 View Map</button>
          <button onclick="alert('Booking simulated!')">Book</button>
        </div>
      </div>
    `;
  });

  html += `</div>`;
  return html;
}

// 🗺️ Show embedded map
function showMap(place) {
  const mapContainer = document.getElementById("mapContainer");

  const url = `https://www.google.com/maps?q=${encodeURIComponent(place)}&output=embed`;

  mapContainer.innerHTML = `<iframe src="${url}"></iframe>`;
}
