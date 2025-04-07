// const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
// console.log("here")
// if (!SpeechRecognition) {
//   alert("Web Speech API is not supported by this browser.");
// } else {
//   const recognition = new SpeechRecognition();
//   recognition.continuous = true;
//   recognition.interimResults = false;
//   recognition.lang = 'en-US';

//   recognition.start();

//   recognition.onstart = () => {
//     console.log('Speech recognition service has started');
//   };

//   recognition.onresult = (event) => {
//     const last = event.results.length - 1;
//     const command = event.results[last][0].transcript.trim().toLowerCase();
//     console.log('Voice Command:', command);

//     if (command.includes('activate app')) {
//       window.electronAPI.activateApp();
//     }
//   };

//   recognition.onspeechend = () => {
//     console.log('Speech recognition service disconnected');
//     recognition.start();
//   };

//   recognition.onerror = (event) => {
//     console.error('Speech Recognition Error:', event.error);
//     if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
//       alert("Microphone access was denied. Please enable it in your browser settings.");
//     }
//   };
// }

// const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
// const SpeechGrammarList = window.SpeechGrammarList || window.webkitSpeechGrammarList;
// const SpeechRecognitionEvent = window.SpeechRecognitionEvent || window.webkitSpeechRecognitionEvent;

// const colors = [
//   "aqua", "azure", "beige", "bisque", "black", "blue", "brown", "chocolate", "coral",
// ];
// const grammar = `#JSGF V1.0; grammar colors; public <color> = ${colors.join(" | ")};`;

// const recognition = new SpeechRecognition();
// const speechRecognitionList = new SpeechGrammarList();
// speechRecognitionList.addFromString(grammar, 1);

// recognition.grammars = speechRecognitionList;
// recognition.continuous = false;
// recognition.lang = "en-US";
// recognition.interimResults = false;
// recognition.maxAlternatives = 1;

// const diagnostic = document.querySelector('.output');
// const bg = document.getElementById('voice');
// //const hints = document.querySelector('.hints');

// // let colorHTML = '';
// // colors.forEach((color) => {
// //   colorHTML += `<span style="background-color:${color};"> ${color} </span>`;
// // });
// // hints.innerHTML = `Tap or click then say a color to change the background color of the app. Try ${colorHTML}.`;

// document.getElementById('voice').onclick = () => {
//   recognition.start();
//   console.log('Ready to receive a color command.');
// };

// recognition.onresult = (event) => {
//   const color = event.results[0][0].transcript;
//   diagnostic.textContent = `Result received: ${color}.`;
//   bg.style.backgroundColor = color;
//   console.log(`Confidence: ${event.results[0][0].confidence}`);
// };

// recognition.onspeechend = () => {
//   recognition.stop();
// };

// recognition.onnomatch = (event) => {
//   diagnostic.textContent = "I didn't recognize that color.";
// };

// recognition.onerror = (event) => {
//   diagnostic.textContent = `Error occurred in recognition: ${event.error}`;
// };

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  const voiceButton = document.getElementById('voice');

  // Check if Web Speech API is available
  if (!window.SpeechRecognition) {
    console.log('Web Speech API is not supported by this browser.');
    return;
  }

  // Initialize SpeechRecognition
  const recognition = new window.SpeechRecognition();
  recognition.continuous = true; // Keep recognizing even if the user pauses while speaking
  recognition.interimResults = true; // Get hypothesis while waiting for the final results
  recognition.lang = 'en-US'; // Set the language model

  // Start speech recognition when the button is clicked
  voiceButton.onclick = () => {
    recognition.start();
    console.log('Speech recognition started...');
  };

  // Handle the result of the speech recognition
  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map(result => result[0])
      .map(result => result.transcript)
      .join('');

    console.log(`Recognized text: ${transcript}`);
    // Here you can update your UI with the recognized text
  };

  // Handle errors
  recognition.onerror = (event) => {
    console.error('Speech recognition error detected: ', event.error);
  };
});

let mediaRecorder;
let recordedChunks = [];
let audioStream;

const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const sendBtn = document.getElementById('sendBtn');
const liveAudio = document.getElementById('liveAudio');

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioStream = stream; // Assign the stream
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };

    mediaRecorder.start();
    console.log('Recording started');
    liveAudio.srcObject = stream; // Play the audio stream in real-time
  } catch (err) {
    console.error('Error starting recording:', err);
  }
}

function stopRecording() {
  mediaRecorder.stop();
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  recordedChunks = []; // Clear the chunks array for the next recording
  
  console.log('Recording stopped');
  
  // Create an audio element and append it to the body
  const player = document.createElement('audio');
  player.controls = true;
  document.body.appendChild(player);
  
  // Set the Blob as the source of the audio element
  player.src = URL.createObjectURL(blob);
  
  // Optionally, play the audio automatically
  player.play();
  
  // Clean up the Blob URL when the audio finishes playing
  player.onended = () => {
    URL.revokeObjectURL(player.src);
  };
}

recordBtn.onclick = () => {
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  startRecording();
};

stopBtn.onclick = () => {
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  sendBtn.disabled = false;
  stopRecording();
};

sendBtn.onclick = () => {
  sendAudioToAPI();
};

function sendAudioToAPI() {
    const formData = new FormData();
    formData.append('file', new Blob(recordedChunks, { type: 'audio/webm' }), 'audio.webm');
  
    fetch('YOUR_API_ENDPOINT', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      console.log('Audio sent successfully:', data);
    })
    .catch(error => {
      console.error('Error sending audio:', error);
    });
  }
  

