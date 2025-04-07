let mediaRecorder;
let recordedChunks = [];

const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const sendBtn = document.getElementById('sendBtn');

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };

    mediaRecorder.start();
    console.log('Recording started');
  } catch (err) {
    console.error('Error starting recording:', err);
  }
}

function stopRecording() {
  mediaRecorder.stop();
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  recordedChunks = []; // Clear the chunks array for the next recording
  console.log('Recording stopped');
}

recordBtn.onclick = () => {
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  startRecording();
};

stopBtn.onclick = () => {
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  sendBtn.disabled = false; // Enable send button once recording stops
  stopRecording();
};

sendBtn.onclick = () => {
  // Send the recorded audio blob to your API
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
  
