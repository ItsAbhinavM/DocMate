document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  window.api.sendMessage();
});

let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const sendBtn = document.getElementById("sendBtn");
const liveAudio = document.getElementById("liveAudio");
const player = document.getElementById("player");

recordBtn.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  liveAudio.srcObject = stream;
  liveAudio.play();

  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.start();

  audioChunks = [];

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    const audioUrl = URL.createObjectURL(audioBlob);
    player.src = audioUrl;

    // Optionally send to main process
    window.electronAPI?.sendAudio(audioBlob); // Only if you're using contextBridge for IPC

    sendBtn.disabled = false;
  };

  recordBtn.disabled = true;
  stopBtn.disabled = false;
});

stopBtn.addEventListener("click", () => {
  mediaRecorder.stop();

  const tracks = liveAudio.srcObject.getTracks();
  tracks.forEach((track) => track.stop());
  liveAudio.srcObject = null;

  recordBtn.disabled = false;
  stopBtn.disabled = true;
});

sendBtn.addEventListener("click", () => {
  // Example: download the audio or send to backend
  const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(audioBlob);
  a.download = "recording.webm";
  a.click();
});

document.getElementById("nav-btn").addEventListener("click", async (e) => {
  let w = document.getElementById("nav").style.display;
  console.log(w);
  if (w != "none" || w == "") {
    document.getElementById("nav").style.display = "none";
  } else {
    document.getElementById("nav").style.display = "flex";
  }
});

let history = {};

document.getElementById("send-chat").addEventListener("click", async (e) => {
  e.preventDefault();
  const prompt = window.api.sendFormData();
  console.log(prompt);
  if (prompt != undefined) {
    ResponseText(prompt);
  }
});

// document file sending
document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("file-input");
  const file = fileInput.files[0];

  if (!file) return;

  const arrayBuffer = await file.arrayBuffer();
  const uint8Array = new Uint8Array(arrayBuffer);

  window.api.uploadFile({
    name: file.name,
    type: file.type,
    buffer: uint8Array,
  });
  console.log("Upload triggered:", file.name);
  document.getElementById("upload-form").reset();
});

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

// Highlight the drop zone on drag
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.backgroundColor = "#f0f0f0";
});

// Remove highlight when drag leaves
dropZone.addEventListener("dragleave", () => {
  dropZone.style.backgroundColor = "";
});

// Handle file drop
dropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropZone.style.backgroundColor = "";

  const file = e.dataTransfer.files[0];
  if (!file) return;

  // Optional: also show it in the file input
  fileInput.files = e.dataTransfer.files;

  // const arrayBuffer = await file.arrayBuffer();
  // const uint8Array = new Uint8Array(arrayBuffer);

  // await window.api.uploadFile({
  //   name: file.name,
  //   type: file.type,
  //   buffer: uint8Array,
  // });

  // console.log("Upload triggered via drag:", file.name);

  // // Reset form
  // document.getElementById("upload-form").reset();
});

function ResponseText(prompt) {
  window.api.getData(prompt);
}
document.addEventListener("DOMContentLoaded", function () {
  const chatBox = document.getElementById("chat-box");
  chatBox.addEventListener("input", function () {
    autoExpand(chatBox);
  });
});

function autoExpand(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
}

document.getElementById("chat-box").addEventListener("focus", async (e) => {
  document.getElementById("dispose").style.flex = "0";
  document.getElementById("chat-form").style.maxWidth = "100%";
  document.getElementById("chat-box").placeholder = "Type your prompt here...";
});

document.getElementById("chat-box").addEventListener("focusout", async (e) => {
  setTimeout(() => {
    console.log(document.getElementById("content").innerHTML);
    if (document.getElementById("content").innerHTML == "") {
      document.getElementById("dispose").style.flex = "1";
      document.getElementById("chat-form").style.maxWidth = "40vw";
      window.api.checkFormData();
    }
  }, 200);
});
// document.addEventListener('DOMContentLoaded', () => {
//     const form = document.getElementById('chat-form');
//     const inputField = document.getElementById('chat-box');

//     inputField.addEventListener('keydown', (event) => {
//         if (event.key === 'Enter') {
//             event.preventDefault();  // Prevent the default action (form submission) on Enter key press
//             form.submit();           // Manually submit the form
//         }
//     });
// });
// document.getElementById('chat-form').addEventListener('submit', async(e)=>{
// 	setTimeout(()=>{
// 		e.preventDefault();
// 		window.api.sendFormData();
// 	})
// })
