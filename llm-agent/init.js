document.addEventListener("DOMContentLoaded", function () {
  // Variables for the different input modes
  let currentMode = "text"; // Can be "text", "audio", or "file"
  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;
  let recordingTimer;
  let recordingSeconds = 0;

  // Elements
  const chatBox = document.getElementById("chat-box");
  const sendBtn = document.getElementById("send-chat");
  const micToggle = document.getElementById("mic-toggle");
  const fileToggle = document.getElementById("file-toggle");
  const fileUploadArea = document.getElementById("file-upload-area");
  const recordingIndicator = document.getElementById("recording-indicator");
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const recordingStatus = document.getElementById("recording-status");
  const recordingTime = document.getElementById("recording-time");
  const liveAudio = document.getElementById("liveAudio");
  const player = document.getElementById("player");

  // Function to reset all input modes
  function resetInputModes() {
    // Reset text
    chatBox.value = "";

    // Reset audio
    if (isRecording) {
      stopRecording();
    }
    recordingIndicator.style.display = "none";

    // Reset file
    fileUploadArea.style.display = "none";
    fileInput.value = "";

    // Reset UI
    currentMode = "text";
    updateSendButtonText();
  }

  // Update send button text based on current input mode
  function updateSendButtonText() {
    switch (currentMode) {
      case "text":
        sendBtn.textContent = "Send";
        break;
      case "audio":
        sendBtn.textContent = "Send Audio";
        break;
      case "file":
        sendBtn.textContent = "Upload File";
        break;
    }
  }

  // Initialize text resize functionality
  function autoExpand(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
  }

  chatBox.addEventListener("input", function () {
    autoExpand(chatBox);

    // If user starts typing, switch to text mode
    if (chatBox.value.trim() !== "" && currentMode !== "text") {
      resetInputModes();
    }
  });

  // Mic toggle button
  micToggle.addEventListener("click", async (e) => {
    e.preventDefault();

    // If already in audio mode, stop recording
    if (currentMode === "audio" && isRecording) {
      stopRecording();
      return;
    }

    // Otherwise, switch to audio mode
    resetInputModes();
    currentMode = "audio";
    updateSendButtonText();

    try {
      // Start recording
      await startRecording();
      recordingIndicator.style.display = "block";
    } catch (error) {
      console.error("Error starting recording:", error);
      resetInputModes();
    }
  });

  // File toggle button
  fileToggle.addEventListener("click", (e) => {
    e.preventDefault();

    // Toggle file upload area
    if (currentMode === "file") {
      resetInputModes();
    } else {
      resetInputModes();
      currentMode = "file";
      updateSendButtonText();
      fileUploadArea.style.display = "block";
    }
  });

  // Recording functionality
  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    liveAudio.srcObject = stream;
    liveAudio.play();

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();

    audioChunks = [];
    isRecording = true;

    // Start recording timer
    recordingSeconds = 0;
    updateRecordingTime();
    recordingTimer = setInterval(updateRecordingTime, 1000);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      const audioUrl = URL.createObjectURL(audioBlob);
      player.src = audioUrl;

      // Keep the currentMode as audio but update recording status
      recordingStatus.textContent = "Recording complete";
      clearInterval(recordingTimer);
    };
  }

  function stopRecording() {
    if (!isRecording) return;

    mediaRecorder.stop();

    const tracks = liveAudio.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    liveAudio.srcObject = null;

    isRecording = false;
  }

  function updateRecordingTime() {
    recordingSeconds++;
    const minutes = Math.floor(recordingSeconds / 60)
      .toString()
      .padStart(2, "0");
    const seconds = (recordingSeconds % 60).toString().padStart(2, "0");
    recordingTime.textContent = `${minutes}:${seconds}`;
  }

  // File drag and drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.backgroundColor = "#444";
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.style.backgroundColor = "";
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.style.backgroundColor = "";

    const file = e.dataTransfer.files[0];
    if (!file) return;

    // Set the file input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
  });

  // Send button handling
  document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    switch (currentMode) {
      case "text":
        const prompt = window.api.sendFormData();
        if (prompt !== undefined) {
          ResponseText(prompt);
        }
        break;

      case "audio":
        if (audioChunks.length > 0) {
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          await window.api.sendAudio(audioBlob);

          // Display user audio message
          const node = document.createElement("div");
          const innerText = document.createElement("p");
          innerText.textContent = "Sent audio message";
          innerText.classList.add("user-text");
          node.classList.add("user-text-container");
          node.appendChild(innerText);
          document.getElementById("content").appendChild(node);

          if (document.getElementById("welcome")) {
            document.getElementById("welcome").remove();
          }
        }
        break;

      case "file":
        const file = fileInput.files[0];
        if (!file) return;

        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);

        await window.api.uploadFile({
          name: file.name,
          type: file.type,
          buffer: uint8Array,
        });

        // Display user file upload message
        const node = document.createElement("div");
        const innerText = document.createElement("p");
        innerText.textContent = `Uploaded file: ${file.name}`;
        innerText.classList.add("user-text");
        node.classList.add("user-text-container");
        node.appendChild(innerText);
        document.getElementById("content").appendChild(node);

        if (document.getElementById("welcome")) {
          document.getElementById("welcome").remove();
        }
        break;
    }

    // Reset after sending
    resetInputModes();
  });

  // Chat box focus/blur events
  chatBox.addEventListener("focus", () => {
    document.getElementById("dispose").style.flex = "0";
    document.getElementById("chat-form").style.maxWidth = "100%";
    chatBox.placeholder = "Type your prompt here...";
  });

  chatBox.addEventListener("focusout", () => {
    setTimeout(() => {
      if (document.getElementById("content").innerHTML == "") {
        document.getElementById("dispose").style.flex = "1";
        document.getElementById("chat-form").style.maxWidth = "40vw";
        if (currentMode === "text") {
          chatBox.placeholder = "Get started";
        }
      }
    }, 200);
  });

  // Initialize with text mode
  resetInputModes();
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
