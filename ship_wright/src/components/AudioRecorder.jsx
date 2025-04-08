// components/AudioRecorder.jsx
import { useState, useRef } from 'react';

function AudioRecorder({ onTranscription }) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState('');
  const [audioBlob, setAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const liveAudioRef = useRef(null);
  
  const startRecording = async (e) => {
    e.preventDefault();
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      if (liveAudioRef.current) {
        liveAudioRef.current.srcObject = stream;
        liveAudioRef.current.play();
      }
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioURL(audioUrl);
        setAudioBlob(audioBlob);
        
        // For Tauri, we would use a different approach
        // This is a placeholder for the audio transcription logic
        sendAudioForTranscription(audioBlob);
      };
      
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };
  
  const stopRecording = (e) => {
    e.preventDefault();
    
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      
      if (liveAudioRef.current && liveAudioRef.current.srcObject) {
        liveAudioRef.current.srcObject.getTracks().forEach(track => track.stop());
        liveAudioRef.current.srcObject = null;
      }
      
      setIsRecording(false);
    }
  };
  
  const sendAudioForTranscription = async (blob) => {
    // In a real Tauri app, you might use the Tauri API to handle this
    // For this example, we'll simulate the response
    try {
      // Convert blob to buffer
      const buffer = await blob.arrayBuffer();
      
      // In real app, you'd send this to backend or use Tauri API
      // For demo purposes, let's assume we get back a transcription
      console.log("Audio sent for transcription");
      
      // Simulating a backend response
      setTimeout(() => {
        // This would be replaced with actual transcription from your backend
        const transcribedText = "This is a simulated transcription of your audio.";
        onTranscription(transcribedText);
      }, 1000);
    } catch (error) {
      console.error('Error sending audio for transcription:', error);
    }
  };
  
  const downloadAudio = (e) => {
    e.preventDefault();
    
    if (audioBlob) {
      const a = document.createElement('a');
      a.href = audioURL;
      a.download = 'recording.webm';
      a.click();
    }
  };
  
  return (
    <div className="flex flex-col items-center mb-4">
      <div className="flex space-x-2 mb-2">
        <button
          onClick={startRecording}
          disabled={isRecording}
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Start Recording
        </button>
        <button
          onClick={stopRecording}
          disabled={!isRecording}
          className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Stop Recording
        </button>
        <button
          onClick={downloadAudio}
          disabled={!audioURL}
          className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Download Audio
        </button>
      </div>
      
      <div className="audio-players flex space-x-4">
        <audio ref={liveAudioRef} controls className="hidden" />
        {audioURL && (
          <audio src={audioURL} controls className="w-full max-w-md" />
        )}
      </div>
    </div>
  );
}

export default AudioRecorder;