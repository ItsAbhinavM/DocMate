// App.jsx - Main component
import { useState, useRef, useEffect } from 'react';
// import { invoke } from '@tauri-apps/api/tauri';
import Logo from "./assets/type_1.gif";
import ChatForm from './components/ChatForm';
import AudioRecorder from './components/AudioRecorder';
import ChatHistory from './components/ChatHistory';
import NavPanel from './components/NavPanel';
import './App.css'
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [showNav, setShowNav] = useState(true);
  const contentRef = useRef(null);

  const addMessage = (text, isUser) => {
    setMessages(prev => [...prev, { text, isUser }]);
  };

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    addMessage(text, true);

    try {
      const response = await axios.post('http://localhost:8000/send_prompt', {
        original_query: text
      }, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });

      const data = response.data;
      addMessage(data.message, false, data.url);
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('Sorry, there was an error processing your request.', false);
    }
  };

  const toggleNav = () => {
    setShowNav(!showNav);
  };

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex w-screen min-h-screen bg-opacity-70 bg-black font-sans overflow-x-hidden">
      {showNav && <NavPanel />}
      
      <div className="flex-1 flex flex-col items-center h-screen">
        {/* Header with logo */}
        <div className="font-light text-white text-center p-5 flex items-center">
          <img src={Logo} alt="Orcha Logo" className="w-24 h-14" />
          <p className="ml-2">Orchestrater</p>
        </div>

        {/* Toggle Navigation Button */}
        <button 
          className="absolute z-10 left-0 top-0 bg-gray-700 text-white p-2"
          onClick={toggleNav}
        >
          X
        </button>

        {/* Main Content Area */}
        <div className="flex-1 w-full flex justify-center items-center flex-col flex-grow overflow-auto">
          <div className="flex flex-col justify-center items-center overflow-y-auto">
            {messages.length === 0 && (
              <h1 className="text-white text-center font-bold tracking-tight mt-40 text-4xl">
                Your Personal<br />LLM
              </h1>
            )}
          </div>
          
          <div className="flex-1 w-full max-w-4xl px-4 overflow-y-auto" ref={contentRef}>
            <ChatHistory messages={messages} />
          </div>
          
          <div className="w-full flex flex-col items-center mb-4">
            <AudioRecorder onTranscription={handleSendMessage} />
            <ChatForm onSendMessage={handleSendMessage} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;