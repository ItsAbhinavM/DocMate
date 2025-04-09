import { useState, useRef, useEffect } from "react";
// import { invoke } from '@tauri-apps/api/tauri';
import Logo from "./assets/type_1.gif";
import ChatForm from "./components/ChatForm";
import AudioRecorder from "./components/AudioRecorder";
import ChatHistory from "./components/ChatHistory";
import NavPanel from "./components/NavPanel";
import "./App.css";
import axios from "axios";
import JSONView from "./components/json_view";

function App() {
  const [messages, setMessages] = useState([]);
  const [showNav, setShowNav] = useState(true);
  const contentRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [runId, setRunId] = useState();

  const addMessage = (text, isUser, image = null) => {
    setMessages((prev) => [...prev, { text, isUser, image }]);
  };

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;
    setIsLoading(true);

    addMessage(text, true);

    let payload = {};
    if (runId) {
      payload = {
        original_query: text,
        run_id: runId,
      };
    } else {
      payload = {
        original_query: text,
      };
    }
    console.log(payload, "here is payload");

    try {
      const response = await axios.post('http://localhost:8000/send_prompt',
        payload,
        {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        });

      const data = response.data;
      console.log("here is the data btw", data);

      setIsLoading(false);

      if (data.status === "waiting_clarification") {
        addMessage(data.message, false);
      } else {
        // Check if there's an image path in the response
        const imagePath = data.image_path || null;

        if (data.message) {
          // Add message with image if available
          addMessage(data.message, false, "http://127.0.0.1:8000/" + imagePath);
        }

        console.log(data["final_dataset"], "this is the json to be viewed");

        if (data["final_dataset"]) {
          addMessage(<JSONView jsoner={{ 0: data["final_dataset"] }} />, false);
        }
      }

      console.log("This is the run id", data["run_id"]);
      setRunId(data["run_id"]);
    } catch (error) {
      console.error("Error sending message:", error);
      addMessage("Sorry, there was an error processing your request.", false);
    }
  };

  const toggleNav = () => {
    setShowNav(!showNav);
  };

  // New chat function to reset runId and clear messages
  const handleNewChat = () => {
    setRunId(undefined);
    setMessages([]);
    console.log("New chat started: runId reset and messages cleared");
  };

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex w-screen min-h-screen bg-opacity-80 bg-black font-sans overflow-x-hidden">
      {showNav && <NavPanel onNewChat={handleNewChat} />}

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
                Your Personal
                <br />
                LLM
              </h1>
            )}
          </div>

          <div
            className="flex-1 w-full max-w-4xl px-4 overflow-y-auto"
            ref={contentRef}
          >
            <ChatHistory messages={messages} />
            <div
              className={`flex space-x-1 ${isLoading ? "ml-[5vw]" : "hidden"}`}
            >
              <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-150" />
              <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-300" />
            </div>
          </div>

          <div className="w-full flex flex-col items-center mb-4">
            <ChatForm onSendMessage={handleSendMessage} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
