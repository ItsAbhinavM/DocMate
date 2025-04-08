// components/ChatForm.jsx
import { useState, useRef, useEffect } from 'react';

function ChatForm({ onSendMessage }) {
  const [message, setMessage] = useState('');
  const [expanded, setExpanded] = useState(false);
  const textareaRef = useRef(null);
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
      
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };
  
  const autoExpand = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };
  
  const handleFocus = () => {
    setExpanded(true);
  };
  
  const handleBlur = () => {
    if (!message.trim()) {
      setTimeout(() => setExpanded(false), 200);
    }
  };
  
  useEffect(() => {
    autoExpand();
  }, [message]);
  
  return (
    <form 
      className={`flex w-full justify-center ${expanded ? 'max-w-full' : 'max-w-lg'} my-5`}
      onSubmit={handleSubmit}
    >
      <div className="w-full max-w-lg border border-gray-500 rounded-3xl flex items-center mr-2">
        <textarea
          ref={textareaRef}
          id="chat-box"
          className="w-full rounded-3xl bg-gray-800 text-white outline-none px-4 py-3 min-h-12 overflow-hidden resize-none"
          placeholder={expanded ? "Type your prompt here..." : "Get started"}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
      </div>
      <div className="flex items-end">
        <button
          type="submit"
          className="bg-amber-100 hover:bg-amber-200 rounded-full min-w-24 h-12 border-none"
        >
          Send
        </button>
      </div>
    </form>
  );
}

export default ChatForm;


