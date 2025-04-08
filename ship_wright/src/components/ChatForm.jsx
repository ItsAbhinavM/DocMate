import { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';

function ChatForm({ onSendMessage }) {
  const [message, setMessage] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const textareaRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch('http://localhost:8000/file_upload', {
          method: 'POST',
          body: formData,
        });
        const result = await response.text();
        console.log('Upload result:', result);
        setSelectedFile(null); // Clear after upload
      } catch (err) {
        console.error('Upload failed', err);
      }

    } else if (message.trim()) {
      onSendMessage(message);
      setMessage('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    }
  };

  const autoExpand = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  const handleFocus = () => setExpanded(true);

  const handleBlur = () => {
    if (!message.trim()) setTimeout(() => setExpanded(false), 200);
  };

  useEffect(() => {
    autoExpand();
  }, [message]);

  return (
    <form
      className={`flex w-full justify-center ${expanded ? 'max-w-full' : 'max-w-lg'} my-5`}
      onSubmit={handleSubmit}
    >
      <div className="w-full max-w-lg border border-gray-500 rounded-3xl flex items-center mr-2 relative">
        {selectedFile ? (
          <div className="flex w-full items-center justify-between bg-gray-800 text-white rounded-3xl px-4 py-3">
            <span className="truncate">{selectedFile.name}</span>
            <button
              type="button"
              onClick={() => setSelectedFile(null)}
              className="text-red-400 hover:text-red-600 ml-2"
            >
              ✕
            </button>
          </div>
        ) : (
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
        )}

        <FileUpload setSelectedFile={setSelectedFile} />
      </div>
      <div className="flex items-end">
        <button
          type="submit"
          className="bg-amber-100 hover:bg-amber-200 rounded-full min-w-24 h-12 border-none"
        >
          {selectedFile ? 'Upload File' : 'Send'}
        </button>
      </div>
    </form>
  );
}

export default ChatForm;
