import React from 'react';

function ChatHistory({ messages }) {
  return (
    <div className="flex flex-col space-y-4 py-4">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-3/4 rounded-xl p-4 ${message.isUser
                ? 'bg-gray-600 text-white'
                : 'bg-gray-800 text-white'
              }`}
          >
            {/* Display message text */}
            <div className="mb-2">{message.text}</div>

            {/* Display image if present */}
            {message.image && (
              <div className="mt-3">
                <img
                  src={message.image}
                  alt="Response image"
                  className="max-w-full rounded-lg"
                  style={{ maxHeight: '300px' }}
                />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ChatHistory;
