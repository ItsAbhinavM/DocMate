function ChatHistory({ messages }) {
    return (
      <div className="flex flex-col space-y-4 p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.isUser ? 'justify-end' : 'items-start'}`}
          >
            {!message.isUser && (
              <img
                src="/Type 1.gif"
                alt="Bot"
                className="w-8 h-8 mr-2 self-start mt-1"
              />
            )}
  
            <div
              className={`p-3 rounded-lg max-w-3xl overflow-x-auto ${
                message.isUser
                  ? 'bg-blue-600 text-white self-end'
                  : 'bg-gray-700 text-white'
              }`}
            >
              {message.component ? (
                message.component
              ) : (
                <p className="whitespace-pre-wrap">{message.text}</p>
              )}
              {message.imageUrl && (
                <img
                  src={message.imageUrl}
                  alt="Response image"
                  className="mt-2 max-w-full rounded"
                />
              )}
            </div>
          </div>
        ))}
      </div>
    );
  }
  
  export default ChatHistory;
  