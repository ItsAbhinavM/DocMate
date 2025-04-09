import { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';
import AudioRecorder from './AudioRecorder';

function ChatForm({ onSendMessage }) {
  const [message, setMessage] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const textareaRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (selectedFiles.length > 0) {
      setUploading(true);

      try {
        // Initialize progress tracking for each file
        const initialProgress = {};
        selectedFiles.forEach((file) => {
          initialProgress[file.name] = 0;
        });
        setUploadProgress(initialProgress);

        // Create an array of upload promises
        const uploadPromises = selectedFiles.map(async (file) => {
          const formData = new FormData();
          formData.append('file', file);

          const response = await fetch('http://localhost:8000/file_upload', {
            method: 'POST',
            body: formData,
          });

          const result = await response.text();
          console.log(`Upload result for ${file.name}:`, result);

          // Update progress for this file
          setUploadProgress(prev => ({
            ...prev,
            [file.name]: 100
          }));

          return result;
        });

        // Wait for all uploads to complete
        await Promise.all(uploadPromises);

        // All files uploaded successfully
        setSelectedFiles([]);
        setUploadProgress({});
      } catch (err) {
        console.error('Upload failed', err);
      } finally {
        setUploading(false);
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

  const removeFile = (indexToRemove) => {
    setSelectedFiles(selectedFiles.filter((_, index) => index !== indexToRemove));
  };

  useEffect(() => {
    autoExpand();
  }, [message]);

  // Calculate overall progress
  const calculateOverallProgress = () => {
    if (selectedFiles.length === 0) return 0;

    const totalProgress = Object.values(uploadProgress).reduce((sum, progress) => sum + progress, 0);
    return Math.round(totalProgress / selectedFiles.length);
  };

  return (
    <form
      className={`flex w-full justify-center ${expanded ? 'max-w-full' : 'max-w-lg'} my-5`}
      onSubmit={handleSubmit}
    >
      <div className="w-full max-w-lg border border-gray-500 rounded-3xl flex flex-col items-center mr-2 relative bg-[#264c71]">
        {selectedFiles.length > 0 ? (
          <div className="w-full bg-gray-800 text-white rounded-t-3xl px-4 py-3">
            <div className="max-h-32 overflow-y-auto">
              {selectedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between mb-2">
                  <span className="truncate">{file.name}</span>
                  {uploading ? (
                    <div className="text-xs text-blue-400">
                      {uploadProgress[file.name] === 100 ? 'Complete' : 'Uploading...'}
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => removeFile(index)}
                      className="text-red-400 hover:text-red-600 ml-2"
                    >
                      ✕
                    </button>
                  )}
                </div>
              ))}
            </div>

            {uploading && (
              <div className="w-full mt-2">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${calculateOverallProgress()}%` }}
                  ></div>
                </div>
                <p className="text-xs text-center mt-1">
                  Uploading: {calculateOverallProgress()}%
                </p>
              </div>
            )}
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
        <div className="self-end flex">
          <AudioRecorder onTranscription={onSendMessage} />
          <FileUpload setSelectedFiles={setSelectedFiles} />
        </div>
      </div>
      <div className="flex items-end">
        <button
          type="submit"
          className="bg-[#264c71] text-white hover:bg-amber-200 rounded-full min-w-24 h-12 border-none"
          disabled={uploading}
        >
          {selectedFiles.length > 0 ? (uploading ? 'Uploading...' : 'Upload Files') : 'Send'}
        </button>
      </div>
    </form>
  );
}

export default ChatForm;
