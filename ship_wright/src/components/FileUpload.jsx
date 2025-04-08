import { useState, useRef } from 'react';
import Clip from './ui/clip';

function FileUpload({ setSelectedFiles }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const [onReveal, setOnReveal] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) setSelectedFiles(files);
  };

  const handleFileInputChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) setSelectedFiles(files);
  };

  return (
    <div className="relative h-full flex items-end">
      <button
        type="button"
        onClick={() => setOnReveal(!onReveal)}
        className="p-2"
      >
        <Clip />
      </button>
      <div className={`${onReveal ? "bottom-[5vw] -ml-[20vw] absolute bg-[#1c1c1c] p-[1vw] rounded-[1vw]" : "hidden"}`}>
        <div
          className={`border-2 border-dashed p-5 text-center rounded-lg mb-4 ${isDragging ? 'bg-gray-700 border-blue-400' : 'border-gray-500'
            }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <p className="text-white size-[20vw] flex justify-center items-center">Drag and drop your files here</p>
        </div>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          className="text-white bg-gray-800 p-2 rounded w-full"
          multiple
        />
      </div>
    </div>
  );
}

export default FileUpload;
