

// components/FileUpload.jsx
import { useState, useRef } from 'react';

function FileUpload() {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (!file) return;
    
    // Optional: Update the file input to show the selected file
    if (fileInputRef.current) {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInputRef.current.files = dataTransfer.files;
    }
    
    await uploadFile(file);
  };
  
  const handleFileInputChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    await uploadFile(file);
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const file = fileInputRef.current?.files[0];
    if (!file) return;
    
    await uploadFile(file);
    e.target.reset();
  };
  
  const uploadFile = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/file_upload', {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.text();
      console.log('Upload result:', result);
      
      // You might want to handle the response here, e.g., display a success message
    } catch (error) {
      console.error('Error uploading file:', error);
      // Handle error, e.g., display an error message
    }
  };
  
  return (
    <div className="flex flex-col items-center w-full max-w-lg mb-8">
      <div
        className={`border-2 border-dashed p-5 w-full text-center rounded-lg mb-4 ${
          isDragging ? 'bg-gray-700 border-blue-400' : 'border-gray-500'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p className="text-white">Drag and drop your file here</p>
      </div>
      
      <form
        className="flex items-center space-x-4 w-full"
        onSubmit={handleSubmit}
      >
        <input
          type="file"
          id="file-input"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          className="text-white bg-gray-800 p-2 rounded flex-grow"
          required
        />
        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
        >
          Upload
        </button>
      </form>
    </div>
  );
}

export default FileUpload;