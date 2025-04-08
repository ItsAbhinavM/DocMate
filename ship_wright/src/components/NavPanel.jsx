import { PlusCircle } from 'lucide-react';

function NavPanel({ onNewChat }) {
  return (
    <div className="bg-gray-800 w-64 h-screen">
      <div className="p-4 text-white">
        <h2 className="text-xl font-bold mb-4">Navigation</h2>

        {/* New Chat button */}
        <button
          onClick={onNewChat}
          className="flex items-center w-full bg-gray-700 hover:bg-gray-600 p-3 rounded-lg mb-4 transition-colors"
        >
          <PlusCircle size={20} className="mr-2" />
          <span>New Chat</span>
        </button>

        <ul className="space-y-2">
          <li className="hover:bg-gray-700 p-2 rounded cursor-pointer">Home</li>
          <li className="hover:bg-gray-700 p-2 rounded cursor-pointer">History</li>
          <li className="hover:bg-gray-700 p-2 rounded cursor-pointer">Settings</li>
        </ul>
      </div>
    </div>
  );
}

export default NavPanel;
