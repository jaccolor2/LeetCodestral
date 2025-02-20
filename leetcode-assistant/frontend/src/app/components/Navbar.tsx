'use client';

export const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 h-14 bg-gray-800 border-b border-gray-700 z-50">
      <div className="h-full px-4 flex items-center justify-between">
        <div className="text-white font-bold text-lg">LeetCode Assistant</div>
        <div className="flex gap-4">
          <button className="text-gray-300 hover:text-white transition-colors">
            Problems
          </button>
          <button className="text-gray-300 hover:text-white transition-colors">
            History
          </button>
        </div>
      </div>
    </nav>
  );
}; 