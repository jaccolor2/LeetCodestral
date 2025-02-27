import { useRouter } from 'next/navigation';
import { useAuth } from '../../hooks/useAuth';
import { MistralLogo } from '../MistralLogo';

export function Navbar() {
  const router = useRouter();
  const { isLoggedIn, setIsLoggedIn } = useAuth();

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    setIsLoggedIn(false);
    router.push('/login');
  };

  return (
    <nav className="bg-[#1A1A1A] p-4 border-b border-[#2D2D2D]">
      <div className="container mx-auto flex justify-between items-center relative">
        <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-2">
          <MistralLogo size={24} variant="navbar" />
          <span className="text-xl font-bold text-white">LeetCodestral</span>
        </div>
        
        {isLoggedIn && (
          <div className="ml-auto">
            <button
              onClick={handleLogout}
              className="bg-[#FF4405] text-white px-4 py-2 rounded hover:opacity-90"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </nav>
  );
} 