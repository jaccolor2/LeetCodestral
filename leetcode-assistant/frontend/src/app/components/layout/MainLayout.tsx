import { Navbar } from './Navbar';
import { useAuth } from '../../hooks/useAuth';
import { Panel, PanelGroup } from 'react-resizable-panels';

interface MainLayoutProps {
  leftPanel: React.ReactNode;
  centerPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  isLeftPanelVisible: boolean;
  onToggleLeftPanel: () => void;
  children?: React.ReactNode;
}

export function MainLayout({
  leftPanel,
  centerPanel,
  rightPanel,
  isLeftPanelVisible,
  onToggleLeftPanel,
  children
}: MainLayoutProps) {
  const { isLoggedIn, isLoading } = useAuth();

  if (isLoading) {
    return <div className="min-h-screen bg-[var(--background)] flex items-center justify-center">
      <div className="text-[var(--foreground)]">Loading...</div>
    </div>;
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[var(--background)]">
      <div className="shadow-[0_4px_20px_rgba(0,0,0,0.4)] z-10">
        <Navbar />
      </div>
      {isLoggedIn ? (
        <main className="flex-1 relative">
          <div className="absolute inset-0">
            <PanelGroup direction="horizontal" className="h-full">
              {isLeftPanelVisible && (
                <Panel 
                  defaultSize={20} 
                  minSize={20} 
                  maxSize={20}
                  className="bg-[var(--secondary)]"
                >
                  <div className="h-full overflow-auto p-4">
                    <div className="h-full rounded-lg overflow-hidden shadow-[0_4px_20px_rgba(0,0,0,0.4)] border border-white/10">
                      {leftPanel}
                    </div>
                  </div>
                </Panel>
              )}
              
              <Panel 
                defaultSize={50} 
                minSize={50} 
                maxSize={50}
                className="bg-[var(--secondary)]"
              >
                <div className="h-full overflow-auto p-4">
                  <div className="h-full rounded-lg overflow-hidden shadow-[0_4px_20px_rgba(0,0,0,0.4)] border border-gray-800">
                    {centerPanel}
                  </div>
                </div>
              </Panel>
              
              <Panel 
                defaultSize={30} 
                minSize={30} 
                maxSize={30}
                className="bg-[var(--secondary)]"
              >
                <div className="h-full overflow-auto p-4">
                  <div className="h-full rounded-lg overflow-hidden shadow-[0_4px_20px_rgba(0,0,0,0.4)] border border-gray-800">
                    {rightPanel}
                  </div>
                </div>
              </Panel>
            </PanelGroup>
          </div>
          {children}
        </main>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-[var(--foreground)] text-center">
            Please login to access the LeetCode Assistant
          </div>
        </div>
      )}
    </div>
  );
} 