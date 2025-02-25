import { Navbar } from './Navbar';
import { useAuth } from '../../hooks/useAuth';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

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
    return <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="text-white">Loading...</div>
    </div>;
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gray-900">
      <Navbar />
      {isLoggedIn ? (
        <main className="flex-1 relative">
          <div className="absolute inset-0">
            <PanelGroup direction="horizontal" className="h-full">
              {isLeftPanelVisible && (
                <>
                  <Panel 
                    defaultSize={20} 
                    minSize={20} 
                    maxSize={20}
                    className="bg-gray-800"
                  >
                    <div className="h-full overflow-auto p-4">
                      {leftPanel}
                    </div>
                  </Panel>
                  <PanelResizeHandle className="w-2 bg-gray-700 hover:bg-gray-600 transition-colors" />
                </>
              )}
              
              <Panel 
                defaultSize={50} 
                minSize={50} 
                maxSize={50}
                className="bg-gray-800"
              >
                <div className="h-full overflow-auto p-4">
                  {centerPanel}
                </div>
              </Panel>
              
              <PanelResizeHandle className="w-2 bg-gray-700 hover:bg-gray-600 transition-colors" />
              
              <Panel 
                defaultSize={30} 
                minSize={30} 
                maxSize={30}
                className="bg-gray-800"
              >
                <div className="h-full overflow-auto p-4">
                  {rightPanel}
                </div>
              </Panel>
            </PanelGroup>
          </div>
          {children}
        </main>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-white text-center">
            Please login to access the LeetCode Assistant
          </div>
        </div>
      )}
    </div>
  );
} 