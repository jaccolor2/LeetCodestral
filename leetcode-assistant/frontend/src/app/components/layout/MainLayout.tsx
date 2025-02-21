import { Navbar } from '../Navbar';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

interface MainLayoutProps {
  leftPanel: React.ReactNode;
  centerPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  isLeftPanelVisible: boolean;
  onToggleLeftPanel: () => void;
}

export function MainLayout({
  leftPanel,
  centerPanel,
  rightPanel,
  isLeftPanelVisible,
  onToggleLeftPanel
}: MainLayoutProps) {
  return (
    <>
      <Navbar />
      <main className="fixed inset-0 mt-14 bg-gray-900 h-[calc(100vh-56px)]">
        <div className="p-4 h-full flex flex-col">
          <div className="flex-1 relative">
            <PanelGroup direction="horizontal" className="h-full">
              {isLeftPanelVisible && (
                <>
                  <Panel 
                    defaultSize={20} 
                    minSize={15} 
                    maxSize={30}
                    className="bg-gray-800 rounded-lg overflow-hidden"
                  >
                    <div className="h-full p-4 overflow-y-auto">
                      {leftPanel}
                    </div>
                  </Panel>
                  <PanelResizeHandle className="w-2 bg-gray-700 hover:bg-gray-600 transition-colors" />
                </>
              )}
              
              <Panel 
                defaultSize={isLeftPanelVisible ? 50 : 70} 
                minSize={30} 
                className="bg-gray-800 rounded-lg overflow-hidden mx-2"
              >
                <div className="h-full p-4">
                  {centerPanel}
                </div>
              </Panel>
              
              <PanelResizeHandle className="w-2 bg-gray-700 hover:bg-gray-600 transition-colors" />
              
              <Panel 
                defaultSize={30} 
                minSize={20} 
                maxSize={40}
                className="bg-gray-800 rounded-lg overflow-hidden"
              >
                <div className="h-full p-4 overflow-y-auto">
                  {rightPanel}
                </div>
              </Panel>
            </PanelGroup>
          </div>
        </div>
      </main>
    </>
  );
} 