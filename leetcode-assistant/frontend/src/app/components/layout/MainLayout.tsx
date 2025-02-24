import { Navbar } from '../Navbar';
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
  onToggleLeftPanel
}: MainLayoutProps) {
  return (
    <>
      <Navbar />
      <main className="fixed inset-0 mt-14 bg-gray-900 h-[calc(100vh-56px)]">
        <div className="h-full">
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
      </main>
    </>
  );
} 