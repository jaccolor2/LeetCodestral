import { Navbar } from './Navbar';
import { PanelGroup } from 'react-resizable-panels';

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
              {isLeftPanelVisible && leftPanel}
              {centerPanel}
              {rightPanel}
            </PanelGroup>
          </div>
        </div>
      </main>
    </>
  );
} 