import React from 'react';

/**
 * Layout component using Tailwind Grid utilities
 * This component creates a responsive layout with:
 * - Sticky header at the top (z-index 30)
 * - Sidebar on desktop (hidden on mobile)
 * - Main content area with proper positioning
 * - Optional floating action button for mobile navigation
 */
export interface LayoutProps {
  header: React.ReactNode;
  sidebar?: React.ReactNode;
  drawer?: React.ReactNode;
  children: React.ReactNode;
  floatingAction?: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ 
  header, 
  sidebar, 
  drawer, 
  children,
  floatingAction
}) => {
  return (
    <div className="grid min-h-screen grid-cols-1 grid-rows-[64px_1fr] md:grid-cols-[250px_1fr] lg:grid-cols-[300px_1fr] bg-background text-foreground relative">
      {/* Header - always visible, spans full width */}
      <header className="sticky top-0 z-30 h-16 flex items-center bg-background border-b border-border col-span-full">
        {header}
      </header>
      
      {/* Sidebar - hidden on mobile, visible on tablet/desktop */}
      {sidebar && (
        <aside className="hidden md:block sticky top-16 h-[calc(100vh-64px)] overflow-y-auto z-20 bg-background border-r border-border row-start-2 col-start-1">
          {sidebar}
        </aside>
      )}
      
      {/* Main content area */}
      <main className="overflow-y-auto p-4 row-start-2 col-start-1 md:col-start-2">
        {children}
      </main>
      
      {/* Mobile drawer - rendered outside grid */}
      {drawer}

      {/* Floating action button for mobile */}
      {floatingAction && (
        <div className="fixed bottom-6 right-6 z-50 md:hidden">
          {floatingAction}
        </div>
      )}
    </div>
  );
};