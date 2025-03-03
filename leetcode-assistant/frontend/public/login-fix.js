// This script patches Monaco Editor's autofill functions that cause errors
(function() {
  // Run only on login page
  if (window.location.pathname.includes('/login')) {
    // Override methods that are causing errors
    window._autoFillControlWithValueWithSingleValueUpdate = function() {};
    window._autoFillControlWithValueRecursively = function() {};
    window._asynchronouslyAutoFillControls = function() {};
    window.autoFillControlsByID = function() {};
    
    // Mock Monaco Editor methods
    if (typeof window.monaco === 'undefined') {
      window.monaco = {
        editor: {
          defineTheme: function() {},
          create: function() { return { dispose: function() {} }; },
          getModels: function() { return []; },
          setTheme: function() {}
        }
      };
    }
    
    // Block specific error sources
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, listener, options) {
      // Block certain events that trigger the autofill error
      if (type === 'input' || type === 'focus' || type === 'blur') {
        const stack = new Error().stack || '';
        if (stack.includes('monaco-editor')) {
          // Skip monaco-related event listeners
          return;
        }
      }
      return originalAddEventListener.call(this, type, listener, options);
    };
  }
})(); 