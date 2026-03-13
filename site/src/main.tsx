import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/manrope";

import App from "./App";
import { initializeAppTheme, ThemeProvider } from "./lib/app-theme";
import "./index.css";

initializeAppTheme();

const rootElement = document.getElementById("root");

if (!(rootElement instanceof HTMLElement)) {
  throw new Error('Root element "#root" was not found.');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
);
