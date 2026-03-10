import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/manrope";

import App from "./App";
import "./index.css";

document.documentElement.classList.remove("dark");
document.documentElement.style.colorScheme = "light";

const rootElement = document.getElementById("root");

if (!(rootElement instanceof HTMLElement)) {
  throw new Error('Root element "#root" was not found.');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
