import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import { CompareProvider } from "./lib/compareContext.jsx";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <CompareProvider>
        <App />
      </CompareProvider>
    </BrowserRouter>
  </StrictMode>,
);
