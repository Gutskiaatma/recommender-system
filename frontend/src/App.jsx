import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import { Loading } from "./components/Bits.jsx";
import HomePage from "./pages/HomePage.jsx";
import SearchPage from "./pages/SearchPage.jsx";
import NotFoundPage from "./pages/NotFoundPage.jsx";

// Chart-heavy pages are loaded on demand so the search page stays light.
const ComparePage = lazy(() => import("./pages/ComparePage.jsx"));
const ProductPage = lazy(() => import("./pages/ProductPage.jsx"));
const InsightsPage = lazy(() => import("./pages/InsightsPage.jsx"));

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="product/:productId" element={<ProductPage />} />
          <Route path="insights" element={<InsightsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
