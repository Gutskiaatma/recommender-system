import { createContext, useCallback, useContext, useMemo, useState } from "react";

// Holds the (max 3) products the user ticked for comparison, plus the query
// they came from, so the comparison can show match scores for that query.
const CompareContext = createContext(null);
export const MAX_COMPARE = 3;

export function CompareProvider({ children }) {
  const [items, setItems] = useState([]); // [{product_id, title, brand}]
  const [query, setQuery] = useState("");

  const toggle = useCallback((product) => {
    setItems((current) => {
      if (current.some((p) => p.product_id === product.product_id)) {
        return current.filter((p) => p.product_id !== product.product_id);
      }
      if (current.length >= MAX_COMPARE) return current;
      return [...current, { product_id: product.product_id, title: product.title, brand: product.brand }];
    });
  }, []);

  const clear = useCallback(() => setItems([]), []);
  const has = useCallback((id) => items.some((p) => p.product_id === id), [items]);

  const value = useMemo(
    () => ({ items, toggle, clear, has, query, setQuery, full: items.length >= MAX_COMPARE }),
    [items, toggle, clear, has, query],
  );
  return <CompareContext.Provider value={value}>{children}</CompareContext.Provider>;
}

export function useCompare() {
  const ctx = useContext(CompareContext);
  if (!ctx) throw new Error("useCompare must be used inside CompareProvider");
  return ctx;
}
