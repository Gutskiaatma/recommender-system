// Shows what the system understood from the free-text query.
export default function ParsedQuery({ parsed, aspectLabels }) {
  const chips = [];
  Object.entries(parsed.aspects).forEach(([a, w]) =>
    chips.push({ key: `a-${a}`, text: `${aspectLabels[a] ?? a}${w > 1 ? " (emphasised)" : ""}`, tone: "accent" }),
  );
  parsed.ignored_aspects.forEach((a) => chips.push({ key: `i-${a}`, text: `Ignoring ${aspectLabels[a] ?? a}`, tone: "muted" }));
  parsed.brands.forEach((b) => chips.push({ key: `b-${b}`, text: `Brand: ${b}`, tone: "plain" }));
  parsed.excluded_brands.forEach((b) => chips.push({ key: `x-${b}`, text: `Not ${b}`, tone: "muted" }));
  if (parsed.price_min !== null || parsed.price_max !== null) {
    const lo = parsed.price_min !== null ? `$${parsed.price_min}` : "";
    const hi = parsed.price_max !== null ? `$${parsed.price_max}` : "";
    chips.push({ key: "price", text: `Price ${lo && hi ? `${lo}–${hi}` : lo ? `≥ ${lo}` : `≤ ${hi}`}`, tone: "plain" });
  }
  const tone = {
    accent: "bg-accent-soft text-accent",
    muted: "bg-surface-2 text-muted",
    plain: "bg-surface-2 text-ink",
  };
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="font-medium text-ink-2">Understood:</span>
      {chips.length === 0 ? (
        <span className="text-muted">no specific aspect, brand or price — ranking by overall relevance</span>
      ) : (
        chips.map((c) => (
          <span key={c.key} className={`rounded-full px-2.5 py-1 font-medium ${tone[c.tone]}`}>
            {c.text}
          </span>
        ))
      )}
    </div>
  );
}
