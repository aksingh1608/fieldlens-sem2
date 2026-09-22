export function ArchitectureDiagram() {
  return (
    <figure className="overflow-x-auto rounded-md border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-950">
      <svg
        viewBox="0 0 920 280"
        role="img"
        aria-labelledby="arch-title arch-desc"
        className="mx-auto h-auto w-full max-w-4xl"
      >
        <title id="arch-title">FieldLens segmentation architecture</title>
        <desc id="arch-desc">
          MiT-B0 encoder on RGB, optional NIR stem with gated fusion, all-MLP decoder, softmax or
          sigmoid heads.
        </desc>
        <defs>
          <marker
            id="arrow"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 Z" fill="#52525b" />
          </marker>
        </defs>

        {/* RGB input */}
        <rect x="20" y="100" width="100" height="56" rx="4" fill="#ecfdf5" stroke="#0d9488" />
        <text x="70" y="132" textAnchor="middle" className="fill-zinc-800 text-[13px] font-medium">
          RGB tile
        </text>

        {/* NIR input */}
        <rect x="20" y="20" width="100" height="56" rx="4" fill="#f0f9ff" stroke="#2563eb" />
        <text x="70" y="52" textAnchor="middle" className="fill-zinc-800 text-[13px] font-medium">
          NIR tile
        </text>

        {/* NIR stem */}
        <rect x="160" y="20" width="110" height="56" rx="4" fill="#fafafa" stroke="#71717a" />
        <text x="215" y="44" textAnchor="middle" className="fill-zinc-800 text-[12px] font-medium">
          NIR stem
        </text>
        <text x="215" y="62" textAnchor="middle" className="fill-zinc-600 text-[11px]">
          conv blocks
        </text>

        {/* MiT-B0 */}
        <rect x="160" y="100" width="130" height="56" rx="4" fill="#fafafa" stroke="#71717a" />
        <text x="225" y="124" textAnchor="middle" className="fill-zinc-800 text-[12px] font-medium">
          MiT-B0 encoder
        </text>
        <text x="225" y="142" textAnchor="middle" className="fill-zinc-600 text-[11px]">
          SegFormer backbone
        </text>

        {/* Gates */}
        <rect x="330" y="52" width="120" height="72" rx="4" fill="#fff7ed" stroke="#c2410c" />
        <text x="390" y="84" textAnchor="middle" className="fill-zinc-800 text-[12px] font-medium">
          Gated fusion
        </text>
        <text x="390" y="102" textAnchor="middle" className="fill-zinc-600 text-[11px]">
          Run 3 only
        </text>

        {/* Decoder */}
        <rect x="490" y="100" width="120" height="56" rx="4" fill="#fafafa" stroke="#71717a" />
        <text x="550" y="132" textAnchor="middle" className="fill-zinc-800 text-[12px] font-medium">
          All-MLP decoder
        </text>

        {/* Heads */}
        <rect x="650" y="60" width="110" height="48" rx="4" fill="#ecfdf5" stroke="#0d9488" />
        <text x="705" y="88" textAnchor="middle" className="fill-zinc-800 text-[11px] font-medium">
          Softmax head
        </text>
        <text x="705" y="102" textAnchor="middle" className="fill-zinc-600 text-[10px]">
          Run 1
        </text>

        <rect x="650" y="148" width="110" height="48" rx="4" fill="#ecfdf5" stroke="#0d9488" />
        <text x="705" y="176" textAnchor="middle" className="fill-zinc-800 text-[11px] font-medium">
          Sigmoid head
        </text>
        <text x="705" y="190" textAnchor="middle" className="fill-zinc-600 text-[10px]">
          Runs 2-3
        </text>

        {/* Output */}
        <rect x="800" y="100" width="100" height="56" rx="4" fill="#f4f4f5" stroke="#52525b" />
        <text x="850" y="124" textAnchor="middle" className="fill-zinc-800 text-[12px] font-medium">
          Masks +
        </text>
        <text x="850" y="142" textAnchor="middle" className="fill-zinc-800 text-[12px] font-medium">
          tile alerts
        </text>

        {/* Arrows */}
        <line x1="120" y1="128" x2="158" y2="128" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="120" y1="48" x2="158" y2="48" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="270" y1="48" x2="328" y2="72" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="290" y1="128" x2="328" y2="100" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="450" y1="88" x2="488" y2="118" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="610" y1="128" x2="648" y2="128" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="610" y1="128" x2="648" y2="172" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="760" y1="84" x2="798" y2="118" stroke="#52525b" markerEnd="url(#arrow)" />
        <line x1="760" y1="172" x2="798" y2="138" stroke="#52525b" markerEnd="url(#arrow)" />
      </svg>
      <figcaption className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
        Shared encoder-decoder with run-specific input path and output head.
      </figcaption>
    </figure>
  );
}
