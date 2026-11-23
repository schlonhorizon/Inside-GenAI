/**
 * Kleine Line-Art-Illustrationen für die Intuitions-Ebene.
 * Einheitlich: 1,25 px Strich, runde Enden, Tinte; Akzent nur für das „aktive“ Detail.
 */

const common = {
  fill: 'none',
  stroke: 'var(--ink)',
  strokeWidth: 1.25,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

/** Würfel – Sampling. Das gezogene Auge ist der Akzent. */
export function DiceIllustration({ size = 132 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" role="img" aria-label="Würfel als Bild für Sampling">
      <g {...common}>
        {/* Vorderseite */}
        <rect x="22" y="40" width="52" height="52" rx="8" />
        {/* Oberseite */}
        <path d="M30 40 L50 24 L96 24 L74 40" />
        <path d="M96 24 Q100 25 100 30 L100 72 Q100 77 96 80 L74 92" />
        {/* Augen vorne: 5 */}
        <circle cx="35" cy="53" r="2.6" />
        <circle cx="61" cy="53" r="2.6" />
        <circle cx="35" cy="79" r="2.6" />
        <circle cx="61" cy="79" r="2.6" />
        {/* Augen oben: 2 */}
        <ellipse cx="54" cy="31" rx="3.2" ry="1.8" />
        <ellipse cx="78" cy="31" rx="3.2" ry="1.8" />
        {/* Augen rechts: 3 */}
        <ellipse cx="82" cy="50" rx="1.8" ry="2.8" />
        <ellipse cx="87" cy="60" rx="1.8" ry="2.8" />
        <ellipse cx="92" cy="70" rx="1.8" ry="2.8" />
        {/* Bewegungsstriche */}
        <path d="M8 58 L16 58 M6 68 L15 68 M10 78 L16 78" stroke="var(--ink-muted)" />
      </g>
      <circle cx="48" cy="66" r="3.4" fill="var(--accent)" />
    </svg>
  );
}

/** Karteikasten – Attention (Query sucht passende Keys, liest Values). */
export function CardIndexIllustration({ size = 132 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" role="img" aria-label="Karteikasten als Bild für Attention">
      <g {...common}>
        {/* Kasten */}
        <path d="M14 62 L14 98 Q14 102 18 102 L82 102 Q86 102 86 98 L86 62" />
        <path d="M10 62 L90 62" />
        {/* Karten */}
        <path d="M22 62 L22 40 Q22 37 25 37 L45 37 Q48 37 48 40 L48 62" />
        <path d="M34 62 L34 46 Q34 43 37 43 L57 43 Q60 43 60 46 L60 62" />
        <path d="M58 62 L58 50 Q58 47 61 47 L77 47 Q80 47 80 50 L80 62" />
        <path d="M27 44 L41 44 M39 50 L53 50 M63 54 L73 54" stroke="var(--ink-muted)" />
        {/* Lupe = Query */}
        <circle cx="94" cy="30" r="11" />
        <path d="M86 38 L76 48" />
      </g>
      {/* gezogene Karte (aktiv) */}
      <path
        d="M34 46 Q34 43 37 43 L57 43 Q60 43 60 46"
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2}
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Förderband – Residual Stream. */
export function ConveyorIllustration({ size = 132 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" role="img" aria-label="Förderband als Bild für den Residual Stream">
      <g {...common}>
        <rect x="8" y="70" width="104" height="18" rx="9" />
        <circle cx="17" cy="79" r="5" />
        <circle cx="103" cy="79" r="5" />
        <circle cx="60" cy="79" r="2" />
        {/* Pakete */}
        <rect x="22" y="52" width="16" height="18" rx="2" />
        <rect x="46" y="48" width="18" height="22" rx="2" />
        {/* Greifarm legt etwas dazu */}
        <path d="M86 18 L86 34 M78 34 L94 34 M80 34 L80 40 M92 34 L92 40" />
        <path d="M60 104 L76 104 M66 110 L84 110" stroke="var(--ink-muted)" />
      </g>
      <rect x="78" y="42" width="16" height="14" rx="2" fill="none" stroke="var(--accent)" strokeWidth={2} />
      <path d="M86 58 L86 64" stroke="var(--accent)" strokeWidth={1.5} strokeLinecap="round" strokeDasharray="2 3" />
    </svg>
  );
}
