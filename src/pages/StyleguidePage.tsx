import { useMemo, type ReactNode } from 'react';
import katex from 'katex';
import { useResolvedTheme } from '../state/settings';
import { CalloutNum, Figure, Stamp } from '../components/primitives';
import { PipelineFigure } from '../components/PipelineFigure';
import { DistributionDemo } from '../components/DistributionDemo';
import { TokenChips, type ChipToken } from '../components/TokenChips';
import { PaletteLab } from '../components/PaletteLab';
import { RolesInUse } from '../components/RolesInUse';
import { CardIndexIllustration, ConveyorIllustration, DiceIllustration } from '../components/Illustrations';

/**
 * #/styleguide – das Musterblatt aus D0, jetzt mit den festgelegten Schriften.
 * Entscheidungen: docs/design-decisions.md
 */

const TOKENS: ChipToken[] = [
  { text: '<|im_start|>', id: 1, special: true },
  { text: 'user', id: 4093 },
  { text: '\n', id: 198 },
  { text: 'Die', id: 21834 },
  { text: ' Zentral', id: 47420 },
  { text: 'bank', id: 6331 },
  { text: ' erhöht', id: 33187 },
  { text: ' den', id: 1667 },
  { text: ' Leit', id: 29466 },
  { text: 'zins', id: 41927 },
  { text: '.', id: 30 },
  { text: '<|im_end|>', id: 2, special: true },
];

function Section({ num, title, intro, children }: { num: string; title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <section className="section" aria-labelledby={`s-${num}`}>
      <header className="section__head">
        <span className="meta">{num}</span>
        <h2 id={`s-${num}`}>{title}</h2>
        {intro && <p className="muted">{intro}</p>}
      </header>
      {children}
    </section>
  );
}

export default function StyleguidePage() {
  const theme = useResolvedTheme();
  const crossEntropy = useMemo(
    () =>
      katex.renderToString(String.raw`\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\left(x_t \mid x_{<t}\right)`, {
        displayMode: true,
        throwOnError: false,
      }),
    [],
  );

  return (
    <article className="page styleguide">
      <header className="page__head">
        <span className="meta">Design-System · Musterblatt</span>
        <h1>Styleguide</h1>
        <p className="lead muted">
          Alle Grundbausteine der Seite an einem Ort: Farben mit ihren Prüfregeln, Schriften, Line Art, Regler, Token-Chips
          und Stempel. Die Entscheidungen dazu stehen in <code>docs/design-decisions.md</code>.
        </p>
      </header>

      <Section num="01" title="Farbe" intro="Grundsätzlich monochrom. Jede Farbe hat genau eine Bedeutung; die Prüfregeln werden live berechnet.">
        <PaletteLab theme={theme} />
        <h3 className="section__sub">Die Rollen im Einsatz</h3>
        <Figure num="1.1" caption="Jede Rolle in ihrem typischen Kontext, zusätzlich über Form oder Symbol kodiert." stamp="illustrative">
          <RolesInUse />
        </Figure>
      </Section>

      <Section num="02" title="Schrift" intro="Fraunces für Headlines, Inter für Fließtext, JetBrains Mono (ohne Ligaturen) für Tokens, IDs und Zahlen.">
        <div className="specimen">
          <div className="specimen__item">
            <span className="meta">Headline · Fraunces · SOFT 50 · WONK 1</span>
            <span className="specimen__display">Wie aus einem Prompt eine Antwort wird</span>
            <span className="specimen__sub">Kapitel 5 · Attention: Worauf achtet ein Token?</span>
          </div>
          <div className="specimen__item">
            <span className="meta">Fließtext · Inter</span>
            <p>
              Ein Sprachmodell gibt keine Antwort aus, sondern eine Wahrscheinlichkeitsverteilung über alle Tokens seines
              Vokabulars. Erst das Sampling macht daraus ein konkretes Wort. Zahlen im Text: 0,31 · 21,7 % · 1.024 Tokens.
            </p>
          </div>
          <div className="specimen__item">
            <span className="meta">Monospace · JetBrains Mono</span>
            <p className="mono specimen__mono">
              {'<|im_start|>user↵ · ID 4218 · p=0.3121'}
              <br />
              {'[T × d_model] · 0123456789 · Il1| O0 · -> != =>'}
            </p>
          </div>
        </div>
      </Section>

      <Section num="03" title="Fließtext & Formeln" intro="Lesebreite ca. 68 Zeichen. Formeln rendert KaTeX.">
        <div className="prose">
          <p>
            Ein Sprachmodell lernt im <span className="term term--static">Pretraining</span> nur eine Aufgabe: das nächste
            Token vorherzusagen. Gemessen wird das mit der Cross-Entropy.
          </p>
          <div className="formula" dangerouslySetInnerHTML={{ __html: crossEntropy }} />
          <aside className="deepdive">
            <span className="meta">Unter der Haube</span>
            <p>Die Division durch √d_k verhindert, dass die Softmax bei vielen Dimensionen fast nur einen Wert auswählt.</p>
          </aside>
          <p className="hero__todo">
            <CalloutNum n={1} /> nummerierter Hinweis im Text
          </p>
        </div>
      </Section>

      <Section num="04" title="Line Art & Bewegung" intro="Linien zeichnen sich einmal beim Erscheinen; der Puls zeigt den Datenfluss.">
        <Figure num="4.1" caption="Ein Durchlauf: vom Prompt zum nächsten Token, dann zurück in den Loop." stamp="illustrative">
          <PipelineFigure />
        </Figure>
        <div className="illus">
          <figure>
            <DiceIllustration />
            <figcaption className="meta">Sampling</figcaption>
          </figure>
          <figure>
            <CardIndexIllustration />
            <figcaption className="meta">Attention</figcaption>
          </figure>
          <figure>
            <ConveyorIllustration />
            <figcaption className="meta">Residual Stream</figcaption>
          </figure>
        </div>
      </Section>

      <Section num="05" title="Regler & Verteilung">
        <Figure num="5.1" caption="Temperatur und Top-p formen die Verteilung über die nächsten Tokens." stamp="illustrative">
          <DistributionDemo />
        </Figure>
      </Section>

      <Section num="06" title="Token-Chips">
        <Figure num="6.1" caption="Chat-Nachricht als Token-Folge mit IDs (Beispiel-Zerlegung)." stamp="illustrative">
          <TokenChips tokens={TOKENS} />
        </Figure>
      </Section>

      <Section num="07" title="Stempel" intro="Jede Abbildung sagt, woher ihre Zahlen kommen.">
        <div className="stamps">
          {(
            [
              ['real', 'live im Browser mit dem echten Sprachmodell berechnet'],
              ['toy', 'live und exakt mit dem Spielzeugmodell berechnet'],
              ['precomputed', 'echte Werte, vorab berechnet'],
              ['illustrative', 'vereinfacht: zeigt das Prinzip, keine echten Werte'],
            ] as const
          ).map(([k, d]) => (
            <div key={k} className="stamps__item">
              <Stamp kind={k} />
              <span className="muted small">{d}</span>
            </div>
          ))}
        </div>
      </Section>
    </article>
  );
}
