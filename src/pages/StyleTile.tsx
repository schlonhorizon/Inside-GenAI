import { useMemo, type ReactNode } from 'react';
import katex from 'katex';
import { useTheme, type ThemePref } from '../lib/theme';
import { CalloutNum, Figure, Segmented, Stamp } from '../components/primitives';
import { PipelineFigure } from '../components/PipelineFigure';
import { DistributionDemo } from '../components/DistributionDemo';
import { TokenChips, type ChipToken } from '../components/TokenChips';
import { PaletteLab } from '../components/PaletteLab';
import { RolesInUse } from '../components/RolesInUse';
import { FontLab, useFontChoice, DISPLAY_FONTS, BODY_FONTS, MONO_FONTS } from '../components/FontLab';
import { CardIndexIllustration, ConveyorIllustration, DiceIllustration } from '../components/Illustrations';

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

export function StyleTile() {
  const { pref, setPref, theme } = useTheme();
  const fonts = useFontChoice();

  const crossEntropy = useMemo(
    () =>
      katex.renderToString(String.raw`\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\left(x_t \mid x_{<t}\right)`, {
        displayMode: true,
        throwOnError: false,
      }),
    [],
  );
  const attention = useMemo(
    () =>
      katex.renderToString(String.raw`\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`, {
        displayMode: false,
        throwOnError: false,
      }),
    [],
  );

  return (
    <div className="page">
      <div className="toolbar">
        <span className="meta toolbar__brand">Inside GenAI · D0</span>
        <Segmented<ThemePref>
          label="Modus"
          value={pref}
          options={[
            { value: 'system', label: 'System' },
            { value: 'light', label: 'Hell' },
            { value: 'dark', label: 'Dunkel' },
          ]}
          onChange={setPref}
        />
        <span className="meta toolbar__fonts">
          {DISPLAY_FONTS[fonts.display]?.label} · {BODY_FONTS[fonts.body]?.label} · {MONO_FONTS[fonts.mono]?.label}
        </span>
      </div>

      <header className="hero">
        <span className="meta">Design-Probe · Style Tile · Schriften &amp; Farben festgelegt</span>
        <h1>Wie aus einem Prompt eine Antwort wird</h1>
        <p className="hero__lead">
          Diese Seite ist keine Lerneinheit, sondern ein Musterblatt. Sie zeigt alle Grundbausteine der künftigen Seite:
          Farben in beiden Modi, Schrift-Kandidaten, Line-Art-Diagramme, Regler, Token-Chips, Stempel und Callouts.
          Wähle Schriften und Modus direkt hier aus; die Auswahl gilt sofort für die ganze Seite.
        </p>
        <ul className="hero__todo">
          <li>
            <CalloutNum n={1} /> Headline-Schrift wählen (Abschnitt 02)
          </li>
          <li>
            <CalloutNum n={2} /> Fließtext- und Mono-Schrift wählen
          </li>
          <li>
            <CalloutNum n={3} /> Farben in Hell und Dunkel prüfen (Abschnitt 01)
          </li>
        </ul>
      </header>

      <Section
        num="01"
        title="Farbe: Tinte auf warmem Papier"
        intro="Grundsätzlich monochrom. Farbe trägt immer eine Bedeutung, und jede Farbe hat genau eine. Die Prüfregeln aus dem Plan werden hier live berechnet."
      >
        <PaletteLab theme={theme} />
        <h3 className="section__sub">Die Rollen im Einsatz</h3>
        <Figure num="1.1" caption="Jede Rolle in ihrem typischen Kontext, zusätzlich über Form oder Symbol kodiert." stamp="illustrative">
          <RolesInUse />
        </Figure>
      </Section>

      <Section
        num="02"
        title="Schrift"
        intro="Entschieden: Fraunces für Headlines, Inter für Fließtext, JetBrains Mono für Tokens, IDs und Zahlen. Die übrigen Kandidaten bleiben bis P0 zum Gegenprüfen sichtbar. Alle Schriften sind Open Source (SIL OFL) und lokal eingebunden."
      >
        <FontLab f={fonts} />
      </Section>

      <Section num="03" title="Fließtext & Formeln" intro="Lesebreite ca. 68 Zeichen. Formeln rendert KaTeX, kein LaTeX-Code auf der Seite.">
        <div className="prose">
          <p>
            Ein Sprachmodell lernt im <span className="term">Pretraining</span> nur eine einzige Aufgabe: das nächste{' '}
            <span className="term">Token</span> vorherzusagen. Gemessen wird das mit der{' '}
            <span className="term">Cross-Entropy</span>, also mit der Frage, wie überrascht das Modell vom tatsächlichen
            nächsten Token war.
          </p>
          <div className="formula" dangerouslySetInnerHTML={{ __html: crossEntropy }} />
          <p>
            Innerhalb jedes Layers verteilt die <span className="term">Attention</span> Aufmerksamkeit über frühere Tokens:{' '}
            <span dangerouslySetInnerHTML={{ __html: attention }} />. Das Skalarprodukt <code>QKᵀ</code> misst, wie gut eine
            Frage zu einem Etikett passt.
          </p>
          <aside className="deepdive">
            <span className="meta">Unter der Haube</span>
            <p>
              Die Division durch <code>√d_k</code> verhindert, dass die Skalarprodukte bei vielen Dimensionen so groß
              werden, dass die Softmax fast nur noch einen einzigen Wert auswählt.
            </p>
          </aside>
        </div>
      </Section>

      <Section
        num="04"
        title="Line Art & Bewegung"
        intro="Linien zeichnen sich einmal beim Erscheinen. Der Puls zeigt, wo Daten fließen. Mit reduzierter Bewegung bleibt alles statisch und vollständig."
      >
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

      <Section num="05" title="Regler & Verteilung" intro="Parameter ändern, Wirkung sofort sehen. Balken in Sepia, das gezogene Token im Akzent.">
        <Figure num="5.1" caption="Temperatur und Top-p formen die Verteilung über die nächsten Tokens." stamp="illustrative">
          <DistributionDemo />
        </Figure>
      </Section>

      <Section
        num="06"
        title="Token-Chips"
        intro="Abwechselnd zwei neutrale Flächen, nur zur Abgrenzung. Special Tokens gestrichelt. Klick wählt ein Token aus (Akzent-Rahmen)."
      >
        <Figure num="6.1" caption="Chat-Nachricht als Token-Folge mit IDs. Die Zerlegung ist hier nur ein Beispiel." stamp="illustrative">
          <TokenChips tokens={TOKENS} />
        </Figure>
      </Section>

      <Section num="07" title="Stempel" intro="Jede Abbildung sagt, woher ihre Zahlen kommen. Unterscheidung nur über die Form, nicht über Farbe.">
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

      <footer className="footer">
        <p>
          <strong>Privates Lernprojekt, nicht mit Anthropic verbunden.</strong>
        </p>
        <p className="muted small">
          Code: MIT-Lizenz. Schriften: Fraunces, Instrument Serif, Young Serif, Inter, IBM Plex Sans/Mono, JetBrains Mono, alle
          unter SIL Open Font License 1.1, lokal eingebunden. Formelsatz: KaTeX (MIT). Animation: Motion (MIT).
        </p>
      </footer>
    </div>
  );
}
