import { MiniPlayground } from '../chapters/shared/MiniPlayground';
import { Fact } from '../components/learn/Learn';
import { href } from '../app/router';

export default function PlaygroundPage() {
  return (
    <article className="page playground">
      <header className="page__head">
        <span className="meta">Ausprobieren</span>
        <h1>Mini-Playground</h1>
        <p className="lead muted">
          Gib einen Anfang vor und lass das Spielzeugmodell weiterschreiben. Es kennt nur{' '}
          <Fact m="toy" k="vocab" /> Wörter und hat <Fact m="toy" k="params" /> Parameter. Darum klingt es schlicht, folgt aber
          denselben Regeln wie ein großes Modell.
        </p>
        <p className="small">
          Hintergrund: <a href={href.chapter('next-token')}>Kapitel 7 (Sampling)</a> · <a href={href.chapter('generierung')}>Kapitel 8 (Generierungs-Loop)</a>.
          Der große Playground mit echtem Modell und A/B-Vergleich folgt in Kapitel 15.
        </p>
      </header>
      <MiniPlayground />
    </article>
  );
}
