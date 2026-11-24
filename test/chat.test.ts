import { describe, expect, it } from 'vitest';
import { fitChat, renderChat, SMOL_SYSTEM, type ChatMessage } from '../src/engine/chat';

describe('Chat-Template (ChatML)', () => {
  it('setzt den Standard-System-Prompt, wenn keiner angegeben ist', () => {
    expect(renderChat([{ role: 'user', content: 'Hi' }])).toBe(
      `<|im_start|>system\n${SMOL_SYSTEM}<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n`,
    );
  });
  it('übernimmt einen eigenen System-Prompt', () => {
    expect(renderChat([{ role: 'system', content: 'S' }, { role: 'user', content: 'U' }], false)).toBe(
      '<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n',
    );
  });
});

describe('Chat auf das Prompt-Limit kürzen', () => {
  // ein „Token“ pro Wort, damit die Rechnung nachvollziehbar bleibt
  const count = (t: string) => t.split(/\s+/).filter(Boolean).length;
  const msgs: ChatMessage[] = [
    { role: 'user', content: 'eins zwei drei vier' },
    { role: 'assistant', content: 'fünf sechs sieben acht' },
    { role: 'user', content: 'neun zehn' },
    { role: 'assistant', content: 'elf zwölf' },
    { role: 'user', content: 'dreizehn' },
  ];

  it('alles passt: nichts wird entfernt', () => {
    const f = fitChat('S', msgs, count, 1000);
    expect(f.dropped).toBe(0);
    expect(f.kept).toEqual(msgs);
    expect(f.tooLong).toBe(false);
  });

  it('entfernt die ältesten Nachrichten paarweise, der System-Prompt bleibt', () => {
    const full = count(renderChat([{ role: 'system', content: 'S' }, ...msgs]));
    const f = fitChat('S', msgs, count, full - 1);
    expect(f.dropped).toBe(2);
    expect(f.kept[0]).toEqual({ role: 'user', content: 'neun zehn' });
    expect(f.tokens).toBeLessThan(full);
  });

  it('bis zur neuesten Nachricht, danach Fehler statt stillem Abschneiden', () => {
    const f = fitChat('S', msgs, count, 3);
    expect(f.kept).toEqual([msgs[4]]);
    expect(f.dropped).toBe(4);
    expect(f.tooLong).toBe(true);
  });
});
