/**
 * Chat-Template (ChatML) wie bei SmolLM2-Instruct und Kürzung auf das Prompt-Limit.
 * Die Golden-Tests prüfen, dass der gerenderte Text dem von `apply_chat_template` (Python) gleicht.
 */

export type ChatRole = 'system' | 'user' | 'assistant';
export interface ChatMessage {
  role: ChatRole;
  content: string;
}

/** Standard-System-Prompt aus dem Chat-Template von SmolLM2-Instruct */
export const SMOL_SYSTEM = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face';

export function renderChat(messages: ChatMessage[], addGenerationPrompt = true, defaultSystem = SMOL_SYSTEM): string {
  let out = '';
  messages.forEach((m, i) => {
    if (i === 0 && m.role !== 'system') out += `<|im_start|>system\n${defaultSystem}<|im_end|>\n`;
    out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  });
  if (addGenerationPrompt) out += '<|im_start|>assistant\n';
  return out;
}

export interface FittedChat {
  /** Nachrichten, die in den Kontext passen (ohne System-Prompt) */
  kept: ChatMessage[];
  /** Anzahl der ältesten Nachrichten, die weggelassen wurden */
  dropped: number;
  /** Tokens des gerenderten Kontexts (mit System-Prompt und Antwortbeginn) */
  tokens: number;
  /** Passt nicht einmal die neueste Nachricht allein? */
  tooLong: boolean;
}

/**
 * Lässt die ältesten Nachrichten weg, bis der gerenderte Chat höchstens `limit` Tokens hat.
 * Entfernt wird paarweise (Frage + Antwort), der System-Prompt bleibt immer.
 */
export function fitChat(system: string, messages: ChatMessage[], countTokens: (text: string) => number, limit: number): FittedChat {
  const render = (ms: ChatMessage[]) => countTokens(renderChat([{ role: 'system', content: system }, ...ms]));
  let start = 0;
  let tokens = render(messages);
  while (tokens > limit && start < messages.length - 1) {
    // bis zur nächsten Nutzernachricht weglassen (ein Paar Frage + Antwort)
    start++;
    while (start < messages.length - 1 && messages[start].role !== 'user') start++;
    tokens = render(messages.slice(start));
  }
  return { kept: messages.slice(start), dropped: start, tokens, tooLong: tokens > limit };
}
