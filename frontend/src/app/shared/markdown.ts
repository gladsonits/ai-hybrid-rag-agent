import DOMPurify from 'dompurify';
import { marked, Tokens } from 'marked';

/**
 * The markdown of the model's answer, rendered to HTML.
 *
 * The answer is markdown, so it is rendered as markdown. Two things
 * happen to the result before it reaches the page:
 *
 * 1. The citation markers the model wrote - `[1]`, `[2]` - become
 *    chips, so a reader can see which passage a claim came from and
 *    click straight to it.
 * 2. The HTML is sanitized, because the answer is text from a model
 *    and the page is not the place to trust it.
 */

/**
 * A citation marker, as the model writes it.
 *
 * The models write the ASCII brackets most of the time and the
 * fullwidth ones 【 】 the rest of the time, and the API understands
 * both - so the chip is made for both as well.
 */
const CITATION = /^[\[【]\s*(\d{1,3})\s*[\]】](?!\()/;

marked.use({
  gfm: true,
  breaks: true,
  extensions: [
    {
      name: 'citation',
      level: 'inline',
      tokenizer(src: string) {
        const match = CITATION.exec(src);
        if (match === null) {
          return undefined;
        }
        return { type: 'citation', raw: match[0], text: match[1] };
      },
      renderer(token: Tokens.Generic) {
        const marker = String(token['text']);
        return `<span class="cite-chip" data-marker="${marker}" title="Passage ${marker}">[${marker}]</span>`;
      },
    },
  ],
});

/**
 * Render an answer to HTML that is safe to put on the page.
 *
 * @param text the raw answer, as the model wrote it
 * @returns the sanitized HTML
 */
export function renderMarkdown(text: string): string {
  if (text.trim() === '') {
    return '';
  }
  const html = marked.parse(text, { async: false }) as string;
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
}
