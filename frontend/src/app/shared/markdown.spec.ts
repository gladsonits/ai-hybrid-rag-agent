import { renderMarkdown } from './markdown';

describe('renderMarkdown', () => {
  it('renders nothing for an empty answer', () => {
    expect(renderMarkdown('')).toBe('');
    expect(renderMarkdown('   ')).toBe('');
  });

  it('renders the markdown of the answer', () => {
    const html = renderMarkdown('**Rose** worked on the project.');

    expect(html).toContain('<strong>Rose</strong>');
  });

  it('turns a citation marker into a chip', () => {
    const html = renderMarkdown('The role was cloud engineer [1].');

    expect(html).toContain('class="cite-chip"');
    expect(html).toContain('data-marker="1"');
  });

  it('turns a fullwidth citation marker into a chip too', () => {
    const html = renderMarkdown('The role was cloud engineer 【4】.');

    expect(html).toContain('class="cite-chip"');
    expect(html).toContain('data-marker="4"');
  });

  it('leaves a link alone', () => {
    const html = renderMarkdown('See [1](https://example.com) for the detail.');

    expect(html).not.toContain('cite-chip');
    expect(html).toContain('href="https://example.com"');
  });

  it('throws away a script the answer tried to smuggle in', () => {
    const html = renderMarkdown('Hello <script>alert("x")</script> there.');

    expect(html).not.toContain('<script');
    expect(html).toContain('Hello');
  });
});
