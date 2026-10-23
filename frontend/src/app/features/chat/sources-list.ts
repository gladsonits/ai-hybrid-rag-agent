import { Component, computed, input, signal } from '@angular/core';

import { ChatCitation, ChatSource } from '../../core/models/chat';

/**
 * The passages an answer was built from.
 *
 * The passages arrive before the answer does, so they can be shown
 * while the answer is still being written. The number in front of a
 * passage is the number the model cites it by.
 */
@Component({
  selector: 'app-sources-list',
  templateUrl: './sources-list.html',
  styleUrl: './sources-list.css',
})
export class SourcesList {
  /** The passages the turn was answered from. */
  readonly sources = input.required<ChatSource[]>();

  /** The passages the answer actually cited. */
  readonly citations = input.required<ChatCitation[]>();

  /** The marker of the passage that is being pointed at. */
  readonly highlight = input<number | null>(null);

  /** Whether the passages are shown. */
  readonly open = signal(false);

  /** The passages whose text is shown in full. */
  private readonly fully = signal<ReadonlySet<number>>(new Set());

  /** How many of the passages the answer actually cited. */
  readonly cited = computed(() => this.citations().length);

  /**
   * The number the model cites a passage by.
   *
   * The citation carries the id of the passage it points at, so the
   * number is looked up rather than guessed from the position. A
   * passage the model did not cite keeps the number of its position.
   */
  markerOf(source: ChatSource, index: number): number {
    const citation = this.citations().find((entry) => entry.chunk_id === source.chunk_id);
    return citation?.marker ?? index + 1;
  }

  /** Whether a passage was cited. */
  isCited(source: ChatSource): boolean {
    return this.citations().some((entry) => entry.chunk_id === source.chunk_id);
  }

  /** Show the text of a passage in full, or fold it away again. */
  toggleText(chunkId: number): void {
    this.fully.update((current) => {
      const next = new Set(current);
      if (next.has(chunkId)) {
        next.delete(chunkId);
      } else {
        next.add(chunkId);
      }
      return next;
    });
  }

  /** Whether the text of a passage is shown in full. */
  isFull(chunkId: number): boolean {
    return this.fully().has(chunkId);
  }

  /**
   * The metadata of a passage, as the short labels worth showing.
   *
   * The metadata carries the heading, the project, the client, the
   * role, the label and the page of the passage - whichever of them the
   * document provided.
   */
  labelsOf(source: ChatSource): { name: string; value: string }[] {
    const metadata = source.metadata ?? {};
    const wanted = ['section', 'project', 'client', 'role', 'label'];
    const labels = wanted
      .map((name) => ({ name, value: readText(metadata[name]) }))
      .filter((label) => label.value !== '');
    return labels;
  }

  /** The pages a passage came from. */
  pagesOf(source: ChatSource): string {
    const metadata = source.metadata ?? {};
    const page = readText(metadata['page']);
    const endPage = readText(metadata['end_page']);
    if (page === '') {
      return '';
    }
    return endPage !== '' && endPage !== page ? `p. ${page}–${endPage}` : `p. ${page}`;
  }

  /** The score of a passage, as it is shown. */
  scoreText(source: ChatSource): string {
    return source.score === null || source.score === undefined
      ? ''
      : `score ${source.score.toFixed(3)}`;
  }
}

/** Read a metadata value as text, when there is one. */
function readText(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }
  if (typeof value === 'number') {
    return String(value);
  }
  return '';
}
