/**
 * A decoder for the server-sent event stream.
 *
 * The stream arrives in chunks that do not line up with the events: one
 * read can hold half an event, or three of them. The decoder holds the
 * incomplete tail until the rest of it arrives, so a frame is only
 * handed over once it is whole.
 *
 * The format is the one the SSE specification defines, and the one the
 * API writes: a frame is a set of `field: value` lines, ended by a blank
 * line. A line that starts with a colon is a comment and is ignored.
 */

/** One whole event, taken off the stream. */
export interface SseFrame {
  /** The value of the `event:` field, or `message` when there is none. */
  event: string;
  /** The value of the `data:` field. Several data lines are joined with a newline. */
  data: string;
}

/** The blank line that ends a frame, in either line-ending style. */
const FRAME_SEPARATOR = /\r?\n\r?\n/;

/** A line of a frame, and the field name it carries. */
const FIELD = /^([^\s:]+)[ \t]*:?[ \t]?(.*)$/;

export class SseDecoder {
  /** The part of the stream that has not been read as a whole frame yet. */
  private buffer = '';

  /**
   * Take a piece of the stream and return the frames it completed.
   *
   * @param chunk the text that arrived
   * @returns the frames that are now whole, in the order they arrived
   */
  push(chunk: string): SseFrame[] {
    this.buffer += chunk;
    const frames: SseFrame[] = [];

    let separator = FRAME_SEPARATOR.exec(this.buffer);
    while (separator !== null) {
      const raw = this.buffer.slice(0, separator.index);
      this.buffer = this.buffer.slice(separator.index + separator[0].length);
      const frame = parseFrame(raw);
      if (frame !== null) {
        frames.push(frame);
      }
      separator = FRAME_SEPARATOR.exec(this.buffer);
    }

    return frames;
  }

  /**
   * Return whatever is left once the stream has ended.
   *
   * A stream that ends without its last blank line still carries a
   * frame, and dropping it would lose the last event of the answer.
   */
  flush(): SseFrame[] {
    const raw = this.buffer;
    this.buffer = '';
    const frame = parseFrame(raw);
    return frame === null ? [] : [frame];
  }
}

/**
 * Read one frame out of its lines.
 *
 * @returns the frame, or null when it carries nothing worth handing over
 */
function parseFrame(raw: string): SseFrame | null {
  let event = 'message';
  const data: string[] = [];

  for (const line of raw.split(/\r?\n/)) {
    if (line === '' || line.startsWith(':')) {
      continue;
    }

    const field = FIELD.exec(line);
    if (field === null) {
      continue;
    }

    const name = field[1];
    const value = field[2];

    if (name === 'event') {
      event = value;
    } else if (name === 'data') {
      data.push(value);
    }
  }

  if (data.length === 0 && event === 'message') {
    return null;
  }

  return { event, data: data.join('\n') };
}
