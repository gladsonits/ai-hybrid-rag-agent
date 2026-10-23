import { SseDecoder } from './sse-decoder';

describe('SseDecoder', () => {
  it('reads a frame that arrived whole', () => {
    const decoder = new SseDecoder();
    const frames = decoder.push('event: token\ndata: {"type":"token"}\n\n');

    expect(frames).toEqual([{ event: 'token', data: '{"type":"token"}' }]);
  });

  it('waits for the rest of a frame that was cut in half', () => {
    const decoder = new SseDecoder();

    expect(decoder.push('event: token\ndata: {"ty')).toEqual([]);
    expect(decoder.push('pe":"token"}\n\n')).toEqual([
      { event: 'token', data: '{"type":"token"}' },
    ]);
  });

  it('reads every frame of a chunk that holds more than one', () => {
    const decoder = new SseDecoder();
    const frames = decoder.push('event: sources\ndata: {"a":1}\n\nevent: done\ndata: {"b":2}\n\n');

    expect(frames).toHaveLength(2);
    expect(frames[0].event).toBe('sources');
    expect(frames[1].event).toBe('done');
  });

  it('understands the carriage returns of a stream that uses CRLF', () => {
    const decoder = new SseDecoder();
    const frames = decoder.push('event: token\r\ndata: hello\r\n\r\n');

    expect(frames).toEqual([{ event: 'token', data: 'hello' }]);
  });

  it('joins several data lines with a newline', () => {
    const decoder = new SseDecoder();
    const frames = decoder.push('event: token\ndata: first\ndata: second\n\n');

    expect(frames[0].data).toBe('first\nsecond');
  });

  it('ignores a comment', () => {
    const decoder = new SseDecoder();
    const frames = decoder.push(': keep-alive\n\nevent: token\ndata: hello\n\n');

    expect(frames).toHaveLength(1);
    expect(frames[0].data).toBe('hello');
  });

  it('hands over the last frame when the stream ends without its blank line', () => {
    const decoder = new SseDecoder();

    expect(decoder.push('event: done\ndata: {"type":"done"}')).toEqual([]);
    expect(decoder.flush()).toEqual([{ event: 'done', data: '{"type":"done"}' }]);
  });

  it('has nothing to hand over when the stream ends on a frame boundary', () => {
    const decoder = new SseDecoder();
    decoder.push('event: done\ndata: {"type":"done"}\n\n');

    expect(decoder.flush()).toEqual([]);
  });
});
