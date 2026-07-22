import { StateEffect, StateField } from '@codemirror/state';
import {
  Decoration,
  type DecorationSet,
  EditorView,
} from '@codemirror/view';
import CodeMirror, { type ReactCodeMirrorRef } from '@uiw/react-codemirror';
import { useEffect, useRef } from 'react';

import { qasm } from './qasm';
import type { ErrorSpan } from './runner';

interface Props {
  value: string;
  onChange: (value: string) => void;
  errorSpan: ErrorSpan | null;
}

/** Set (or clear, with `null`) the highlighted error range, in char offsets. */
const setErrorRange = StateEffect.define<{ from: number; to: number } | null>();

const errorMark = Decoration.mark({ class: 'cm-error-span' });

/** Holds the error highlight; maps through edits so it stays anchored. */
const errorSpanField = StateField.define<DecorationSet>({
  create() {
    return Decoration.none;
  },
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const effect of tr.effects) {
      if (effect.is(setErrorRange)) {
        deco =
          effect.value && effect.value.to > effect.value.from
            ? Decoration.set([
                errorMark.range(effect.value.from, effect.value.to),
              ])
            : Decoration.none;
      }
    }
    return deco;
  },
  provide: (field) => EditorView.decorations.from(field),
});

/** UTF-8 byte offset → CodeMirror (UTF-16) character offset. */
function byteToChar(source: string, byteOffset: number): number {
  const bytes = new TextEncoder().encode(source);
  return new TextDecoder().decode(bytes.slice(0, byteOffset)).length;
}

export function SourceEditor({ value, onChange, errorSpan }: Props) {
  const editorRef = useRef<ReactCodeMirrorRef>(null);

  useEffect(() => {
    const view = editorRef.current?.view;
    if (!view) return;
    const range = errorSpan
      ? { from: byteToChar(value, errorSpan.start), to: byteToChar(value, errorSpan.end) }
      : null;
    view.dispatch({ effects: setErrorRange.of(range) });
    // Keyed on errorSpan only: convert against the source at error time, then
    // let the field map the decoration through any later edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [errorSpan]);

  return (
    <CodeMirror
      ref={editorRef}
      value={value}
      onChange={onChange}
      extensions={[qasm, errorSpanField]}
      theme="dark"
      height="100%"
      className="editor"
    />
  );
}
