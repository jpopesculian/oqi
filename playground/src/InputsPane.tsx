import { json } from '@codemirror/lang-json';
import CodeMirror from '@uiw/react-codemirror';

interface Props {
  value: string;
  onChange: (value: string) => void;
  error: string | null;
}

export function InputsPane({ value, onChange, error }: Props) {
  return (
    <div className="inputs-pane">
      <div className="pane-title">Inputs</div>
      <CodeMirror
        value={value}
        onChange={onChange}
        extensions={[json()]}
        theme="dark"
        height="100%"
        className="editor"
        basicSetup={{ foldGutter: false }}
      />
      {error !== null && <div className="inline-error">{error}</div>}
    </div>
  );
}
