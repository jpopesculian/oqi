import CodeMirror from '@uiw/react-codemirror';

import { qasm } from './qasm';

interface Props {
  value: string;
  onChange: (value: string) => void;
}

export function SourceEditor({ value, onChange }: Props) {
  return (
    <CodeMirror
      value={value}
      onChange={onChange}
      extensions={[qasm]}
      theme="dark"
      height="100%"
      className="editor"
    />
  );
}
