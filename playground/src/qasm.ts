import { StreamLanguage, type StringStream } from '@codemirror/language';

// Keyword sets lifted from the oqi lexer (lex/src/lib.rs).
const KEYWORDS = new Set([
  'OPENQASM', 'include', 'defcalgrammar', 'def', 'cal', 'defcal', 'gate',
  'extern', 'box', 'let', 'break', 'continue', 'if', 'else', 'end', 'return',
  'for', 'while', 'in', 'switch', 'case', 'default', 'nop', 'input', 'output',
  'const', 'readonly', 'mutable', 'gphase', 'inv', 'pow', 'ctrl', 'negctrl',
  'durationof', 'delay', 'reset', 'measure', 'barrier', 'pragma',
]);

const TYPES = new Set([
  'qreg', 'creg', 'qubit', 'bool', 'bit', 'int', 'uint', 'float', 'angle',
  'complex', 'array', 'void', 'duration', 'stretch', 'waveform', 'port',
  'frame',
]);

const ATOMS = new Set(['true', 'false', 'pi', 'π', 'tau', 'τ', 'euler', 'ℇ']);

interface State {
  inBlockComment: boolean;
}

function token(stream: StringStream, state: State): string | null {
  if (state.inBlockComment) {
    if (stream.match(/^.*?\*\//)) {
      state.inBlockComment = false;
    } else {
      stream.skipToEnd();
    }
    return 'comment';
  }
  if (stream.eatSpace()) return null;
  if (stream.match('//')) {
    stream.skipToEnd();
    return 'comment';
  }
  if (stream.match('/*')) {
    state.inBlockComment = true;
    return 'comment';
  }
  if (stream.match(/^"([^"\\]|\\.)*"/)) return 'string';
  // Numbers, with optional timing (ns|us|µs|ms|s|dt) or imaginary suffix.
  if (
    stream.match(/^0[bB][01_]+/) ||
    stream.match(/^0[oO][0-7_]+/) ||
    stream.match(/^0[xX][0-9a-fA-F_]+/) ||
    stream.match(/^(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?\s*(ns|us|µs|ms|s|dt|im)?\b/)
  ) {
    return 'number';
  }
  if (stream.match(/^\$\d+/)) return 'atom'; // hardware qubits
  if (stream.match(/^@\w[\w.]*/)) return 'meta'; // annotations
  if (stream.match('#dim')) return 'keyword';
  if (stream.match(/^[\p{L}_][\p{L}\p{N}_]*/u)) {
    const word = stream.current();
    if (KEYWORDS.has(word)) return 'keyword';
    if (TYPES.has(word)) return 'typeName';
    if (ATOMS.has(word)) return 'atom';
    return 'variableName';
  }
  stream.next();
  return null;
}

export const qasm = StreamLanguage.define<State>({
  name: 'openqasm',
  startState: () => ({ inBlockComment: false }),
  token,
  languageData: {
    commentTokens: { line: '//', block: { open: '/*', close: '*/' } },
  },
});
