/**
 * Split a display equation at `\qquad` separators outside braces and environments,
 * so that its relations can wrap onto new lines on narrow screens.
 */
export function splitTop(s: string): string[] {
  const parts: string[] = [];
  let depth = 0, env = 0, last = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '\\') {
      if (s.startsWith('\\begin', i)) env++;
      else if (s.startsWith('\\end', i)) env--;
      else if (depth === 0 && env === 0 && s.startsWith('\\qquad', i) && !/[a-zA-Z]/.test(s[i + 6] ?? '')) {
        parts.push(s.slice(last, i)); last = i + 6; i += 5; continue;
      }
      i++;
      continue;
    }
    if (c === '{') depth++;
    else if (c === '}') depth--;
  }
  parts.push(s.slice(last));
  return parts.map((p) => p.trim()).filter(Boolean);
}
