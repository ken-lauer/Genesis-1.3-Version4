// Read and rewrite single values in Genesis input decks and lattice files, so
// that the sliders and the text boxes stay two views of the same deck.

const NUMBER = String.raw`[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?`;
const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function block(text, namelist) {
    const m = new RegExp(String.raw`^[ \t]*&${escape(namelist)}[ \t]*\r?\n([\s\S]*?)^[ \t]*&end`, "m").exec(text);
    return m && { start: m.index + m[0].indexOf(m[1]), body: m[1] };
}

const keyLine = (key) => new RegExp(String.raw`^([ \t]*${escape(key)}[ \t]*=[ \t]*)(${NUMBER})`, "m");

/** Value of `key` in the first `&namelist` block, or null if absent or not a plain number. */
export function getNamelistValue(text, namelist, key) {
    const b = block(text, namelist);
    const m = b && keyLine(key).exec(b.body);
    return m ? Number(m[2]) : null;
}

/** Deck text with `key` in the first `&namelist` block set to `value`; the key is added if missing. */
export function setNamelistValue(text, namelist, key, value) {
    const b = block(text, namelist);
    if (!b) return text;
    const body = keyLine(key).test(b.body) ? b.body.replace(keyLine(key), `$1${value}`) : `${b.body}${key}=${value}\n`;
    return text.slice(0, b.start) + body + text.slice(b.start + b.body.length);
}

const latticeKey = (element, key) =>
    new RegExp(String.raw`^([ \t]*${escape(element)}[ \t]*:[^;]*?\b${escape(key)}[ \t]*=[ \t]*)(${NUMBER})`, "m");

/** Value of `key` on lattice element `element`, or null. */
export function getLatticeValue(text, element, key) {
    const m = latticeKey(element, key).exec(text);
    return m ? Number(m[2]) : null;
}

/** Lattice text with `key` on element `element` set to `value`. */
export function setLatticeValue(text, element, key, value) {
    return text.replace(latticeKey(element, key), `$1${value}`);
}
