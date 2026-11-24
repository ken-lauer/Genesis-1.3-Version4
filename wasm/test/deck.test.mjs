import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { getLatticeValue, getNamelistValue, setLatticeValue, setNamelistValue } from "../demo/deck.js";

const root = new URL("../../examples/", import.meta.url);
const deck = readFileSync(new URL("Example1-SteadyState/Example1.in", root), "utf8");
const lattice = readFileSync(new URL("Example1-SteadyState/Example1.lat", root), "utf8");

for (const [namelist, key, expected] of [
    ["beam", "current", 3000],
    ["beam", "ex", 4e-7],
    ["field", "power", 5e3],
    ["setup", "gamma0", 11357.82],
    ["beam", "missing", null],
    ["nosuch", "current", null],
]) {
    test(`reads ${namelist}/${key}`, () => assert.equal(getNamelistValue(deck, namelist, key), expected));
}

test("sets one key and leaves the rest of the deck alone", () => {
    const out = setNamelistValue(deck, "beam", "current", 1500);
    assert.equal(getNamelistValue(out, "beam", "current"), 1500);
    assert.equal(out.replace("current=1500", "current=3000"), deck);
});

test("does not touch the same key in another namelist", () => {
    // &field and &setup come before &beam; only &field has `power`
    const out = setNamelistValue(deck, "field", "power", 10);
    assert.equal(getNamelistValue(out, "field", "power"), 10);
    assert.equal(getNamelistValue(out, "beam", "current"), 3000);
});

test("adds a missing key inside the namelist", () => {
    const out = setNamelistValue(deck, "setup", "npart", 2048);
    assert.equal(getNamelistValue(out, "setup", "npart"), 2048);
    assert.equal(getNamelistValue(out, "beam", "npart"), null);
});

test("reads and sets a lattice element value", () => {
    assert.equal(getLatticeValue(lattice, "UND", "aw"), 0.84853);
    const out = setLatticeValue(lattice, "UND", "aw", 0.85);
    assert.equal(getLatticeValue(out, "UND", "aw"), 0.85);
    assert.equal(getLatticeValue(out, "UND", "nwig"), 266);
    assert.equal(getLatticeValue(out, "QF", "k1"), 2);
});
