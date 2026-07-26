import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("./static/link.js", import.meta.url), "utf8");
const populateSource = source.match(/async function populatePlants[\s\S]*?\n}\nfunction cleanObs/)[0]
  .replace(/\nfunction cleanObs$/, "");

let resolvePlants;
const picker = { value: "09", innerHTML: "unchanged" };
const state = { loadToken: 1 };
const context = {
  state,
  document: {
    getElementById(id) {
      if (id === "link-plant") return picker;
      if (id === "plant-select") return { value: "09" };
      return null;
    },
  },
  fetchJson: () => new Promise((resolve) => { resolvePlants = resolve; }),
};
vm.runInNewContext(`${populateSource}; globalThis.populatePlants = populatePlants;`, context);

const token = state.loadToken;
const staleLoad = context.populatePlants(token);
state.loadToken += 1;
resolvePlants({ plants: [{ plant: "02", n_dates: 4 }] });
const loaded = await staleLoad;

assert.equal(loaded, false);
assert.equal(picker.value, "09");
assert.equal(picker.innerHTML, "unchanged");
let cloudLoads = 0;
if (loaded && token === state.loadToken) cloudLoads += 1;
assert.equal(cloudLoads, 0);

const renumberSource = source.match(/function chainBaseHeight[\s\S]*?\n}\n\n\/\/ ---------- load \+ save ----------/)[0]
  .replace(/\n\n\/\/ ---------- load \+ save ----------$/, "");
const renumberState = {
  chains: [{ obs: { d1: 1 } }, { obs: { d1: 2 } }],
  allViews: {
    d1: {
      otype: [2, 2],
      leafid: [1, 2],
      geometry: { attributes: { position: { getZ: (i) => [20, 5][i] } } },
    },
  },
};
let status = "";
const rankContext = {
  state: renumberState,
  refreshColors() {},
  renderUi() {},
  setStatus(value) { status = value; },
};
vm.runInNewContext(`${renumberSource}; globalThis.renumberByHeight = renumberByHeight;`, rankContext);
rankContext.renumberByHeight();

assert.deepEqual(renumberState.chains.map((chain) => chain.obs.d1), [2, 1]);
assert.match(status, /click Save/);
