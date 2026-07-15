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
