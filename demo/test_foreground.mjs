import assert from "node:assert/strict";
import { frontmostIndices } from "./static/foreground.js";

const cells = Int32Array.from([10, 10, 12, 15]);
const depths = Float32Array.from([4, 4.005, 5, 7]);
assert.deepEqual(frontmostIndices([0, 1, 2, 3], cells, depths, 0.01, 10), [0, 1, 3]);
