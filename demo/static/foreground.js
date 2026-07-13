export function frontmostIndices(indices, cells, depths, tolerance, columns) {
  const front = new Map();
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] < 0) continue;
    const old = front.get(cells[i]);
    if (old === undefined || depths[i] < old) front.set(cells[i], depths[i]);
  }
  return indices.filter((i) => {
    if (cells[i] < 0) return false;
    const x = cells[i] % columns;
    let nearest = Infinity;
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        if (x + dx < 0 || x + dx >= columns) continue;
        nearest = Math.min(nearest, front.get(cells[i] + dx + dy * columns) ?? Infinity);
      }
    }
    return depths[i] <= nearest + tolerance;
  });
}
