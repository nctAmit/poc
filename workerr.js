/* meshWorker.js */
/* CPU-bound pipeline:
   - Parse GeoTIFF
   - Build positions/colors/indices (center-fan triangles, 4 tris per cell)
   - Compute normals
   Returns typed arrays via Transferables
*/

self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ----- small utils -----
function degToRad(d) { return d * Math.PI / 180; }

// Per-vertex normals accumulated from triangle faces
function computeNormals(positions /* Float64Array */, indices /* Uint32Array */) {
  const normals = new Float32Array(positions.length); // zero-initialized
  for (let i = 0; i < indices.length; i += 3) {
    const ia = indices[i] * 3;
    const ib = indices[i + 1] * 3;
    const ic = indices[i + 2] * 3;

    const ax = positions[ia],     ay = positions[ia + 1],     az = positions[ia + 2];
    const bx = positions[ib],     by = positions[ib + 1],     bz = positions[ib + 2];
    const cx = positions[ic],     cy = positions[ic + 1],     cz = positions[ic + 2];

    // edges from A
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;

    // face normal = e1 x e2
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;

    normals[ia]     += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib]     += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic]     += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }
  // normalize
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i], ny = normals[i + 1], nz = normals[i + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      normals[i]     = nx / len;
      normals[i + 1] = ny / len;
      normals[i + 2] = nz / len;
    }
  }
  return normals;
}

/**
 * Build center-fan per-cell geometry with one flat color per cell (taken from top-left).
 * Input positionsAll are laid out as [lat, lon, height] per vertex (to match your pipeline).
 * We duplicate vertices per cell so color is not interpolated across cells.
 */
function buildCenterFanGrid(positionsAll, colorsAll, indexMap, width, height) {
  // Count valid quads (all four corners exist)
  let quadCount = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;
      if (indexMap[tl] !== -1 && indexMap[tr] !== -1 && indexMap[bl] !== -1 && indexMap[br] !== -1) {
        quadCount++;
      }
    }
  }

  // For each quad we create 5 vertices (TL, BL, BR, TR, C) and 4 triangles (12 indices)
  const VERTS_PER_QUAD = 5;
  const TRIS_PER_QUAD = 4;
  const outPositions = new Float64Array(quadCount * VERTS_PER_QUAD * 3);
  const outColors    = new Uint8Array (quadCount * VERTS_PER_QUAD * 4);
  const outIndices   = new Uint32Array(quadCount * TRIS_PER_QUAD  * 3);

  let vtx = 0;  // vertex cursor
  let idx = 0;  // index cursor

  // helper to copy one source vertex into outputs with the given RGBA
  function addCopy(srcIndex, r, g, b, a) {
    const pBase = srcIndex * 3;
    const o = vtx * 3;
    outPositions[o]     = positionsAll[pBase];
    outPositions[o + 1] = positionsAll[pBase + 1];
    outPositions[o + 2] = positionsAll[pBase + 2];

    const co = vtx * 4;
    outColors[co]     = r;
    outColors[co + 1] = g;
    outColors[co + 2] = b;
    outColors[co + 3] = a;

    return vtx++; // returns index of the vertex we just wrote
  }

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;

      const iTL = indexMap[tl];
      const iTR = indexMap[tr];
      const iBL = indexMap[bl];
      const iBR = indexMap[br];

      if (iTL === -1 || iTR === -1 || iBL === -1 || iBR === -1) continue;

      // Cell color from top-left original vertex
      const cBase = iTL * 4;
      const r = colorsAll[cBase];
      const g = colorsAll[cBase + 1];
      const b = colorsAll[cBase + 2];
      const a = colorsAll[cBase + 3];

      // Corner duplicates (so color is flat per cell & we control indices)
      const vTL = addCopy(iTL, r, g, b, a);
      const vBL = addCopy(iBL, r, g, b, a);
      const vBR = addCopy(iBR, r, g, b, a);
      const vTR = addCopy(iTR, r, g, b, a);

      // Center vertex: average of the 4 corners in your (lat, lon, h) space
      const tlP = iTL * 3, blP = iBL * 3, brP = iBR * 3, trP = iTR * 3;
      const cLat = (positionsAll[tlP] + positionsAll[blP] + positionsAll[brP] + positionsAll[trP]) * 0.25;
      const cLon = (positionsAll[tlP + 1] + positionsAll[blP + 1] + positionsAll[brP + 1] + positionsAll[trP + 1]) * 0.25;
      const cH   = (positionsAll[tlP + 2] + positionsAll[blP + 2] + positionsAll[brP + 2] + positionsAll[trP + 2]) * 0.25;

      const vC = vtx; // index for center
      outPositions[vC * 3]     = cLat;
      outPositions[vC * 3 + 1] = cLon;
      outPositions[vC * 3 + 2] = cH;

      outColors[vC * 4]     = r;
      outColors[vC * 4 + 1] = g;
      outColors[vC * 4 + 2] = b;
      outColors[vC * 4 + 3] = a;
      vtx++;

      // Build 4 triangles around the center (counter-clockwise winding):
      // (C, TL, BL), (C, BL, BR), (C, BR, TR), (C, TR, TL)
      outIndices[idx++] = vC;  outIndices[idx++] = vTL; outIndices[idx++] = vBL;
      outIndices[idx++] = vC;  outIndices[idx++] = vBL; outIndices[idx++] = vBR;
      outIndices[idx++] = vC;  outIndices[idx++] = vBR; outIndices[idx++] = vTR;
      outIndices[idx++] = vC;  outIndices[idx++] = vTR; outIndices[idx++] = vTL;
    }
  }

  return { outPositions, outColors, outIndices };
}

// Optional: WebMercator to lon/lat (not used below but kept if needed)
function webMercatorToLonLat(x, y) {
  const R = 6378137.0;
  const lon = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI);
  return [lon, lat];
}

self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type === 'PROCESS_TIFF') {
      const { arrayBuffer, deltaZ = 0 } = payload;

      const tiff   = await GeoTIFF.fromArrayBuffer(arrayBuffer);
      const image  = await tiff.getImage();
      const rasters = await image.readRasters(); // [0]=elev, [1]=R, [2]=G, [3]=B
      const width  = image.getWidth();
      const height = image.getHeight();

      const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY] (assumed degrees)
      const minX = bbox[0], minY = bbox[1], maxX = bbox[2], maxY = bbox[3];
      const lonStep = (maxX - minX) / (width  - 1 || 1);
      const latStep = (maxY - minY) / (height - 1 || 1);

      const elev = rasters[0];
      const R = rasters[1], G = rasters[2], B = rasters[3];

      // Build sparse vertex map (skip nodata/zero elev so you can keep “hole” transparency behavior)
      const positions = [];
      const colors    = [];
      const indexMap  = new Int32Array(width * height);
      indexMap.fill(-1);

      let nextIdx = 0;

      for (let y = 0; y < height; y++) {
        const lat = maxY - y * latStep;
        for (let x = 0; x < width; x++) {
          const lon = minX + x * lonStep;
          const i = y * width + x;

          let h = (elev[i] === NODATA_ELEV) ? 0 : Number(elev[i]);
          let r = Number(R[i]), g = Number(G[i]), b = Number(B[i]);
          let a = 255;

          if (h === 0) {
            // visualize “no elev” as semi-transparent white
            h = 0;
            a = 100;
            r = 255; g = 255; b = 255;
          } else {
            h += Number(deltaZ || 0);
          }

          if (h > 0) {
            // NOTE: positions are stored as [lat, lon, height] to match your pipeline
            positions.push(lat, lon, h);
            colors.push(r, g, b, a);
            indexMap[i] = nextIdx++;
          }
        }
      }

      const posArr = new Float64Array(positions);
      const colArr = new Uint8Array(colors);

      // Center-fan (smooth triangulation), single color per cell from TL
      const { outPositions, outColors, outIndices } =
        buildCenterFanGrid(posArr, colArr, indexMap, width, height);

      const normals = computeNormals(outPositions, outIndices);

      self.postMessage({
        id,
        ok: true,
        result: {
          width,
          height,
          bbox, // [minX, minY, maxX, maxY]
          positions: outPositions.buffer,
          colors: outColors.buffer,
          indices: outIndices.buffer,
          normals: normals.buffer
        }
      }, [outPositions.buffer, outColors.buffer, outIndices.buffer, normals.buffer]);

    } else {
      throw new Error(`Unknown worker message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: (err && err.message) || String(err)
    });
  }
};
